from typing import Any, Dict

from fixedincomelib.analytics.european_options import CallOrPut, SimpleMetrics
from fixedincomelib.analytics.sabr import SabrMetrics
from fixedincomelib.date import *
from fixedincomelib.sabr import sabr_parameters
from fixedincomelib.sabr.utilities import SABRPriceAndRiskCalculator
from fixedincomelib.valuation import *
from fixedincomelib.sabr.sabr_model import *
from fixedincomelib.yield_curve import *
from fixedincomelib.product import *
from fixedincomelib.sabr.sabr_model import SABRModel


class ValuationEngineRFRCapletFloorlet(ValuationEngineProduct):
    def __init__(
        self,
        model: SABRModel,
        valuation_parameters_collection: ValuationParametersCollection,
        product: ProductRFRCapletFloorlet,
        request: ValuationRequest,
    ):
        super().__init__(model, valuation_parameters_collection, product, request)

        self.currency_ = product.currency
        self.expiry_date_ = product.expiry_date
        self.effective_date_ = product.effective_date_
        self.termination_date_ = product.termination_date
        self.pay_date_ = product.payment_date
        self.sign_ = 1.0 if product.long_or_short == LongOrShort.LONG else -1.0
        self.notional_ = product.notional
        self.strike_ = product.strike
        self.overnight_index_ = product.on_index_
        self.cap_or_floor_ = product.cap_or_floor_
        self.accrual_ = product.accrual_
        self.call_or_put_ = CallOrPut.CALL if self.cap_or_floor_ == CapOrFloor.CAP else CallOrPut.PUT

        self.vpc_: ValuationParametersCollection = valuation_parameters_collection
        assert self.vpc_.has_vp_type(FundingIndexParameter._vp_type)
        self.funding_vp_: FundingIndexParameter = self.vpc_.get_vp_from_build_method_collection(
            FundingIndexParameter._vp_type
        )
        self.funding_index_ = self.funding_vp_.get_funding_index(self.currency_)

        tortd = TermOrTerminationDate(self.termination_date_.ISO())
        self.index_engine_ = ValuationEngineAnalyticsOvernightIndex(
            self.model_,
            self.vpc_,
            self.overnight_index_,
            self.effective_date_,
            tortd,
            CompoundingMethod.COMPOUND,
        )

        self.df_ = 1.0
        self.forward_ = None
        self.option_value_ = 0.0
        self.time_to_expiry_ = 0.0
        self.tenor_ = self.accrual_
        self.sabr_result_ = {}
        self.first_order_risk_ = {}

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.value_ = 0.0
        self.cash_ = 0.0
        self.df_ = 1.0
        self.forward_ = None
        self.option_value_ = 0.0
        self.time_to_expiry_ = 0.0
        self.tenor_ = self.accrual_
        self.sabr_result_ = {}

        if self.value_date_ > self.pay_date_:
            return

        scaler = self.sign_ * self.notional_

        # Forward compounded overnight rate over [effective date, termination date].
        self.index_engine_.calculate_value()
        self.forward_ = self.index_engine_.value()

        # On payment date, the payoff is cash today and no discounting is required.
        if self.value_date_ == self.pay_date_:
            if self.call_or_put_ == CallOrPut.CALL:
                self.option_value_ = max(self.forward_ - self.strike_, 0.0)
            else:
                self.option_value_ = max(self.strike_ - self.forward_, 0.0)
            self.cash_ = scaler * self.accrual_ * self.option_value_
            self.value_ = self.cash_
            return

        self.df_ = self.model_.discount_factor(self.funding_index_, self.pay_date_)

        # After expiry, the payoff is already fixed and only intrinsic value remains.
        if self.value_date_ >= self.expiry_date_:
            if self.call_or_put_ == CallOrPut.CALL:
                self.option_value_ = max(self.forward_ - self.strike_, 0.0)
            else:
                self.option_value_ = max(self.strike_ - self.forward_, 0.0)
        else:
            self.time_to_expiry_ = accrued(self.value_date_, self.expiry_date_)
            self.tenor_ = self.accrual_
            self.sabr_calculator_ = SABRPriceAndRiskCalculator(
                self.model_,
                self.vpc_,
                IndexRegistry().get("SOFR-1B-CAPFLOOR"),
                self.forward_,
                self.strike_,
                self.time_to_expiry_,
                self.tenor_,
                self.call_or_put_,
                calc_risk=True,
            )
            self.sabr_result_ = self.sabr_calculator_.calculate_value()
            self.option_value_ = self.sabr_result_[SimpleMetrics.PV]

        self.value_ = scaler * self.df_ * self.accrual_ * self.option_value_

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False):
        if self.forward_ is None:
            self.calculate_value()

        local_grad = []
        self.model_.resize_gradient(local_grad)

        if self.value_date_ > self.pay_date_:
            if gradient is None:
                gradient = []
            self.model_.resize_gradient(gradient)
            if not accumulate:
                for i in range(len(gradient)):
                    gradient[i] = 0.0 * gradient[i]
            self.first_order_risk_ = gradient
            return

        total_scaler = scaler * self.sign_ * self.notional_ * self.accrual_

        if self.value_date_ == self.pay_date_:
            if self.call_or_put_ == CallOrPut.CALL:
                doption_dforward = 1.0 if self.forward_ > self.strike_ else 0.0
            else:
                doption_dforward = -1.0 if self.forward_ < self.strike_ else 0.0
            self.index_engine_.calculate_risk(local_grad, total_scaler * doption_dforward, True)

        elif self.value_date_ >= self.expiry_date_:
            dv_ddf = total_scaler * self.option_value_
            funding_model: YieldCurve = self.model_
            funding_model.discount_factor_gradient_wrt_state(
                self.funding_index_, self.pay_date_, local_grad, dv_ddf, True
            )

            if self.call_or_put_ == CallOrPut.CALL:
                doption_dforward = 1.0 if self.forward_ > self.strike_ else 0.0
            else:
                doption_dforward = -1.0 if self.forward_ < self.strike_ else 0.0
            dv_dforward = total_scaler * self.df_ * doption_dforward
            self.index_engine_.calculate_risk(local_grad, dv_dforward, True)

        else:
            dv_ddf = total_scaler * self.option_value_
            funding_model: YieldCurve = self.model_
            funding_model.discount_factor_gradient_wrt_state(
                self.funding_index_, self.pay_date_, local_grad, dv_ddf, True
            )

            dv_dforward = total_scaler * self.df_ * self.sabr_result_[SimpleMetrics.DELTA]
            self.index_engine_.calculate_risk(local_grad, dv_dforward, True)

            dv_dsabr = total_scaler * self.df_
            self.sabr_calculator_.calculate_risk(local_grad, dv_dsabr)

        if gradient is None:
            gradient = []
        self.model_.resize_gradient(gradient)

        if accumulate:
            for i in range(len(gradient)):
                gradient[i] += local_grad[i]
        else:
            gradient[:] = local_grad

        self.first_order_risk_ = gradient

    def create_cash_flows_report(self) -> CashflowsReport:
        this_cf = CashflowsReport()
        this_cf.add_row(
            0,
            self.product_._product_type,
            self.val_engine_type(),
            self.notional_,
            self.sign_,
            self.pay_date_,
            self.value_ / self.df_ if self.df_ != 0.0 else 0.0,
            self.value_,
            self.df_,
            fixing_date=self.expiry_date_,
            start_date=self.effective_date_,
            end_date=self.termination_date_,
            accrued=self.accrual_,
            index_or_fixed=self.overnight_index_.name(),
            index_value=self.forward_,
        )
        return this_cf

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report


class ValuationEngineRFRCapFloor(ValuationEngineProduct):
    """SABR valuation engine for an RFR cap/floor as a strip of RFR caplets/floorlets."""

    def __init__(
        self,
        model: SABRModel,
        valuation_parameters_collection: ValuationParametersCollection,
        product: ProductRFRCapFloor,
        request: ValuationRequest,
    ):
        super().__init__(model, valuation_parameters_collection, product, request)

        self.currency_ = product.currency
        self.caplets_ = product.caplets_
        self.engines_ = [
            ValuationEngineRFRCapletFloorlet(
                model,
                valuation_parameters_collection,
                caplet,
                request,
            )
            for caplet in self.caplets_
        ]
        self.first_order_risk_ = {}

    @classmethod
    def val_engine_type(cls) -> str:
        return cls.__name__

    def calculate_value(self):
        self.value_ = 0.0
        self.cash_ = 0.0

        for engine in self.engines_:
            engine.calculate_value()
            self.value_ += engine.value_
            self.cash_ += engine.cash_

    def calculate_first_order_risk(self, gradient=None, scaler: float = 1.0, accumulate: bool = False):
        if gradient is None:
            gradient = []
        self.model_.resize_gradient(gradient)

        if not accumulate:
            for i in range(len(gradient)):
                gradient[i] = 0.0 * gradient[i]

        for engine in self.engines_:
            if engine.forward_ is None:
                engine.calculate_value()
            engine.calculate_first_order_risk(gradient, scaler, True)

        self.first_order_risk_ = gradient

    def create_cash_flows_report(self) -> CashflowsReport:
        report = CashflowsReport()

        for engine in self.engines_:
            if engine.forward_ is None:
                engine.calculate_value()
            report.add_row(
                0,
                engine.product_._product_type,
                engine.val_engine_type(),
                engine.notional_,
                engine.sign_,
                engine.pay_date_,
                engine.value_ / engine.df_ if engine.df_ != 0.0 else 0.0,
                engine.value_,
                engine.df_,
                fixing_date=engine.expiry_date_,
                start_date=engine.effective_date_,
                end_date=engine.termination_date_,
                accrued=engine.accrual_,
                index_or_fixed=engine.overnight_index_.name(),
                index_value=engine.forward_,
            )

        return report

    def get_value_and_cash(self) -> PVCashReport:
        report = PVCashReport(self.currency_)
        report.set_pv(self.currency_, self.value_)
        report.set_cash(self.currency_, self.cash_)
        return report


_SABR_ENGINE_MAP = {
    ProductRFRCapletFloorlet._product_type: ValuationEngineRFRCapletFloorlet,
    ProductRFRCapFloor._product_type: ValuationEngineRFRCapFloor,
}

for prod_type, eng_cls in _SABR_ENGINE_MAP.items():
    ValuationEngineProductRegistry().register(
        (SABRModel._model_type.to_string(), prod_type, AnalyticValParam._vp_type),
        eng_cls,
    )
