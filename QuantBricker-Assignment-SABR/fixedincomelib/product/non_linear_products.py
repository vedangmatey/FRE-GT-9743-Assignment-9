from calendar import Calendar, c
from enum import Enum
from turtle import st
from webbrowser import get
import pandas as pd
from typing import List, Optional, Union
from dataclasses import dataclass
import QuantLib as ql
import numpy as np
from regex import E
from fixedincomelib.date.utilities import add_period, frequency_from_period
from fixedincomelib.market.basics import *
from fixedincomelib.market.registries import IndexRegistry, DataConventionRegistry
from fixedincomelib.market.data_conventions import (
    CompoundingMethod,
    DataConventionRFRCapFloor,
    DataConventionRFRCapletFloorlet,
)
from fixedincomelib.market.indices import FXIndex
from fixedincomelib.market import (
    Currency,
    AccrualBasis,
    BusinessDayConvention,
    HolidayConvention,
    DataConventionRegistry,
    IndexRegistry
)
from fixedincomelib.product.utilities import LongOrShort, PayOrReceive
from fixedincomelib.product.product_interfaces import (
    Product,
    ProductVisitor,
    ProductBuilderRegistry,
)
from fixedincomelib.date import Date, Period, TermOrTerminationDate, make_schedule, accrued
from fixedincomelib.product.product_portfolio import ProductPortfolio


class CapOrFloor(Enum):
    CAP = "CAP"
    FLOOR = "FLOOR"

    @classmethod
    def from_string(cls, value: str) -> "CapOrFloor":
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class ProductRFRCapletFloorlet(Product):

    _version = 1
    _product_type = "PRODUCT_RFR_CAPLET_FLOORLET"

    def __init__(
        self,
        effective_date: Date,
        # expiry_date: Date,
        expiry_offset: Union[str,Period],
        term_or_termination_date: TermOrTerminationDate,
        payment_date: Date,
        on_index: str,
        strike: float,
        notional: float,
        cap_or_floor: CapOrFloor,
        accrual_basis: AccrualBasis,
        long_or_short: LongOrShort = LongOrShort.LONG
    ) -> None:

        super().__init__()

        self.on_index_str_ = on_index
        self.on_index_ = IndexRegistry().get(on_index)

        self.first_date_ = self.effective_date_ = effective_date
        self.payment_date_ = payment_date
        self.last_date_ = self.payment_date_
        self.expiry_offset_ = Period(expiry_offset) if isinstance(expiry_offset, str) else expiry_offset

        calendar = self.on_index_.fixingCalendar()
        self.expiry_date_ = Date(
            calendar.advance(
                self.effective_date_,
                self.expiry_offset_,
                self.on_index_.businessDayConvention(),
            )
        )
        if term_or_termination_date.is_term():
            calendar = self.on_index_.fixingCalendar()
            self.termination_date_ = Date(
                calendar.advance(
                    self.effective_date_,
                    term_or_termination_date.get_term(),
                    self.on_index_.businessDayConvention(),
                )
            )
        else:
            self.termination_date_ = term_or_termination_date.get_date()
        
        self.strike_ = strike
        self.notional_ = notional
        self.long_or_short_ = long_or_short
        self.accrual_basis_ = accrual_basis
        self.cap_or_floor_ = cap_or_floor
        self.currency_ = Currency(self.on_index_.currency().code())
        self.accrual_ = accrued(self.effective_date_, self.termination_date_, self.accrual_basis_)

    @property
    def effective_date(self) -> Date:
        return self.effective_date_
    
    @property
    def expiry_date(self) -> Date:
        return self.expiry_date_
    
    @property
    def expiry_offset(self) -> Period:
        return self.expiry_offset_
    
    @property
    def termination_date(self) -> Date:
        return self.termination_date_
    
    @property
    def payment_date(self) -> Date:
        return self.payment_date_
    
    @property
    def on_index_str(self) -> str:
        return self.on_index_str_
    
    @property
    def on_index(self) -> ql.QuantLib.OvernightIndex:
        return self.on_index_

    @property
    def strike(self) -> float:
        return self.strike_
    
    @property
    def notional(self) -> float:
        return self.notional_
    
    @property
    def cap_or_floor(self) -> CapOrFloor:
        return self.cap_or_floor_
    
    @property
    def accrual_basis(self) -> AccrualBasis:
        return self.accrual_basis_
    
    @property
    def accrual(self) -> float:
        return self.accrual_

    @property
    def currency(self):
        return self.currency_

    @property
    def long_or_short(self):
        return self.long_or_short_

    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)

    def serialize(self) -> dict:
        content = {}
        content["VERSION"] = self._version
        content["TYPE"] = self._product_type
        content["EFFECTIVE_DATE"] = self.effective_date.ISO()
        content["EXPIRY_OFFSET"] = str(self.expiry_offset_)
        content["EXPIRY_DATE"] = self.expiry_date_.ISO()
        content["TERMINATION_DATE"] = self.termination_date.ISO()
        content["PAYMENT_DATE"] = self.payment_date.ISO()
        content["ON_INDEX"] = self.on_index_str
        content["NOTIONAL"] = self.notional
        content["STRIKE"] = self.strike
        content["LONG_OR_SHORT"] = self.long_or_short.to_string().upper()
        content["CAP_OR_FLOOR"] = self.cap_or_floor.to_string().upper()
        content["ACCRUAL_BASIS"] = self.accrual_basis.value_str
        
        return content

    @classmethod
    def deserialize(cls, input_dict) -> "ProductRFRCapletFloorlet":
        effective_date = Date(input_dict["EFFECTIVE_DATE"])
        expiry_offset = input_dict["EXPIRY_OFFSET"]
        termination_date = TermOrTerminationDate(input_dict["TERMINATION_DATE"])
        payment_date = Date(input_dict["PAYMENT_DATE"])
        on_index = input_dict["ON_INDEX"]
        notional = float(input_dict["NOTIONAL"])
        strike = float(input_dict["STRIKE"])
        long_or_short = LongOrShort.from_string(input_dict["LONG_OR_SHORT"])
        cap_or_floor = CapOrFloor.from_string(input_dict["CAP_OR_FLOOR"])
        accrual_basis = AccrualBasis(input_dict["ACCRUAL_BASIS"])

        return cls(effective_date, 
                   expiry_offset,
                   termination_date, 
                   payment_date, 
                   on_index,
                   strike,
                   notional,
                   cap_or_floor,
                   accrual_basis,
                   long_or_short,
                   )

class ProductRFRCapFloor(Product):

    _version = 1
    _product_type = "PRODUCT_RFR_CAP_FLOOR"

    def __init__(
        self,
        effective_date: Date,
        term_or_termination_date: TermOrTerminationDate,
        on_index: str,
        strike: float,
        notional: float,
        cap_or_floor: CapOrFloor,
        accrual_period: Period,
        accrual_basis: AccrualBasis,
        payment_offset: Period,
        payment_business_day_convention: BusinessDayConvention,
        payment_holiday_convention: HolidayConvention,
        long_or_short: LongOrShort = LongOrShort.LONG,
        business_day_convention: Optional[BusinessDayConvention] = BusinessDayConvention("F"),
        holiday_convention: Optional[HolidayConvention] = HolidayConvention("USGS")
    ) -> None:
        super().__init__()

        self.on_index_str_ = on_index
        self.on_index_ = IndexRegistry().get(on_index)

        self.first_date_ = self.effective_date_ = effective_date
        if term_or_termination_date.is_term():
            calendar = self.on_index_.fixingCalendar()
            self.termination_date_ = Date(
                calendar.advance(
                    self.effective_date_,
                    term_or_termination_date.get_term(),
                    self.on_index_.businessDayConvention(),
                )
            )
        else:
            self.termination_date_ = term_or_termination_date.get_date()
        
        self.strike_ = strike
        self.notional_ = notional
        self.cap_or_floor_ = cap_or_floor
        self.accrual_period_ = accrual_period
        self.accrual_basis_ = accrual_basis
        self.payment_offset_ = payment_offset
        self.payment_business_day_convention_ = payment_business_day_convention
        self.payment_holiday_convention_ = payment_holiday_convention
        self.long_or_short_ = long_or_short
        self.business_day_convention_ = business_day_convention
        self.holiday_convention_ = holiday_convention
        self.currency_ = Currency(self.on_index_.currency().code())
        
        schedule = make_schedule(
            start_date=self.effective_date_,
            end_date=self.termination_date_,
            accrual_period=self.accrual_period_,
            holiday_convention=self.holiday_convention_,
            business_day_convention=self.business_day_convention_,
            accrual_basis=self.accrual_basis_,
            payment_offset=self.payment_offset_,
            payment_business_day_convention=self.payment_business_day_convention_,
            payment_holiday_convention=self.payment_holiday_convention_,
        )
        self.caplets_ = []
        # TODO:
        # Build one ProductRFRCapletFloorlet for each accrual period in the schedule,
        # and append it to self.caplets_.
        for _, row in schedule.iterrows():
            pass
        
        if len(self.caplets_) > 0:
            self.last_date_ = self.caplets_[-1].payment_date
        else:
            self.last_date_ = self.termination_date_
    
    @property
    def effective_date(self) -> Date:
        return self.effective_date_
    
    @property
    def termination_date(self) -> Date:
        return self.termination_date_
    
    @property
    def on_index_str(self) -> str:
        return self.on_index_str_
    
    @property
    def on_index(self) -> ql.QuantLib.OvernightIndex:
        return self.on_index_
    
    @property
    def strike(self) -> float:
        return self.strike_
    
    @property
    def notional(self) -> float:
        return self.notional_
    
    @property
    def cap_or_floor(self) -> CapOrFloor:
        return self.cap_or_floor_
    
    @property
    def accrual_period(self) -> Period:
        return self.accrual_period_
    
    @property
    def accrual_basis(self) -> AccrualBasis:
        return self.accrual_basis_
    
    @property
    def payment_offset(self) -> Period:
        return self.payment_offset_
    
    @property
    def payment_business_day_convention(self) -> BusinessDayConvention:
        return self.payment_business_day_convention_
    
    @property
    def payment_holiday_convention(self) -> HolidayConvention:
        return self.payment_holiday_convention_
    
    @property
    def long_or_short(self) -> LongOrShort:
        return self.long_or_short_
    
    @property
    def currency(self) -> Currency:
        return self.currency_
    
    def num_caplets(self) -> int:
        return len(self.caplets_)
    
    def caplets(self, i: int) -> ProductRFRCapletFloorlet:
        return self.caplets_[i]
    
    def accept(self, visitor: ProductVisitor):
        return visitor.visit(self)
    
    def serialize(self) -> dict:
        content = {}
        content["VERSION"] = self._version
        content["TYPE"] = self._product_type
        content["EFFECTIVE_DATE"] = self.effective_date.ISO()
        # content["EXPIRY_DATE"] = self.expiry_date_.ISO()
        content["TERMINATION_DATE"] = self.termination_date.ISO()
        content["ON_INDEX"] = self.on_index_str
        content["STRIKE"] = self.strike
        content["NOTIONAL"] = self.notional
        content["CAP_OR_FLOOR"] = self.cap_or_floor.to_string().upper()
        content["ACCRUAL_PERIOD"] = str(self.accrual_period)
        content["ACCRUAL_BASIS"] = self.accrual_basis.value_str
        content["PAYMENT_OFFSET"] = str(self.payment_offset)
        content["PAYMENT_BUSINESS_DAY_CONVENTION"] = self.payment_business_day_convention.value_str
        content["PAYMENT_HOLIDAY_CONVENTION"] = self.payment_holiday_convention.value_str
        content["LONG_OR_SHORT"] = self.long_or_short.to_string().upper()
        content["BUSINESS_DAY_CONVENTION"] = self.business_day_convention_.value_str
        content["HOLIDAY_CONVENTION"] = self.holiday_convention_.value_str
        return content
    
    @classmethod
    def deserialize(cls, input_dict) -> "ProductRFRCapFloor":
        effective_date = Date(input_dict["EFFECTIVE_DATE"])
        # expiry_date = Date(input_dict["EXPIRY_DATE"])
        termination_date = TermOrTerminationDate(input_dict["TERMINATION_DATE"])
        on_index = input_dict["ON_INDEX"]
        strike = float(input_dict["STRIKE"])
        notional = float(input_dict["NOTIONAL"])
        cap_or_floor = CapOrFloor.from_string(input_dict["CAP_OR_FLOOR"])
        accrual_period = Period(input_dict["ACCRUAL_PERIOD"])
        accrual_basis = AccrualBasis(input_dict["ACCRUAL_BASIS"])
        payment_offset = Period(input_dict["PAYMENT_OFFSET"])
        payment_business_day_convention = BusinessDayConvention(input_dict["PAYMENT_BUSINESS_DAY_CONVENTION"])
        payment_holiday_convention = HolidayConvention(input_dict["PAYMENT_HOLIDAY_CONVENTION"])
        long_or_short = LongOrShort.from_string(input_dict["LONG_OR_SHORT"])
        business_day_convention = BusinessDayConvention(input_dict["BUSINESS_DAY_CONVENTION"])
        holiday_convention = HolidayConvention(input_dict["HOLIDAY_CONVENTION"])

        return cls(effective_date, 
                #    expiry_date,
                   termination_date, 
                   on_index,
                   strike,
                   notional,
                   cap_or_floor,
                   accrual_period,
                   accrual_basis,
                   payment_offset,
                   payment_business_day_convention,
                   payment_holiday_convention,
                   long_or_short,
                   business_day_convention,
                   holiday_convention
                   )


# register
ProductBuilderRegistry().register(ProductRFRCapletFloorlet._product_type, ProductRFRCapletFloorlet)
ProductBuilderRegistry().register(ProductRFRCapFloor._product_type, ProductRFRCapFloor)

# support de-serilization
ProductBuilderRegistry().register(f"{ProductRFRCapletFloorlet._product_type}_DES", ProductRFRCapletFloorlet.deserialize)
ProductBuilderRegistry().register(f"{ProductRFRCapFloor._product_type}_DES", ProductRFRCapFloor.deserialize)