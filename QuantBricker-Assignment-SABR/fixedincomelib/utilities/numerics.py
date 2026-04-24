import copy
import numpy as np
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

class InterpMethod(Enum):

    PIECEWISE_CONSTANT_LEFT_CONTINUOUS = 'PIECEWISE_CONSTANT_LEFT_CONTINUOUS'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'InterpMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class ExtrapMethod(Enum):
    
    FLAT = 'FLAT'
    LINEAR = 'LINEAR'

    @classmethod
    def from_string(cls, value: str) -> 'ExtrapMethod':
        if not isinstance(value, str):
            raise TypeError("value must be a string")
        try:
            return cls(value.upper())
        except ValueError:
            raise ValueError(f"Invalid token: {value}")

    def to_string(self) -> str:
        return self.value

class Interpolator1D(ABC):

    def __init__(self,
                 axis1 : np.ndarray, 
                 values : np.ndarray, 
                 interpolation_method : InterpMethod,
                 extrpolation_method : ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrpolation_method
        self.length_ = len(self.axis1)

    @abstractmethod
    def interpolate(self, x : float) -> float:
        pass

    @abstractmethod
    def integrate(self, start_x : float, end_x : float):
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x : float):
        pass

    @abstractmethod
    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):
        pass
    
    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_
    
    @property
    def values(self) -> np.ndarray:
        return self.values_
    
    @property
    def length(self) -> int:
        return self.length_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()
    
    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()

class Interpolator1DPCP(Interpolator1D):

    def __init__(self, axis1: np.ndarray, values: np.ndarray, extrpolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, values, InterpMethod.LINEAR, extrpolation_method)
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float) -> float:
        
        if x < self.axis1[0]:
            # flat left extrapolation
            return self.values[0]
        if x >= self.axis1[-1]:
            # flat right extrpolation
            return self.values[-1]
        
        for i in range(len(self.axis1) - 1):
            if x >= self.axis1[i] and x < self.axis1[i+1]:
                return self.values[i+1]
    
    def gradient_wrt_ordinate(self, x : float):
        
        grad = np.zeros(self.length, dtype=float)

        if x < self.axis1[0]:
            # flat left extrapolation
            grad[0] = 1.0
            return grad
        if x >= self.axis1[-1]:
            # flat right extrpolation
            grad[-1] = 1.0
            return grad

        for i in range(len(self.axis1) - 1):
            if x >= self.axis1[i] and x < self.axis1[i+1]:
                grad[i+1] = 1
        return grad

    def integrate(self, start_x : float, end_x : float):
        
        if self.length == 1:
            return (end_x - start_x) * self.values[0]

        acc = 0.
        start_fixed = False
        # two pointers
        for i in range(self.length + 1):
            interval_s, interval_e, interval_v = None, None, None
            if i == 0:
                interval_s, interval_e = -np.inf, self.axis1[0]
                interval_v = self.values[0]
            elif i == self.length:
                interval_s, interval_e = self.axis1[-1], np.inf
                interval_v = self.values[-1]
            else:
                interval_s, interval_e = self.axis1[i-1], self.axis1[i]
                interval_v = self.values[i]
            # if both of them are in the same interval
            if start_x >= interval_s and start_x < interval_e and \
                end_x >= interval_s and end_x < interval_e:
                acc = (end_x - start_x) * interval_v
                break
            # if start hits this interval
            if not start_fixed and start_x >= interval_s and start_x < interval_e:
                acc += (interval_e - start_x) * interval_v
                start_fixed = True
                continue
            # start already fixed, end hits this interval
            if start_fixed:
                if end_x >= interval_s and end_x < interval_e:
                    # if hit, wrap up
                    acc += (end_x - interval_s) * interval_v
                    break
                else:
                    #  otherwise, count in the whole interval
                    acc += (interval_e - interval_s) * interval_v

        return acc

    def gradient_of_integrated_value_wrt_ordinate(self, start_x : float, end_x : float):

        grad = np.zeros(self.length, dtype=float)
        
        if self.length == 1:
            grad[0] = end_x - start_x
            return grad

        # acc = 0.
        start_fixed = False
        # two pointers
        for i in range(self.length + 1):            
            interval_s, interval_e, interval_i = None, None, None
            if i == 0:
                interval_s, interval_e, interval_i = 0, self.axis1[0], 0
            elif i == self.length:
                interval_s, interval_e, interval_i = self.axis1[-1], np.inf, self.length - 1                 
            else:
                interval_s, interval_e, interval_i = self.axis1[i-1], self.axis1[i], i
            # if both of them are in the same interval
            if start_x >= interval_s and start_x < interval_e and \
                end_x >= interval_s and end_x < interval_e:
                grad[interval_i] += (end_x - start_x)
                break
            # if start hits this interval
            if not start_fixed and start_x >= interval_s and start_x < interval_e:                
                grad[interval_i] += (interval_e - start_x)
                start_fixed = True
                continue
            # start already fixed, end hits this interval
            if start_fixed:
                if end_x >= interval_s and end_x < interval_e:
                    # if hit, wrap up                    
                    grad[interval_i] += (end_x - interval_s)
                    break
                else:
                    #  otherwise, count in the whole interval                    
                    grad[interval_i] += (interval_e - interval_s)

        return grad

class InterpolatorFactory:

    @staticmethod
    def create_1d_interpolator(axis1 : np.ndarray | List, 
                               values : np.ndarray | List, 
                               interpolation_method : InterpMethod,
                               extrpolation_method : ExtrapMethod):


        axis1_ = copy.deepcopy(axis1)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(values_.shape) == 1
        assert len(axis1_) == len(values_)
        assert np.all(np.diff(axis1_) >= 0)
    
        if interpolation_method == InterpMethod.PIECEWISE_CONSTANT_LEFT_CONTINUOUS:
            return Interpolator1DPCP(axis1_, values_, extrpolation_method)
        else:
            raise Exception('Currently only support PCP interpolation')
    
    @staticmethod
    def create_2d_interpolator(axis1 : np.ndarray | List, 
                               axis2 : np.ndarray | List, 
                               values : np.ndarray | List, 
                               interpolation_method : InterpMethod,
                               extrpolation_method : ExtrapMethod):

        axis1_ = copy.deepcopy(axis1)
        axis2_ = copy.deepcopy(axis2)
        values_ = copy.deepcopy(values)
        if isinstance(axis1_, list):
            axis1_ = np.array(axis1_)
        if isinstance(axis2_, list):
            axis2_ = np.array(axis2_)
        if isinstance(values_, list):
            values_ = np.array(values_)
        assert len(axis1_.shape) == 1 and len(axis2_.shape) == 1 and len(values_.shape) == 2
        assert len(axis1_) == values_.shape[0] and len(axis2_) == values_.shape[1]
        assert np.all(np.diff(axis1_) >= 0) and np.all(np.diff(axis2_) >= 0)

        if interpolation_method == InterpMethod.LINEAR and extrpolation_method == ExtrapMethod.FLAT:
            return Interpolator2DLinearFlat(axis1_, axis2_, values_, interpolation_method, extrpolation_method)
        else:
            raise Exception('Currently only support linear interpolation with flat extrapolation')

class Interpolator2D(ABC):

    def __init__(self,
                 axis1 : np.ndarray, 
                 axis2 : np.ndarray, 
                 values : np.ndarray, 
                 interpolation_method : InterpMethod,
                 extrpolation_method : ExtrapMethod) -> None:

        self.axis1_ = axis1
        self.axis2_ = axis2
        self.values_ = values
        self.interp_method_ = interpolation_method
        self.extrap_method_ = extrpolation_method
        self.size1_ = len(self.axis1_)
        self.size2_ = len(self.axis2_)

    @abstractmethod
    def interpolate(self, x : float, y : float) -> float:
        pass

    @abstractmethod
    def gradient_wrt_ordinate(self, x : float, y : float) -> np.ndarray:
        pass
    
    @property
    def axis1(self) -> np.ndarray:
        return self.axis1_

    @property
    def axis2(self) -> np.ndarray:
        return self.axis2_
    
    @property
    def values(self) -> np.ndarray:
        return self.values_
    
    @property
    def size1(self) -> int:
        return self.size1_

    @property
    def size2(self) -> int:
        return self.size2_

    @property
    def interp_method(self) -> str:
        return self.interp_method_.to_string()
    
    @property
    def extrap_method(self) -> str:
        return self.extrap_method_.to_string()


class Interpolator2DLinearFlat(Interpolator2D):

    def __init__(self, axis1: np.ndarray, axis2 : np.ndarray, values: np.ndarray, interpolation_method : InterpMethod, extrpolation_method: ExtrapMethod) -> None:
        super().__init__(axis1, axis2, values, interpolation_method, extrpolation_method)
        assert self.interp_method_ == InterpMethod.LINEAR
        assert self.extrap_method_ == ExtrapMethod.FLAT

    def interpolate(self, x: float, y : float) -> float:
        if x < self.axis1[0]:
            x = self.axis1[0]
        if x >= self.axis1[-1]:
            x = self.axis1[-1]
        if y < self.axis2[0]:
            y = self.axis2[0]
        if y >= self.axis2[-1]:
            y = self.axis2[-1]
        
        if x == self.axis1[-1]:
            i = self.size1 - 2
        else:
            for k in range(self.size1 - 1):
                if x >= self.axis1[k] and x < self.axis1[k + 1]:
                    i = k
                    break

        if y == self.axis2[-1]:
            j = self.size2 - 2
        else:
            for k in range(self.size2 - 1):
                if y >= self.axis2[k] and y < self.axis2[k + 1]:
                    j = k
                    break
        
        x1, x2 = self.axis1[i], self.axis1[i + 1]
        y1, y2 = self.axis2[j], self.axis2[j + 1]
        Q11 = self.values[i][j]
        Q12 = self.values[i][j + 1]
        Q21 = self.values[i + 1][j]
        Q22 = self.values[i + 1][j + 1]
        f1 = Q11 + (Q21 - Q11) * (x - x1) / (x2 - x1)
        f2 = Q12 + (Q22 - Q12) * (x - x1) / (x2 - x1)

        return f1 + (f2 - f1) * (y - y1) / (y2 - y1)
    
    def gradient_wrt_ordinate(self, x : float, y : float) -> np.ndarray:
        grad = np.zeros((self.size1 * self.size2), dtype=float)
        if x < self.axis1[0]:
            x = self.axis1[0]
        if x >= self.axis1[-1]:
            x = self.axis1[-1]
        if y < self.axis2[0]:
            y = self.axis2[0]
        if y >= self.axis2[-1]:
            y = self.axis2[-1]
        
        if x == self.axis1[-1]:
            i = self.size1 - 2
        else:
            for k in range(self.size1 - 1):
                if x >= self.axis1[k] and x < self.axis1[k + 1]:
                    i = k
                    break

        if y == self.axis2[-1]:
            j = self.size2 - 2
        else:
            for k in range(self.size2 - 1):
                if y >= self.axis2[k] and y < self.axis2[k + 1]:
                    j = k
                    break
        
        wx1 = (x-self.axis1[i]) / (self.axis1[i + 1] - self.axis1[i])
        wx2 = 1 - wx1
        wy1 = (y-self.axis2[j]) / (self.axis2[j + 1] - self.axis2[j])
        wy2 = 1 - wy1

        grad[i * self.size2 + j] = wx2 * wy2
        grad[i * self.size2 + (j + 1)] = wx2 * wy1
        grad[(i + 1) * self.size2 + j] = wx1 * wy2
        grad[(i + 1) * self.size2 + (j + 1)] = wx1 * wy1

        return grad