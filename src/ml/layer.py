import numpy as np

from typing import Union, Tuple

from abc import abstractmethod

class Layer:
    _last: np.ndarray
    _grad: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]
    _id: int

    def __init__(self):
        self._grad = ()

    def set_id(self, id: int):
        self._id = id

    def id(self) -> int:
        return self._id

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, error: np.ndarray) -> np.ndarray:
        """
        Given error = dC / da_l
        Then, gradient is: dC / da_l * a_(l-1)^T
        Returns dc/da_(l-1) = da_l / da_(l-1) * error by chain rule (denominator layout)
        Example: if it is sigmoid, then return sigmoid(last) * (1 - sigmoid(last)) * dC / da
        :return dC / d(a - 1)
        """
        pass

    def get_gradient(self) -> Union[Tuple[np.ndarray, ...], Tuple[float, ...]]:
        return self._grad

    @abstractmethod
    def move(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    def load(self, o):
        pass

    def save(self) -> object:
        return {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._last = x
        return self.forward(x)