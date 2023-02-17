from enum import Enum
from typing import Union, Type, Dict, List

import numpy as np
from abc import abstractmethod

from ml.activation import Softmax, Sigmoid
from ml.layer import Layer


class Loss:
    _target: np.ndarray
    _out: np.ndarray
    _before: List[Layer]

    def __init__(self, before: List[Layer]):
        if before is None:
            before = []
        self._before = before

    def loss_of(self, x: np.ndarray, targ: np.ndarray) -> np.ndarray:
        for layer in self._before:
            x = layer(x)
        self._target = targ
        self._out = x
        return self._loss(x, targ)

    @abstractmethod
    def _loss(self, x: np.ndarray, targ: np.ndarray) -> np.ndarray:
        pass

    def grad_of(self) -> np.ndarray:
        error = self._gradient()
        for layer in reversed(self._before):
            error = layer.gradient(error)
        return error

    @abstractmethod
    def _gradient(self) -> np.ndarray:
        """
        Returns the dC / dx, where x is the output, C is the loss
        It is only for THIS loss calculation, not for before
        :return:
        """
        pass

    def __call__(self, x: np.ndarray, targ: np.ndarray) -> np.ndarray:
        return self.loss_of(x, targ)


class MSELoss(Loss):
    def __init__(self, before=None):
        super(MSELoss, self).__init__(before)

    def _loss(self, x: np.ndarray, targ: np.ndarray) -> np.ndarray:
        return np.mean(np.square(np.linalg.norm((self._out - self._target), axis=0)))

    def _gradient(self) -> np.ndarray:
        return 2 * (self._out - self._target) / self._target.shape[-1]


class BinaryCrossEntropyLoss(Loss):
    class Before(Enum):
        SOFTMAX = 1
        SIGMOID = 2

    _beforeLayerClass: Dict[int, Type[Layer]] = {
        Before.SOFTMAX: Softmax,
        Before.SIGMOID: Sigmoid
    }

    @staticmethod
    def _get_layer_before(before: Union[int, Layer, List[Layer]]):
        if isinstance(before, int):
            return BinaryCrossEntropyLoss._beforeLayerClass[before]()
        elif isinstance(before, Layer):
            return [before]
        elif isinstance(before, list):
            return before
        else:
            return []

    _before: List[Layer]

    def __init__(self, before: Union[int, Layer, List[Layer]] = None):
        super(BinaryCrossEntropyLoss, self).__init__(BinaryCrossEntropyLoss._get_layer_before(before))

    def _loss(self, x: np.ndarray, targ: np.ndarray) -> np.ndarray:
        return np.mean(- targ * np.log(x) + (1 - targ) * np.log(1. - x))

    def _gradient(self) -> np.ndarray:
        return (1. - self._target) / (1. - self._out) - self._target / self._out
