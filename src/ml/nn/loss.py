from enum import Enum
from typing import Union, Type, Dict, List, Callable

from ml.device import device as np
from abc import abstractmethod

from ml.nn.activation import Softmax, Sigmoid
from ml.nn.base import Loss, Layer


class MSELoss(Loss):
    def __init__(self, before=None):
        super(MSELoss, self).__init__(before)

    def _loss(self, x: np.ndarray, targ: np.ndarray) -> np.ndarray:
        return np.mean(np.square(np.linalg.norm(self._out - self._target, axis=1)))

    def _gradient(self) -> np.ndarray:
        return 2 * (self._out - self._target) / self._target.shape[0]


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

    epsilon: float
    clipper: Callable[[np.ndarray], np.ndarray]

    def __init__(self, before: Union[int, Layer, List[Layer]] = None, epsilon: float = 1e-9):
        super(BinaryCrossEntropyLoss, self).__init__(BinaryCrossEntropyLoss._get_layer_before(before))
        self.epsilon = epsilon
        self.clipper = lambda x: np.clip(x, self.epsilon, 1. - epsilon)

    def _loss(self, x: np.ndarray, targ: np.ndarray) -> np.ndarray:
        x = self.clipper(x)
        return np.mean(- targ * np.log(x) + (1 - targ) * np.log(1. - x))

    def _gradient(self) -> np.ndarray:
        return self.clipper(1. - self._target) / self.clipper(1. - self._out) - self.clipper(
            self._target) / self.clipper(self._out)
