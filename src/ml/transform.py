from typing import Union, Tuple

from ml.neural import Layer
import numpy as np


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        return np.reshape(x, (np.product(shape[:-1]), shape[-1]))

    def gradient(self, error: np.ndarray) -> np.ndarray:
        shape = self._last.shape
        return error.reshape(shape)

    def move(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        pass

    def zero_grad(self):
        self._grad = (0.0,)