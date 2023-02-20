from typing import Union, Tuple

from ml.nn.base import Layer
from ml.device import device as np


class Flatten(Layer):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        return np.reshape(x, (np.product(shape[:-1]), shape[-1]))

    def gradient(self, error: np.ndarray) -> np.ndarray:
        shape = self._last.shape
        return error.reshape(shape)

    def move(self, delta: Tuple[Union[np.ndarray, float], ...]):
        pass

    def zero_grad(self):
        self._grad = (0.0,)