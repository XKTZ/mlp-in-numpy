from typing import Union, Tuple

from ml.nn.base import Layer
from ml.device import device as np


class Reshape(Layer):

    last_shape: Tuple[int, ...]
    shape: Tuple[int, ...]

    def __init__(self, shape: Tuple[int, ...]):
        super().__init__()
        self.shape = shape

    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.reshape(x, (x.shape[0], *self.shape))

    def gradient(self, error: np.ndarray) -> np.ndarray:
        return np.reshape(error, self._last.shape)

    def update(self, delta: Tuple[Union[np.ndarray, float], ...]):
        pass

    def zero_grad(self):
        pass


class Flatten(Layer):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        shape = x.shape
        return np.reshape(x, (x.shape[0], -1))

    def gradient(self, error: np.ndarray) -> np.ndarray:
        return error.reshape(self._last.shape)

    def move(self, delta: Tuple[Union[np.ndarray, float], ...]):
        pass

    def update(self, delta: Tuple[Union[np.ndarray, float], ...]):
        pass

    def zero_grad(self):
        self._grad = ()
