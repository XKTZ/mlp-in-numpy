from typing import Union, Tuple, Callable
from ml.nn.base import Layer
from ml.device import device as np


class Sigmoid(Layer):

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x)

    def gradient(self, error: np.ndarray) -> np.ndarray:
        y = self._sigmoid(self._last)
        return error * (y * (1. - y))

    def zero_grad(self):
        pass

    def update(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        pass


class ReLU(Layer):

    @staticmethod
    def _relu(x: np.ndarray) -> np.ndarray:
        return np.maximum(x, 0)

    @staticmethod
    def _drelu(x: np.ndarray) -> np.ndarray:
        return (x > 0) * 1.

    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return ReLU._relu(x)

    def gradient(self, error: np.ndarray) -> np.ndarray:
        return self._drelu(self._last) * error

    def zero_grad(self):
        pass

    def update(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        pass


class LeakyReLU(Layer):
    _alpha: float

    def __init__(self, alpha: float = 0.1):
        super(LeakyReLU, self).__init__()
        self._alpha = alpha

    def _leaky_relu(self, x: np.ndarray):
        return np.maximum(x * self._alpha, x)

    def _d_leaky_relu(self, x: np.ndarray):
        return np.where(x > 0, 1, self._alpha)

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self._leaky_relu(x)

    def gradient(self, error: np.ndarray) -> np.ndarray:
        return self._d_leaky_relu(self._last) * error

    def zero_grad(self):
        pass

    def update(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        pass


class Softmax(Layer):
    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        shiftx = x - np.max(x, axis=1).reshape(-1, 1)
        exps = np.exp(shiftx)
        return exps / np.sum(exps, axis=1).reshape(-1, 1)

    def __init__(self):
        super(Softmax, self).__init__()

    def forward(self, x: np.ndarray) -> np.ndarray:
        return Softmax._softmax(x)

    def gradient(self, error: np.ndarray) -> np.ndarray:
        last = self._last
        b, n = last.shape
        S = self._softmax(last)
        I = np.zeros((b, n, n))
        diag = np.arange(n)
        I[:, diag, diag] = S
        return np.einsum("ij,ijk->ik", error, -np.einsum("ij,ik->ijk", S, S) + I)

    def zero_grad(self):
        pass

    def update(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        pass
