from abc import abstractmethod
from typing import Tuple, Union

from ml.nn.base import Layer
from ml.device import device as np


class BasicTrainNormalizationLayer(Layer):
    train_mode: bool

    def __init__(self):
        super().__init__()
        self.train_mode = True

    def train(self):
        self.train_mode = True

    def eval(self):
        self.train_mode = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.train_mode:
            return self.train_mode_forward(x)
        else:
            return x

    def gradient(self, error: np.ndarray) -> np.ndarray:
        if self.train_mode:
            return self.train_mode_gradient(error)
        else:
            return error

    @abstractmethod
    def train_mode_forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def train_mode_gradient(self, error: np.ndarray) -> np.ndarray:
        pass


class Dropout(BasicTrainNormalizationLayer):
    prob: float
    last_mask: np.ndarray

    def __init__(self, prob: float):
        super().__init__()
        assert prob <= 1
        self.prob = prob

    def train_mode_forward(self, x: np.ndarray) -> np.ndarray:
        self.last_mask = np.random.choice(a=[0, 1], p=[self.prob, 1 - self.prob], size=x.shape)
        return x * self.last_mask

    def train_mode_gradient(self, error: np.ndarray) -> np.ndarray:
        return error * self.last_mask

    def zero_grad(self):
        self._grad = ()

    def update(self, delta: Tuple[Union[np.ndarray, float], ...]):
        pass
