import math
from typing import Tuple, Union

from ml.device import device as np
from ml.nn.base import Layer


class Linear(Layer):
    in_channel: int
    out_channel: int
    bias: bool

    mat: np.ndarray
    b: np.ndarray

    def __init__(self, in_channel: int, out_channel: int, bias: bool = True):
        super(Linear, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.bias = bias

        if self.bias:
            self.b = np.random.uniform(- 1 / math.sqrt(in_channel), 1 / math.sqrt(in_channel), size=out_channel)
        else:
            self.b = np.zeros(out_channel)
        self.mat = np.random.normal(0, math.sqrt(2 / in_channel), size=(out_channel, in_channel))

    def get_parameter(self) -> Tuple[Union[np.ndarray, float], ...]:
        return self.mat, self.b

    def forward(self, x: np.ndarray) -> np.ndarray:
        return x.dot(self.mat.T) + np.full((x.shape[0], self.out_channel), self.b)

    def gradient(self, error: np.ndarray) -> np.ndarray:
        self._grad = (self._grad[0] + (error.T.dot(self._last)),
                      self._grad[1] + np.sum(error, axis=0))
        return error.dot(self.mat)

    def update(self, delta: Tuple[Union[np.ndarray, float], ...]):
        self.mat += delta[0]
        if self.bias:
            self.b += delta[1]

    def zero_grad(self):
        self._grad = (0.0, 0.0)

    def load(self, o):
        self.mat = np.array(o["w"])
        self.b = np.array(o["b"])
        self.bias = o["biased"]

    def save(self) -> object:
        return {
            "w": self.mat.tolist(),
            "b": self.b.tolist(),
            "biased": self.bias
        }
