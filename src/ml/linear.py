from typing import Tuple, Union

import numpy as np
from ml.layer import Layer


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
            self.b = np.random.rand(out_channel)
        else:
            self.b = np.zeros(out_channel)
        self.mat = np.zeros((out_channel, in_channel))

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.mat.dot(x) + np.full((x.shape[-1], self.out_channel), self.b).T

    def gradient(self, error: np.ndarray) -> np.ndarray:
        self._grad = (self._grad[0] + error.dot(self._last.T), np.mean(self._grad[1] + error, axis=1))
        return self.mat.T.dot(error)

    def move(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        self.mat += delta[0]
        if self.bias:
            self.b += delta[1]

    def zero_grad(self):
        self._grad = (0.0, 0.0)

    def load(self, o):
        self.mat = np.array(o["w"])
        self.b = np.array(o["b"])

    def save(self) -> object:
        return {
            "w": self.mat.tolist(),
            "b": self.b.tolist()
        }
