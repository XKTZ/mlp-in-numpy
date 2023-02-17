from typing import List, Dict, Union, Tuple

import numpy as np
from abc import abstractmethod, ABC
from ml.loss import Loss
from ml.layer import Layer


class Optimizer:
    def __init__(self):
        pass

    @abstractmethod
    def step(self, layers: List[Layer]):
        pass

    @abstractmethod
    def optimize(self, layer: Layer):
        pass


class NeuralNetwork:
    _layers: List[Layer]
    _loss: Loss
    _optim: Optimizer
    _last: np.ndarray

    @staticmethod
    def _to_batch_at_back(x: np.ndarray) -> np.ndarray:
        return x.transpose((*(idx + 1 for idx, _ in enumerate(x.shape[1:])), 0))

    @staticmethod
    def _to_batch_at_front(x: np.ndarray) -> np.ndarray:
        return x.transpose((-1, *(idx for idx, _ in enumerate(x.shape[1:]))))

    def __init__(self, layers: List[Layer], loss: Loss, optim: Optimizer):
        for i in range(len(layers)):
            layers[i].set_id(i)

        self._layers = layers
        self._loss = loss
        self._optim = optim

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = NeuralNetwork._to_batch_at_back(x)
        for layer in self._layers:
            x = layer(x)
        self._last = x
        return self._to_batch_at_front(x)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def loss(self, targ: np.ndarray) -> np.ndarray:
        loss = self._loss(self._last, NeuralNetwork._to_batch_at_back(targ))
        return loss

    def backward(self):
        error = self._loss.grad_of()
        for layer in reversed(self._layers):
            error = layer.gradient(error)

    def step(self):
        self._optim.step(self._layers)

    def zero_grad(self):
        for layer in self._layers:
            layer.zero_grad()

    def state_dict(self):
        p = {}
        for i in self._layers:
            p[i.id()] = i.save()
        return p

    def load_state_dict(self, st: Dict[str, object]):
        for layer in self._layers:
            layer.load(st[str(layer.id())])
