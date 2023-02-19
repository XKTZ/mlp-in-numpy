from typing import List, Dict, Union, Tuple

import numpy as np
from abc import abstractmethod, ABC
class Layer:
    _last: np.ndarray
    _grad: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]
    _id: int

    def __init__(self):
        self._grad = ()

    def set_id(self, id: int):
        self._id = id

    def id(self) -> int:
        return self._id

    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def gradient(self, error: np.ndarray) -> np.ndarray:
        """
        Given error = dC / da_l
        Then, gradient is: dC / da_l * a_(l-1)^T
        Returns dc/da_(l-1) = da_l / da_(l-1) * error by chain rule (denominator layout)
        Example: if it is sigmoid, then return sigmoid(last) * (1 - sigmoid(last)) * dC / da
        :return dC / d(a - 1)
        """
        pass

    def get_gradient(self) -> Union[Tuple[np.ndarray, ...], Tuple[float, ...]]:
        return self._grad

    @abstractmethod
    def move(self, delta: Union[Tuple[np.ndarray, ...], Tuple[float, ...]]):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    def load(self, o):
        pass

    def save(self) -> object:
        return {}

    def __call__(self, x: np.ndarray) -> np.ndarray:
        self._last = x
        return self.forward(x)


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
