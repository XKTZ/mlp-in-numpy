from typing import List, Dict, Union, Tuple

from abc import abstractmethod, ABC

from mlnp.device import device as np


class Layer:
    _last: np.ndarray
    _grad: Tuple[Union[np.ndarray, float], ...]
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
        pass

    def get_parameter(self) -> Tuple[Union[np.ndarray, float], ...]:
        return ()

    def get_gradient(self) -> Tuple[Union[np.ndarray, float], ...]:
        return self._grad

    @abstractmethod
    def update(self, delta: Tuple[Union[np.ndarray, float], ...]):
        pass

    @abstractmethod
    def zero_grad(self):
        pass

    def eval(self):
        pass

    def train(self):
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

    def train(self):
        for layer in self._before:
            layer.train()

    def eval(self):
        for layer in self._before:
            layer.eval()

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

    def step(self, layers: List[Layer]):
        for layer in layers:
            self.optimize(layer)

    @abstractmethod
    def optimize(self, layer: Layer):
        pass


class NeuralNetwork:
    _layers: List[Layer]
    _loss: Loss
    _optim: Optimizer
    _last: np.ndarray

    def __init__(self, layers: List[Layer], loss: Loss, optim: Optimizer):
        for i in range(len(layers)):
            layers[i].set_id(i)

        self._layers = layers
        self._loss = loss
        self._optim = optim

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = x
        for layer in self._layers:
            x = layer(x)
        self._last = x
        return x

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def set_optimizer(self, optim: Optimizer):
        self._optim = optim

    def loss(self, targ: np.ndarray) -> np.ndarray:
        loss = self._loss(self._last, targ)
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

    def train(self):
        for layer in self._layers:
            layer.train()
        self._loss.train()

    def eval(self):
        for layer in self._layers:
            layer.eval()
        self._loss.eval()

    def state_dict(self):
        p = {}
        for i in self._layers:
            p[i.id()] = i.save()
        return p

    def load_state_dict(self, st: Dict):
        for layer in self._layers:
            layer.load(st[layer.id()])
