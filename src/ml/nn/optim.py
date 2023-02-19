from typing import List, Dict, Union, Tuple

from ml.device import device as np

from ml.nn.base import Optimizer, Layer


class GradientDescent(Optimizer):
    lr: float

    def __init__(self, lr: float):
        super(GradientDescent, self).__init__()
        self.lr = lr

    def step(self, layers: List[Layer]):
        for layer in layers:
            self.optimize(layer)

    def optimize(self, layer: Layer):
        layer.update(tuple(-g * self.lr for g in layer.get_gradient()))


class Annealing(Optimizer):
    t0: int
    t1: int
    t: int

    def __init__(self, t0: int, t1: int):
        super(Annealing, self).__init__()
        self.t0 = t0
        self.t1 = t1
        self.t = 0

    def step(self, layers: List[Layer]):
        self.t += 1
        for layer in layers:
            self.optimize(layer)

    def optimize(self, layer: Layer):
        layer.update(tuple(- g * (self.t0 / (self.t1 + self.t)) for g in layer.get_gradient()))


class SGD(Optimizer):
    _eta: float
    _gamma: float

    _past: Dict[int, Union[Tuple[np.ndarray, ...], Tuple[float, ...]]]

    def __init__(self, eta: float, gamma: float = 0.9):
        super(SGD, self).__init__()
        self._eta = eta
        self._gamma = gamma
        self._past = {}

    def optimize(self, layer: Layer):
        id = layer.id()

        # get gradient
        gs = layer.get_gradient()

        # if not visited gradient yet, set to 0
        if id not in self._past:
            self._past[id] = tuple(0.0 for _ in range(len(gs)))

        # create now
        now = tuple((-g * self._eta + p * self._gamma) for g, p in zip(gs, self._past[id]))

        # move a now
        layer.update(now)

        self._past[id] = now


class Adagrad(Optimizer):
    epsilon: float

    lr: float
    _past: Dict[int, Union[Tuple[np.ndarray, ...], Tuple[float, ...]]]

    def __init__(self, lr: float, epsilon: float = 1e-9):
        super().__init__()
        self.lr = lr
        self.epsilon = epsilon
        self._past = {}

    def optimize(self, layer: Layer):
        id = layer.id()

        grads = layer.get_gradient()

        if id not in self._past:
            self._past[id] = tuple(0. for _ in range(len(grads)))

        g_past = self._past[id]

        g_now = tuple(x ** 2 + y for x, y in zip(grads, g_past))

        self._past[id] = g_now

        lr = self.lr
        epsilon = self.epsilon

        layer.update(tuple(-lr / np.sqrt(y + epsilon) * x for x, y in zip(grads, g_now)))

