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

    _past: Dict[int, Tuple[Union[np.ndarray, float], ...]]

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
    _past: Dict[int, Tuple[Union[np.ndarray, float], ...]]

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


class RMSProp(Optimizer):
    lr: float
    gamma: float
    epsilon: float

    _past: Dict[int, Tuple[Union[np.ndarray, float], ...]]

    def __init__(self, lr: float, gamma: float = 0.9, epsilon: float = 1e-9):
        super().__init__()
        self.lr = lr
        self._past = {}
        self.gamma = gamma
        self.epsilon = epsilon

    def optimize(self, layer: Layer):
        past = self._past
        lr, gamma, epsilon = self.lr, self.gamma, self.epsilon

        id = layer.id()
        grads = layer.get_gradient()

        if id not in past:
            past[id] = tuple(0. for _ in range(len(grads)))

        now = tuple(gamma * y + (1 - gamma) * (x ** 2) for x, y in zip(grads, past[id]))
        past[id] = now

        layer.update(tuple(-lr / np.sqrt(y + epsilon) * x for x, y in zip(grads, now)))


class AdaDelta(Optimizer):

    @staticmethod
    def _perform_expect(now: Tuple[Union[np.ndarray, float], ...], past: Tuple[Union[np.ndarray, float], ...],
                        gamma: float) -> Tuple[Union[np.ndarray, float], ...]:
        return tuple(x * (1 - gamma) + y * gamma for x, y in zip(now, past))

    lr: float
    gamma: float
    epsilon: float

    _expect_g: Dict[int, Tuple[Union[np.ndarray, float], ...]]
    _expect_theta: Dict[int, Tuple[Union[np.ndarray, float], ...]]

    def __init__(self, lr: float, gamma: float = 0.9, epsilon: float = 1e-9):
        super().__init__()
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon

        self._expect_g = {}
        self._expect_theta = {}

    def optimize(self, layer: Layer):
        id = layer.id()

        lr, gamma, epsilon = self.lr, self.gamma, self.epsilon

        grads = layer.get_gradient()
        params = layer.get_parameter()

        expect_g, expect_theta = self._expect_g, self._expect_theta

        if id not in expect_g:
            expect_g[id] = tuple(0. for _ in range(len(grads)))
        if id not in expect_theta:
            expect_theta[id] = tuple(0. for _ in range(len(grads)))

        grad_last = expect_g[id]
        theta_last = expect_theta[id]

        grad_now = self._perform_expect(grads, grad_last, gamma)
        theta_now = self._perform_expect(params, theta_last, gamma)

        expect_g[id] = grad_now
        expect_theta[id] = theta_now

        layer.update(
            tuple(- lr * (np.sqrt(x ** 2 + epsilon)) / (np.sqrt(y ** 2 + epsilon)) * g for g, x, y in
                  zip(grads, grad_now, theta_now))
        )


class Adam(Optimizer):
    @staticmethod
    def _perform(now: Tuple[Union[np.ndarray, float], ...], past: Tuple[Union[np.ndarray, float], ...],
                 beta: float) -> Tuple[Union[np.ndarray, float], ...]:
        return tuple(x * (1 - beta) + y * beta for x, y in zip(now, past))

    lr: float
    betas: Tuple[float, float]
    epsilon: float
    t: int

    _momentum: Dict[int, Tuple[Union[np.ndarray, float], ...]]
    _variance: Dict[int, Tuple[Union[np.ndarray, float], ...]]

    def __init__(self, lr: float, betas: Tuple[float, float] = (0.9, 0.999), epsilon: float = 1e-9):
        super().__init__()
        self.lr = lr
        self.betas = betas
        self.epsilon = epsilon
        self.t = 0
        self._momentum, self._variance = {}, {}

    def optimize(self, layer: Layer):
        self.t += 1

        id = layer.id()
        lr, betas, epsilon, t = self.lr, self.betas, self.epsilon, self.t
        beta1, beta2 = betas

        grads = layer.get_gradient()

        momentum = self._momentum
        variance = self._variance

        if id not in momentum:
            momentum[id] = tuple(0. for _ in range(len(grads)))
        if id not in variance:
            variance[id] = tuple(0. for _ in range(len(grads)))

        momentum_last = momentum[id]
        variance_last = variance[id]

        momentum_now = self._perform(grads, momentum_last, beta1)
        variance_now = self._perform(tuple(x ** 2 for x in grads), variance_last, beta2)

        momentum[id] = momentum_now
        variance[id] = variance_now

        momentum_hat = tuple(x / (1 - np.power(beta1, t)) for x in momentum_now)
        variance_hat = tuple(x / (1 - np.power(beta2, t)) for x in variance_now)

        layer.update(tuple(
            - lr * m / np.sqrt(v + epsilon) for m, v in zip(momentum_hat, variance_hat))
        )
