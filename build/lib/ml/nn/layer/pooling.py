from ml.device import device as np
from ml.nn.base import Layer
from typing import Tuple, Union


def max_pool_2d_array_and_index(x: np.ndarray, kernel: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    kh, kw = kernel
    h, w = x.shape[-2:]

    x: np.ndarray = x[..., :h - h % kh, : w - w % kw]

    h_out, w_out = h // kh, w // kw

    x: np.ndarray = x.reshape((*x.shape[:-2], h_out, kh, w_out, kw))

    x = x.swapaxes(-2, -3)

    x = x.reshape((*x.shape[:-2], -1))

    return x, x.argmax(axis=-1)


class MaxPool2d(Layer):
    kernel: Tuple[int, int]
    padding: Tuple[int, int, int, int]
    padding_by: float

    _last_padded: np.ndarray
    _last_pooling_array: np.ndarray
    _last_pooling_index: np.ndarray

    def __init__(self, kernel: Tuple[int, int],
                 padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = (0, 0, 0, 0),
                 stride: Tuple[int, int] = None,
                 padding_by: float = -np.inf):
        super().__init__()
        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif isinstance(padding, tuple) and len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])

        if stride is None:
            stride = kernel

        assert stride == kernel

        self.kernel = kernel
        self.padding = padding
        self.padding_by = padding_by

    def forward(self, x: np.ndarray) -> np.ndarray:
        x_pad = np.pad(x,
                       (*((0, 0) for _ in x.shape[:-2]),
                        (self.padding[0], self.padding[1]), (self.padding[2], self.padding[3])),
                       constant_values=self.padding_by)
        self._last_padded = x_pad
        pooling_array, pooling_index = max_pool_2d_array_and_index(x_pad, self.kernel)
        self._last_pooling_array = pooling_array
        self._last_pooling_index = pooling_index
        return np.max(pooling_array, axis=-1)

    def gradient(self, error: np.ndarray) -> np.ndarray:

        padded = self._last_padded

        pad_height, pad_width = padded.shape[-2:]

        error_height, error_width = error.shape[-2:]
        kernel_height, kernel_width = self.kernel

        pooling_array, pooling_index = self._last_pooling_array, self._last_pooling_index
        mask = np.full(shape=(*error.shape, kernel_height * kernel_width),
                       fill_value=np.arange(0, kernel_height * kernel_width)) \
               == pooling_index[..., np.newaxis]

        # (..., EH, EW, kh, kw)
        grad: np.ndarray = error[..., np.newaxis] * mask

        grad = grad.reshape((*grad.shape[:-1], kernel_height, kernel_width))

        # (..., EH, kh, EW, kw)
        grad = grad.swapaxes(-2, -3)

        # (..., PH, PW)
        grad = grad.reshape((*grad.shape[:-4], error_height * kernel_height, error_width * kernel_width))

        input_h, input_w = self._last.shape[-2:]

        grad = np.pad(grad,
                      (*(((0, 0),) * (len(grad.shape) - 2)),
                       (0, input_h - error_height * kernel_height),
                       (0, input_w - error_width * kernel_width)))

        return grad

    def update(self, delta: Tuple[Union[np.ndarray, float], ...]):
        pass

    def zero_grad(self):
        self._grad = ()
