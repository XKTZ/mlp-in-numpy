from typing import Tuple, Union
from ml.device import device as np
from ml.nn.base import Layer
import math


def raw_conv2d(img: np.ndarray, ker: np.ndarray) -> np.ndarray:
    """
    Do a 2d convolution of [B, C, H, W] to a kernel [C_out, C, H, W] with padding and stride
    :param img: image
    :param ker: kernel
    :param padding: padding
    :param stride: stride
    :return: conved image
    """

    ker_out, ker_c, ker_h, ker_w = ker.shape

    """
    q = np.zeros((batch, c_out, out_h, out_w))
    for b in range(batch):
        for co in range(c_out):
            for ci in range(c_in):
                for i in range(out_h):
                    for j in range(out_w):
                        q[b, co, i, j] += np.sum(x[b,ci,i:i+ker_h,j:j+ker_w] * y[co,ci])
    q
    """

    return np.einsum("bphwijk,oijk->bohw",
                     np.lib.stride_tricks.sliding_window_view(img, window_shape=(ker_c, ker_h, ker_w),
                                                              axis=(1, 2, 3)),
                     ker)


class Conv2d(Layer):

    @staticmethod
    def backward_kernel(x: np.ndarray) -> np.ndarray:
        return np.flip(np.flip(x, -1), -2).swapaxes(0, 1)

    @staticmethod
    def unpad(x: np.ndarray, pad_t: int, pad_b: int, pad_l: int, pad_r: int) -> np.ndarray:
        b, c, h, w = x.shape
        return x[:, :, pad_t:h - pad_b, pad_l: w - pad_r]

    channel_in: int
    channel_out: int
    pad: Tuple[int, int, int, int]
    stride: int

    biased: bool
    bias: np.ndarray

    kernel_size: Tuple[int, int]
    kernel: np.ndarray

    _last_padded: np.ndarray

    def __init__(self, input_dim: int, output_dim: int,
                 kernel: Union[int, Tuple[int, int]],
                 padding: Union[int, Tuple[int, int], Tuple[int, int, int, int]] = 0,
                 stride: int = 1, bias: bool = True):
        super().__init__()

        self.channel_in = input_dim
        self.channel_out = output_dim

        if isinstance(kernel, int):
            kernel = (kernel, kernel)
        self.kernel_size = kernel

        if isinstance(padding, int):
            padding = (padding, padding, padding, padding)
        elif len(padding) == 2:
            padding = (padding[0], padding[0], padding[1], padding[1])

        self.pad = padding
        self.stride = stride

        self.kernel = np.random.normal(0, math.sqrt(2 / output_dim), size=(output_dim, input_dim, *kernel))
        self.biased = bias
        if bias:
            self.bias = np.random.uniform(- 1 / math.sqrt(output_dim), 1 / math.sqrt(output_dim), size=(output_dim,))
        else:
            self.bias = np.zeros((output_dim,))

    def forward(self, x: np.ndarray) -> np.ndarray:
        pad_t, pad_b, pad_l, pad_r = self.pad

        img_pad = np.pad(x, ((0, 0), (0, 0), (pad_t, pad_b), (pad_l, pad_r)))

        self._last_padded = img_pad

        if self.biased:
            # (B, O, H, W) + (O,) -> (O, B, H, W) + (0,) -> back
            return (raw_conv2d(img_pad, self.kernel).swapaxes(0, 1)
                    + self.bias[:, np.newaxis, np.newaxis, np.newaxis]).swapaxes(0, 1)
        else:
            return raw_conv2d(img_pad, self.kernel)

    def gradient(self, error: np.ndarray) -> np.ndarray:
        ker_h, ker_w = self.kernel_size
        pad_t, pad_b, pad_l, pad_r = self.pad

        error_pad = np.pad(error, ((0, 0), (0, 0), (ker_h - 1, ker_h - 1), (ker_w - 1, ker_w - 1)))
        backward_kernel = self.backward_kernel(self.kernel)

        if self.biased:
            self._grad = (
                self._grad[0] + raw_conv2d(self._last_padded.swapaxes(0, 1), error.swapaxes(0, 1)).swapaxes(0, 1),
                self._grad[1] + np.sum(error, axis=(0, -2, -1))
            )
        else:
            self._grad = (
                self._grad[0] + raw_conv2d(self._last_padded.swapaxes(0, 1), error.swapaxes(0, 1)).swapaxes(0, 1),
                self._grad[1] + np.array([])
            )

        return self.unpad(raw_conv2d(error_pad, backward_kernel), pad_t, pad_b, pad_l, pad_r)

    def update(self, delta: Tuple[Union[np.ndarray, float], ...]):
        self.kernel += delta[0]
        if self.biased:
            self.bias += delta[1]

    def zero_grad(self):
        self._grad = (0., 0.)

    def load(self, o):
        self.kernel = np.array(o["w"])
        self.bias = np.array(o["b"])
        self.biased = o["biased"]

    def save(self) -> object:
        return {
            "w": self.kernel.tolist(),
            "b": self.bias.tolist(),
            "biased": self.biased
        }