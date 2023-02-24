import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

trunc_normal_ = nn.initializer.TruncatedNormal(std=0.02)
zeros_ = nn.initializer.Constant(value=0.0)
ones_ = nn.initializer.Constant(value=1.0)

class LayerNorm(nn.Layer):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self,
                 normalized_shape,
                 epsilon=1e-6,
                 data_format="channels_last"):
        super().__init__()

        self.weight = paddle.create_parameter(shape=[normalized_shape],
                                              dtype='float32',
                                              default_initializer=ones_)

        self.bias = paddle.create_parameter(shape=[normalized_shape],
                                            dtype='float32',
                                            default_initializer=zeros_)

        self.epsilon = epsilon
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight,
                                self.bias, self.epsilon)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / paddle.sqrt(s + self.epsilon)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def _no_grad_uniform_(tensor, a, b):
    with paddle.no_grad():
        tensor.set_value(
            paddle.uniform(
                shape=tensor.shape, dtype=tensor.dtype, min=a, max=b))
    return tensor

def uniform_(tensor, a, b):
    """
    Modified tensor inspace using uniform_
    Args:
        tensor (paddle.Tensor): paddle Tensor
        a (float|int): min value.
        b (float|int): max value.
    Return:
        tensor
    """
    return _no_grad_uniform_(tensor, a, b)

def conv_init_(module):
    bound = 1 / np.sqrt(np.prod(module.weight.shape[1:]))
    uniform_(module.weight, -bound, bound)
    if module.bias is not None:
        uniform_(module.bias, -bound, bound)