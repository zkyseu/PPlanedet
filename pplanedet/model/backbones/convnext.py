# This file is mainly modified from ConvNeXt in PASSL
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from paddleseg.utils import load_entire_model
from ..common_model import (DropPath,Identity,trunc_normal_,zeros_,ones_,LayerNorm)
from ..builder import BACKBONES


class Block(nn.Layer):
    """ ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2D(dim, dim, kernel_size=7, padding=3,
                                groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, epsilon=1e-6)
        self.pwconv1 = nn.Linear(
            dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = paddle.create_parameter(
            shape=[dim],
            dtype='float32',
            default_initializer=nn.initializer.Constant(
                value=layer_scale_init_value)
        ) if layer_scale_init_value > 0 else None

        self.drop_path = DropPath(drop_path) if drop_path > 0. else Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.transpose([0, 2, 3, 1])  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose([0, 3, 1, 2])  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

@BACKBONES.register()
class ConvNeXt(nn.Layer):
    """ ConvNeXt
        A Paddle impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """

    def __init__(
        self,
        in_chans=3,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        drop_path_rate=0.,
        layer_scale_init_value=1e-6,
        pretrain = None,
        out_conv = False,
        cfg = None
    ):
        super().__init__()

        self.downsample_layers = nn.LayerList(
        )  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2D(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], epsilon=1e-6, data_format="channels_first"))
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], epsilon=1e-6, data_format="channels_first"),
                nn.Conv2D(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.LayerList(
        )  # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [
            x.item() for x in paddle.linspace(0, drop_path_rate, sum(depths))
        ]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                Block(dim=dims[i],
                      drop_path=dp_rates[cur + j],
                      layer_scale_init_value=layer_scale_init_value)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], epsilon=1e-6)  # final norm layer
        if out_conv:
            self.out_conv = nn.Conv2D(dims[-1],dims[-1]*4,1)

        self.apply(self._init_weights)
        if pretrain is not None:
            load_entire_model(self,pretrain)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2D, nn.Linear)):
            trunc_normal_(m.weight)
            zeros_(m.bias)

    def forward(self, x):
        output = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            output.append(x)
        if hasattr(self,'out_conv'):
            output[-1] = self.out_conv(output[-1])
        return output


if __name__ == '__main__':

    model = paddle.Model(ConvNeXt())
    model.summary((1,3,224,224))
