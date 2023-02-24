import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from ..common_model.norm import conv_init_

from ..builder import AGGREGATORS

class SimConv(nn.Layer):
    """Simplified Conv BN ReLU"""

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False):
        super(SimConv, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = nn.ReLU()
        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

@AGGREGATORS.register()
class SimSPPF(nn.Layer):
    """Simplified SPPF with SimConv, use relu"""

    def __init__(self, in_channels, out_channels, kernel_size=5,cfg = None):
        super(SimSPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = SimConv(in_channels, hidden_channels, 1, 1)
        self.mp = nn.MaxPool2D(
            kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.conv2 = SimConv(hidden_channels * 4, out_channels, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.mp(x)
        y2 = self.mp(y1)
        y3 = self.mp(y2)
        concats = paddle.concat([x, y1, y2, y3], 1)
        return self.conv2(concats)

@AGGREGATORS.register()
class SimCSPSPPF(nn.Layer):
    # SimCSPSPPF proposed in YOLOv6
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5,cfg= None):
        super(SimCSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(in_channels, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)
        
        self.m = nn.MaxPool2D(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv7 = SimConv(2 * c_, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        y1 = self.m(x1)
        y2 = self.m(y1)
        y3 = self.cv6(self.cv5(paddle.concat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(paddle.concat((y0, y3), axis=1))