import paddle
from paddle import nn
import paddle.nn.functional as F

from ..builder import AGGREGATORS

@AGGREGATORS.register()
class SCNN(nn.Layer):
    def __init__(self, cfg=None):
        super(SCNN, self).__init__()
        self.conv_d = nn.Conv2D(128, 128, (1, 9), padding=(0, 4), bias_attr=False)
        self.conv_u = nn.Conv2D(128, 128, (1, 9), padding=(0, 4), bias_attr=False)
        self.conv_r = nn.Conv2D(128, 128, (9, 1), padding=(4, 0), bias_attr=False)
        self.conv_l = nn.Conv2D(128, 128, (9, 1), padding=(4, 0), bias_attr=False)

    def forward(self, x):
        x = x.clone()
        for i in range(1, x.shape[2]):
            x[..., i:i+1, :].set_value(x[..., i:i+1, :] + F.relu(self.conv_d(x[..., i-1:i, :])))

        for i in range(x.shape[2] - 2, 0, -1):
            x[..., i:i+1, :].set_value(x[..., i:i+1, :] + F.relu(self.conv_u(x[..., i+1:i+2, :])))

        for i in range(1, x.shape[3]):
            x[..., i:i+1].set_value(x[..., i:i+1] + F.relu(self.conv_r(x[..., i-1:i])))

        for i in range(x.shape[3] - 2, 0, -1):
            x[..., i:i+1].set_value(x[..., i:i+1] + F.relu(self.conv_l(x[..., i+1:i+2])))
        return x