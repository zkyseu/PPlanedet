import paddle
from paddle import nn

from .norm import zeros_

def fill_fc_weights(layers):
    for m in layers.sublayers():
        if isinstance(m, nn.Conv2D):
            if m.bias is not None:
                zeros_(m.bias)

class SPGU(nn.Layer):
    def __init__(self, heads, channels_in, train_cfg=None, test_cfg=None, down_ratio=4, final_kernel=1, head_conv=256, branch_layers=0):
        super(SPGU, self).__init__()
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2D(channels_in, head_conv,
                    kernel_size=3, padding=1, bias_attr=True),
                  nn.ReLU(),
                  nn.Conv2D(head_conv, classes,
                    kernel_size=final_kernel, stride=1,
                    padding=final_kernel // 2, bias_attr=True))
              if 'hm' in head:
                inits = nn.initializer.Constant(value = -2.19)
                inits(fc[-1].bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2D(channels_in, classes,
                  kernel_size=final_kernel, stride=1,
                  padding=final_kernel // 2, bias_attr=True)
              if 'hm' in head:
                inits = nn.initializer.Constant(value = -2.19)
                inits(fc[-1].bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return z