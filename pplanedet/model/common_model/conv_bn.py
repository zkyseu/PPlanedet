from paddleseg.models.backbones.resnet_vd import ConvBNLayer
from paddleseg.cvlibs.param_init import kaiming_normal_init,constant_init
from paddleseg.models import layers
import paddle.nn as nn

class ConvModule(nn.Layer):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1,
                groups=1,
                is_vd_mode=False,
                act=None,
                norm = None,
                data_format='NCHW'):
        super().__init__()
        if dilation != 1 and kernel_size != 3:
            raise RuntimeError("When the dilation isn't 1," \
                "the kernel_size should be 3.")

        self.is_vd_mode = is_vd_mode
        self.use_norm = norm
        self.use_act = act
        if is_vd_mode:
            self._pool2d_avg = nn.AvgPool2D(
                kernel_size=2,
                stride=2,
                padding=0,
                ceil_mode=True,
                data_format=data_format)
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - 1) // 2 \
                if dilation == 1 else dilation,
            dilation=dilation,
            groups=groups,
            bias_attr=False,
            data_format=data_format)
        
        if norm:
            self._batch_norm = layers.SyncBatchNorm(
                out_channels, data_format=data_format) 
        if act is not None:
            self._act_op = layers.Activation(act=act)
        
        self.apply(self._init_weight)
    
    def _init_weight(self,m):
        if isinstance(m, nn.Conv2D):
            kaiming_normal_init(m.weight)
            if m.bias is not None:
                constant_init(m.bias, value=0)
        elif isinstance(m, (nn.SyncBatchNorm, nn.BatchNorm2D)):
            constant_init(m.weight, value=1.0)
            constant_init(m.bias, value=0)
    
    def forward(self,inputs):
        if self.is_vd_mode:
            inputs = self._pool2d_avg(inputs)
        y = self._conv(inputs)
        if self.use_norm:
            y = self._batch_norm(y)
        if self.use_act:
            y = self._act_op(y)

        return y
        
    
