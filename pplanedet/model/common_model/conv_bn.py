from paddleseg.models.backbones.resnet_vd import ConvBNLayer
from paddleseg.cvlibs.param_init import kaiming_normal_init,constant_init
from paddleseg.models import layers
import paddle.nn as nn
import paddle.nn.functional as F
import paddle

from .norm import conv_init_
from paddle import ParamAttr
from paddle.regularizer import L2Decay
from paddle.nn.initializer import Constant

class swish(nn.Layer):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return x * F.sigmoid(x)

class SiLU(nn.Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def forward(self, x):
        return x * F.sigmoid(x)

class ConvModule(nn.Layer):
    def __init__(self,
                in_channels,
                out_channels,
                kernel_size,
                stride=1,
                dilation=1,
                groups=1,
                is_vd_mode=False,
                padding = None,
                act=None,
                norm = None,
                bias = True,
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
        if padding is None:
            padding = (kernel_size - 1) // 2 if dilation == 1 else dilation
        self._conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias_attr=bias,
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
        
    
class BaseConv(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 ksize,
                 stride,
                 groups=1,
                 bias=False,
                 act="silu"):
        super(BaseConv, self).__init__()
        self.conv = nn.Conv2D(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=(ksize - 1) // 2,
            groups=groups,
            bias_attr=bias)
        self.bn = nn.BatchNorm2D(
            out_channels,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        self.act = nn.Silu()
        self._init_weights()

    def _init_weights(self):
        conv_init_(self.conv)

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.Silu):
                self.act = SiLU()
            y = self.act(x)
        return y

class RepConv(nn.Layer):
    """
    RepVGG Conv BN Relu Block, see https://arxiv.org/abs/2101.03697
    named RepVGGBlock in YOLOv6
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 groups=1,
                 act='relu',
                 deploy=False):
        super(RepConv, self).__init__()
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        assert kernel_size == 3
        assert padding == 1
        padding_11 = padding - kernel_size // 2
        self.stride = stride  # not always 1

        self.nonlinearity = nn.ReLU()  # always relu in YOLOv6

        if self.deploy:
            self.rbr_reparam = nn.Conv2D(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias_attr=True)
        else:
            self.rbr_identity = (nn.BatchNorm2D(in_channels)
                                 if out_channels == in_channels and stride == 1
                                 else None)
            self.rbr_dense = nn.Sequential(* [
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,  #
                    padding,
                    groups=groups,
                    bias_attr=False),
                nn.BatchNorm2D(out_channels),
            ])
            self.rbr_1x1 = nn.Sequential(* [
                nn.Conv2D(
                    in_channels,
                    out_channels,
                    1,
                    stride,
                    padding_11,  #
                    groups=groups,
                    bias_attr=False),
                nn.BatchNorm2D(out_channels),
            ])

    def forward(self, inputs):
        if hasattr(self, "rbr_reparam"):
            x = self.rbr_reparam(inputs)
            y = self.nonlinearity(x)
            return y

        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)

        x = self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out
        y = self.nonlinearity(x)
        return y

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(
            kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return nn.functional.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch[0].weight
            running_mean = branch[1]._mean
            running_var = branch[1]._variance
            gamma = branch[1].weight
            beta = branch[1].bias
            eps = branch[1]._epsilon
        else:
            assert isinstance(branch, nn.BatchNorm2D)
            if not hasattr(self, "id_tensor"):
                input_dim = self.in_channels // self.groups
                kernel_value = paddle.zeros([self.in_channels, input_dim, 3, 3])
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value
            kernel = self.id_tensor
            running_mean = branch._mean
            running_var = branch._variance
            gamma = branch.weight
            beta = branch.bias
            eps = branch._epsilon
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape((-1, 1, 1, 1))
        return kernel * t, beta - running_mean * gamma / std

    def convert_to_deploy(self):
        if hasattr(self, "rbr_reparam"):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2D(
            self.rbr_dense[0]._in_channels,
            self.rbr_dense[0]._out_channels,
            self.rbr_dense[0]._kernel_size,
            self.rbr_dense[0]._stride,
            padding=self.rbr_dense[0]._padding,
            groups=self.rbr_dense[0]._groups,
            bias_attr=True)
        self.rbr_reparam.weight.set_value(kernel)
        self.rbr_reparam.bias.set_value(bias)
        self.__delattr__("rbr_dense")
        self.__delattr__("rbr_1x1")
        if hasattr(self, "rbr_identity"):
            self.__delattr__("rbr_identity")
        if hasattr(self, "id_tensor"):
            self.__delattr__("id_tensor")
        self.deploy = True

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class BaseConv_C3(nn.Layer):
    '''Standard convolution in BepC3-Block'''

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super(BaseConv_C3, self).__init__()
        self.conv = nn.Conv2D(
            c1, c2, k, s, autopad(k, p), groups=g, bias_attr=False)
        self.bn = nn.BatchNorm2D(
            c2,
            weight_attr=ParamAttr(regularizer=L2Decay(0.0)),
            bias_attr=ParamAttr(regularizer=L2Decay(0.0)))
        if act == True:
            self.act = nn.ReLU()
        else:
            if isinstance(act, nn.Layer):
                self.act = act
            else:
                self.act = nn.Identity()

    def forward(self, x):
        x = self.bn(self.conv(x))
        if self.training:
            y = self.act(x)
        else:
            if isinstance(self.act, nn.Silu):
                self.act = SiLU()
            y = self.act(x)
        return y

class RepLayer_BottleRep(nn.Layer):
    """
    RepLayer with RepConvs for M/L, like CSPLayer(C3) in YOLOv5/YOLOX
    named RepBlock in YOLOv6
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 basic_block=RepConv):
        super(RepLayer_BottleRep, self).__init__()
        # in m/l
        self.conv1 = BottleRep(
            in_channels, out_channels, basic_block=basic_block, alpha=True)
        num_repeats = num_repeats // 2
        self.block = nn.Sequential(*(BottleRep(
            out_channels, out_channels, basic_block=basic_block, alpha=True
        ) for _ in range(num_repeats - 1))) if num_repeats > 1 else None

    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class BottleRep(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 basic_block=RepConv,
                 alpha=True):
        super(BottleRep, self).__init__()
        # basic_block: RepConv or ConvBNSiLUBlock
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if alpha:
            self.alpha = self.create_parameter(
                shape=[1],
                attr=ParamAttr(initializer=Constant(value=1.)),
                dtype="float32")
        else:
            self.alpha = 1.0

    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs