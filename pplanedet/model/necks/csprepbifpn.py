import paddle 
import paddle.nn as nn
import paddle.nn.functional as F

from ..aggregators.simsppf import SimConv
from ..common_model import BaseConv,RepConv,BaseConv_C3,RepLayer_BottleRep
from ..common_model.utils import make_divisible
from ..builder import NECKS

class Transpose(nn.Layer):
    '''Normal Transpose, default for upsampling'''

    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.upsample_transpose = nn.Conv2DTranspose(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias_attr=True)

    def forward(self, x):
        return self.upsample_transpose(x)

class BiFusion(nn.Layer):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.cv1 = SimConv(in_channels[0], out_channels, 1, 1)
        self.cv2 = SimConv(in_channels[1], out_channels, 1, 1)
        self.cv3 = SimConv(out_channels * 3, out_channels, 1, 1)

        self.upsample = Transpose(
            in_channels=out_channels, out_channels=out_channels)
        self.downsample = SimConv(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2)

    def forward(self, x):
        x0 = self.upsample(x[0])
        x1 = self.cv1(x[1])
        x2 = self.downsample(self.cv2(x[2]))
        return self.cv3(paddle.concat([x0, x1, x2], 1))

class ConvBNSiLUBlock(nn.Layer):
    # ConvWrapper
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = BaseConv(in_channels, out_channels, kernel_size,
                                   stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)

class ConvBNReLUBlock(nn.Layer):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 groups=1,
                 bias=True):
        super().__init__()
        self.base_block = SimConv(in_channels, out_channels, kernel_size,
                                  stride, groups, bias)

    def forward(self, x):
        return self.base_block(x)

def get_block(mode):
    if mode == 'repvgg':
        return RepConv
    elif mode == 'conv_silu':
        return ConvBNSiLUBlock
    elif mode == 'conv_relu':
        return ConvBNReLUBlock
    else:
        raise ValueError('Unsupported mode :{}'.format(mode))

class BepC3Layer(nn.Layer):
    # Beer-mug RepC3 Block, named BepC3 in YOLOv6
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_repeats=1,
                 csp_e=0.5,
                 block=RepConv,
                 act='relu',
                 cfg=None):
        super(BepC3Layer, self).__init__()
        c_ = int(out_channels * csp_e)  # hidden channels
        self.cv1 = BaseConv_C3(in_channels, c_, 1, 1)
        self.cv2 = BaseConv_C3(in_channels, c_, 1, 1)
        self.cv3 = BaseConv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvBNSiLUBlock and act == 'silu':
            self.cv1 = BaseConv_C3(in_channels, c_, 1, 1, act=nn.Silu())
            self.cv2 = BaseConv_C3(in_channels, c_, 1, 1, act=nn.Silu())
            self.cv3 = BaseConv_C3(2 * c_, out_channels, 1, 1, act=nn.Silu())

        self.m = RepLayer_BottleRep(c_, c_, num_repeats, basic_block=block)

    def forward(self, x):
        return self.cv3(paddle.concat((self.m(self.cv1(x)), self.cv2(x)), 1))

@NECKS.register()
class CSPRepBiFPAN(nn.Layer):
    """
    CSPRepBiFPAN of YOLOv6 m/l in v3.0
    change lateral_conv + up(Transpose) to BiFusion
    """
    __shared__ = ['depth_mult', 'width_mult', 'act', 'training_mode']

    def __init__(self,
                 depth_mult=1.0,
                 width_mult=1.0,
                 in_channels=[128, 256, 512, 1024],
                 out_channels = 64,
                 training_mode='repvgg',
                 csp_e=0.5,
                 act='relu',
                 cfg= None):
        super(CSPRepBiFPAN, self).__init__()
        backbone_ch_list = in_channels
        backbone_num_repeats = [1, 6, 12, 18, 6]

        ch_list = backbone_ch_list + [out_channels for i in range(6)]
        ch_list = [make_divisible(i * width_mult, 8) for i in (ch_list)]

        num_repeats = backbone_num_repeats + [12, 12, 12, 12]
        num_repeats = [(max(round(i * depth_mult), 1) if i > 1 else i)
                       for i in (num_repeats)]

        self.in_channels = in_channels
        if csp_e == 0.67:
            csp_e = float(2) / 3
        block = get_block(training_mode)
        # RepConv(or RepVGGBlock) in M, but ConvBNSiLUBlock(or ConvWrapper) in L

        # Rep_p4
        self.reduce_layer0 = SimConv(ch_list[3], ch_list[4], 1, 1)
        self.Bifusion0 = BiFusion([ch_list[2], ch_list[1]], ch_list[4])
        self.Rep_p4 = BepC3Layer(
            ch_list[4], ch_list[4], num_repeats[4], csp_e, block=block, act=act)

        # Rep_p3
        self.reduce_layer1 = SimConv(ch_list[4], ch_list[5], 1, 1)
        self.Bifusion1 = BiFusion([ch_list[1], ch_list[0]], ch_list[5])
        self.Rep_p3 = BepC3Layer(
            ch_list[5], ch_list[5], num_repeats[5], csp_e, block=block, act=act)

        # Rep_n3
        self.downsample2 = SimConv(ch_list[5], ch_list[6], 3, 2)
        self.Rep_n3 = BepC3Layer(
            ch_list[5] + ch_list[6],
            ch_list[7],
            num_repeats[6],
            csp_e,
            block=block,
            act=act)

        # Rep_n4
        self.downsample1 = SimConv(ch_list[7], ch_list[8], 3, 2)
        self.Rep_n4 = BepC3Layer(
            ch_list[4] + ch_list[8],
            ch_list[9],
            num_repeats[7],
            csp_e,
            block=block,
            act=act)
        
        #p2 fusion
        self.upsamplep3 = Transpose(ch_list[8],ch_list[8])
        self.reduce_layerp3 = SimConv(ch_list[8],in_channels[0],1,1)
        self.Rep_up_p3 = BepC3Layer(in_channels[0]*2,out_channels,num_repeats[5])

    def forward(self, feats, for_mot=False):
        assert len(feats) == len(self.in_channels)
        [x3, x2, x1, x0] = feats  # p2, p3, p4, p5 

        # top-down FPN
        fpn_out0 = self.reduce_layer0(x0)
        f_concat_layer0 = self.Bifusion0([fpn_out0, x1, x2])
        f_out0 = self.Rep_p4(f_concat_layer0)

        fpn_out1 = self.reduce_layer1(f_out0)
        f_concat_layer1 = self.Bifusion1([fpn_out1, x2, x3])
        pan_out2 = self.Rep_p3(f_concat_layer1)

        # bottom-up PAN
        down_feat1 = self.downsample2(pan_out2)
        p_concat_layer1 = paddle.concat([down_feat1, fpn_out1], 1)
        pan_out1 = self.Rep_n3(p_concat_layer1)

        down_feat0 = self.downsample1(pan_out1)
        p_concat_layer2 = paddle.concat([down_feat0, fpn_out0], 1)
        pan_out0 = self.Rep_n4(p_concat_layer2)

        #p2 fusion
        up_p3 = self.reduce_layerp3(self.upsamplep3(pan_out2))
        fuse_p2 = paddle.concat([x3,up_p3],axis=1)
        pan_outp2 = self.Rep_up_p3(fuse_p2)

        return [pan_outp2, pan_out2, pan_out1, pan_out0]
