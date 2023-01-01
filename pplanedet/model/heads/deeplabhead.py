import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.models import layers
import cv2
import numpy as np
from ..lane import Lane

from ..builder import HEADS, build_head, build_loss

@HEADS.register()
class DeepLabV3SegHead(nn.Layer):
    def __init__(self,
                 decoder, 
                 seg_loss,
                 exist=None, 
                 exist_loss = None,
                 thr=0.6,
                 sample_y=None, 
                 cfg=None):
        super().__init__()
        self.cfg = cfg
        self.thr = thr
        self.sample_y = sample_y

        self.decoder = build_head(decoder, cfg)
        self.exist = build_head(exist, cfg) if exist else None 

        self.seg_loss = build_loss(seg_loss,cfg)
        self.exist_loss = build_loss(exist_loss,cfg) if exist_loss else None

    def get_lanes(self, output):
        segs = output['seg']
        segs = F.softmax(segs, axis=1)
        segs = segs.detach().cpu().numpy()
        if 'exist' in output:
            exists = output['exist']
            exists = exists.detach().cpu().numpy()
            exists = exists > 0.5
        else:
            exists = [None for _ in segs]

        ret = []
        for seg, exist in zip(segs, exists):
            lanes = self.probmap2lane(seg, exist)
            ret.append(lanes)
        return ret

    def probmap2lane(self, probmaps, exists=None):
        lanes = []
        probmaps = probmaps[1:, ...]
        if exists is None:
            exists = [True for _ in probmaps]
        for probmap, exist in zip(probmaps, exists):
            if exist == 0:
                continue
            probmap = cv2.blur(probmap, (9, 9), borderType=cv2.BORDER_REPLICATE)
            cut_height = self.cfg.cut_height
            ori_h = self.cfg.ori_img_h - cut_height
            coord = []
            for y in self.sample_y:
                proj_y = round((y - cut_height) * self.cfg.img_height/ori_h)
                line = probmap[proj_y]
                if np.max(line) < self.thr:
                    continue
                value = np.argmax(line)
                x = value*self.cfg.ori_img_w/self.cfg.img_width#-1.
                if x > 0:
                    coord.append([x, y])
            if len(coord) < 5:
                continue

            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= self.cfg.ori_img_w
            coord[:, 1] /= self.cfg.ori_img_h
            lanes.append(Lane(coord))
    
        return lanes

    def loss(self, output, batch):
        loss = 0.
        loss_stats = {}
        seg_loss = self.seg_loss(output['seg'],batch['mask'].astype('int64'))
        loss += seg_loss
        loss_stats.update({'seg_loss': seg_loss})

        if 'exist' in output:
            exist_loss = self.exist_loss(output['exist'], batch['lane_exist'])
            loss += exist_loss
            loss_stats.update({'exist_loss': exist_loss})

        ret = {'loss': loss}
        return ret

    def forward(self, x, evaluate = False,**kwargs):
        output = {}
        exist_feat = x[-1]
        output.update(self.decoder(x))
        if self.exist and not evaluate:
            output.update(self.exist(exist_feat))
        return output 

@HEADS.register()
class DeepLabV3PHead(nn.Layer):
    """
    The DeepLabV3PHead implementation based on PaddlePaddle.
    Args:
        num_classes (int): The unique number of target classes.
        backbone_indices (tuple): Two values in the tuple indicate the indices of output of backbone.
            the first index will be taken as a low-level feature in Decoder component;
            the second one will be taken as input of ASPP component.
            Usually backbone consists of four downsampling stage, and return an output of
            each stage. If we set it as (0, 3), it means taking feature map of the first
            stage in backbone as low-level feature used in Decoder, and feature map of the fourth
            stage as input of ASPP.
        backbone_channels (tuple): The same length with "backbone_indices". It indicates the channels of corresponding index.
        aspp_ratios (tuple): The dilation rates using in ASSP module.
        aspp_out_channels (int): The output channels of ASPP module.
        align_corners (bool): An argument of F.interpolate. It should be set to False when the output size of feature
            is even, e.g. 1024x512, otherwise it is True, e.g. 769x769.
        data_format(str, optional): Data format that specifies the layout of input. It can be "NCHW" or "NHWC". Default: "NCHW".
    """

    def __init__(self,
                 num_classes,
                 backbone_indices,
                 backbone_channels,
                 aspp_ratios,
                 aspp_out_channels,
                 align_corners,
                 data_format='NCHW',
                 cfg = None):
        super().__init__()

        self.aspp = layers.ASPPModule(
            aspp_ratios,
            backbone_channels[1],
            aspp_out_channels,
            align_corners,
            use_sep_conv=True,
            image_pooling=True,
            data_format=data_format)
        self.decoder = Decoder(
            num_classes,
            backbone_channels[0],
            align_corners,
            data_format=data_format,
            cfg = cfg
            )
        self.backbone_indices = backbone_indices

    def forward(self, feat_list):
        output = {}
        low_level_feat = feat_list[self.backbone_indices[0]]
        x = feat_list[self.backbone_indices[1]]
        x = self.aspp(x)
        logit = self.decoder(x, low_level_feat)
        output.update({'seg':logit})

        return output

class Decoder(nn.Layer):
    """
    Decoder module of DeepLabV3P model
    Args:
        num_classes (int): The number of classes.
        in_channels (int): The number of input channels in decoder module.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 align_corners,
                 data_format='NCHW',
                 cfg = None):
        super(Decoder, self).__init__()

        self.data_format = data_format
        self.conv_bn_relu1 = layers.ConvBNReLU(
            in_channels=in_channels,
            out_channels=48,
            kernel_size=1,
            data_format=data_format)

        self.conv_bn_relu2 = layers.SeparableConvBNReLU(
            in_channels=304,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format)
        self.conv_bn_relu3 = layers.SeparableConvBNReLU(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=1,
            data_format=data_format)
        self.conv = nn.Conv2D(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            data_format=data_format)

        self.align_corners = align_corners
        self.cfg = cfg

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv_bn_relu1(low_level_feat)
        if self.data_format == 'NCHW':
            low_level_shape = paddle.shape(low_level_feat)[-2:]
            axis = 1
            final_shape = [self.cfg.img_height,  self.cfg.img_width]
        else:
            low_level_shape = paddle.shape(low_level_feat)[1:3]
            axis = -1
            final_shape = [self.cfg.img_height,  self.cfg.img_width]
        x = F.interpolate(
            x,
            low_level_shape,
            mode='bilinear',
            align_corners=self.align_corners,
            data_format=self.data_format)
        x = paddle.concat([x, low_level_feat], axis=axis)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)
        x = self.conv(x)
        x = F.interpolate(
            x,
            final_shape,
            mode='bilinear',
            align_corners=self.align_corners,
            data_format=self.data_format)
        return x