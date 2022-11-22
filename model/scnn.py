import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import numpy as np
import cv2

from paddleseg.cvlibs import manager
from paddleseg.utils import utils

from .lane import Lane

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

@manager.MODELS.add_component
class SCNN(nn.Layer):
    def __init__(self, 
                backbone,
                featuremap_out_channel,
                num_classes,
                thr,
                sample_y = range(710, 150, -10),
                ignore_idx = 255,
                backbone_indices = None,
                out_conv = False,
                pretrain = None,
                use_exist = False,
                **kwargs):
        super(SCNN, self).__init__()

        self.backbone = backbone

        backbone_channels = [
            backbone.feat_channels[i] for i in backbone_indices
        ]

        self.pretrained = pretrain

        if out_conv:
            self.out_conv = conv1x1(backbone_channels[-1],featuremap_out_channel)

        self.sample_y = sample_y
        self.bg_weight = 0.4

        self.scnn = Spatial_convolution(featuremap_out_channel,featuremap_out_channel)

        self.seg_decoder = LaneSeg(featuremap_out_channel,
                                   num_classes,
                                   thr,
                                   sample_y,
                                   self.bg_weight,
                                   ignore_idx,
                                   use_exist = use_exist,
                                   **kwargs)    

        self.init_weight()
        self.use_exist = use_exist

    def get_lanes(self, output):
        if "exist" in output.keys():
            _ = output.pop("exist")
        return self.seg_decoder.get_lanes(output)

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, x):
        

        feat_list = self.backbone(x)
        feat_list[-1] = self.out_conv(feat_list[-1])
        feat_list[-1] = self.scnn(feat_list[-1])

        out = self.seg_decoder(feat_list)

        self.out = out

        if self.use_exist:
            return [out["seg"],out["exist"]]
        
        return [out["seg"]]


class PlainDecoder(nn.Layer):
    def __init__(self, 
                featuremap_out_channel,
                num_classes,
                img_width,
                img_height,
                featuremap_out_stride = 8,
                use_exist = False):
        super(PlainDecoder, self).__init__()

        self.dropout = nn.Dropout2D(0.1)
        self.conv8 = nn.Conv2D(featuremap_out_channel, num_classes, 1)

        if use_exist:
            stride = featuremap_out_stride * 2
            self.fc9 = nn.Linear(
                int(num_classes * img_width / stride * img_height / stride), 128)
            self.fc10 = nn.Linear(128, num_classes-1)

        self.use_exist = use_exist

    def forward(self, x,img_height,img_width):

        x = self.dropout(x)
        x = self.conv8(x)
        exist_feat = x.clone()
        x = F.interpolate(x, size=[img_height,  img_width],
                           mode='bilinear', align_corners=False)

        output = {'seg': x}

        if self.use_exist:
            exist_feat = F.softmax(exist_feat, axis=1)
            exist_feat = F.avg_pool2d(exist_feat, 2, stride=2, padding=0)
            exist_feat = exist_feat.reshape((-1, exist_feat.numel() // exist_feat.shape[0]))
            exist_feat = self.fc9(exist_feat)
            exist_feat = F.relu(exist_feat)
            exist_feat = self.fc10(exist_feat)
            exist_feat = F.sigmoid(exist_feat)

            output.update({"exist":exist_feat})

        return output 


class Spatial_convolution(nn.Layer):
    def __init__(self,in_channel,out_channel):
        super().__init__()

        self.conv_d = nn.Conv2D(in_channel, out_channel, (1, 9), padding=(0, 4), bias_attr=False)
        self.conv_u = nn.Conv2D(in_channel, out_channel, (1, 9), padding=(0, 4), bias_attr=False)
        self.conv_r = nn.Conv2D(in_channel, out_channel, (9, 1), padding=(4, 0), bias_attr=False)
        self.conv_l = nn.Conv2D(in_channel, out_channel, (9, 1), padding=(4, 0), bias_attr=False)


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

class LaneSeg(nn.Layer):
    def __init__(self, 
                featuremap_out_channel,
                num_classes,
                thr=0.6,
                sample_y=None, 
                bg_weight = None,
                ignore_label = 255,
                use_exist = False,
                **kwargs):
        super(LaneSeg, self).__init__()
        self.thr = thr
        self.sample_y = sample_y

        self.num_classes = num_classes
        self.bg_weight = bg_weight
        self.ignore_label = ignore_label

        if kwargs is not None:

            self.cut_height = kwargs["cut_height"]
            self.ori_img_h = kwargs["ori_img_h"]
            self.img_height = kwargs["img_height"]
            self.ori_img_w = kwargs["ori_img_w"]
            self.img_width = kwargs["img_width"]
        
        self.decoder = PlainDecoder(featuremap_out_channel,
                                    num_classes,
                                    img_width = self.img_width,
                                    img_height = self.img_height,
                                    use_exist = use_exist)


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
            cut_height = self.cut_height
            ori_h = self.ori_img_h - cut_height
            coord = []
            for y in self.sample_y:
                proj_y = round((y - cut_height) * self.img_height/ori_h)
                line = probmap[proj_y]
                if np.max(line) < self.thr:
                    continue
                value = np.argmax(line)
                x = value*self.ori_img_w/self.img_width#-1.
                if x > 0:
                    coord.append([x, y])
            if len(coord) < 5:
                continue

            coord = np.array(coord)
            coord = np.flip(coord, axis=0)
            coord[:, 0] /= self.ori_img_w
            coord[:, 1] /= self.ori_img_h
            lanes.append(Lane(coord))
    
        return lanes


    def forward(self, x, **kwargs):
        output = {}
        x = x[-1]
        output.update(self.decoder(x,self.img_height,self.img_width))

        return output 