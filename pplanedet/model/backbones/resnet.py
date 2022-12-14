from paddleseg.models.backbones.resnet_vd import ResNet_vd
from paddleseg.utils import utils
from ..builder import BACKBONES
import paddle.nn as nn

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2D(in_planes, out_planes, kernel_size=1, stride=stride, bias_attr=False)

@BACKBONES.register()
class ResNet(ResNet_vd):
    def __init__(self,
                 return_idx = [0,1,2],
                 pretrained = None,
                 out_conv=True,
                 cfg = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.return_idx = return_idx
        self.pretrained = pretrained
        self.out_convs = out_conv
        
        backbone_channels = [
            self.feat_channels[i] for i in return_idx
        ]
    
        if out_conv:
            self.out_conv = conv1x1(backbone_channels[-1],cfg.featuremap_out_channel) 
        
        self.init_weight() 

    def init_weight(self):
        if self.pretrained is not None:
            utils.load_entire_model(self, self.pretrained)

    def forward(self, inputs):
        y = self.conv1_1(inputs)
        y = self.conv1_2(y)
        y = self.conv1_3(y)
        self.conv1_logit = y.clone()
        y = self.pool2d_max(y)

        # A feature list saves the output feature map of each stage.
        feat_list = []
        for idx,stage in enumerate(self.stage_list):
            if idx in self.return_idx:
                for block in stage:
                    y = block(y)
                feat_list.append(y)

        if self.out_convs:
            feat_list[-1] = self.out_conv(feat_list[-1]              )

        return feat_list    
