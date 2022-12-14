import paddle
from paddle import nn
import paddle.nn.functional as F


from ..builder import HEADS

@HEADS.register()
class PlainDecoder(nn.Layer):
    def __init__(self, cfg):
        super(PlainDecoder, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2D(0.1)
        self.conv8 = nn.Conv2D(cfg.featuremap_out_channel, cfg.num_classes, 1)

    def forward(self, x):

        x = self.dropout(x)
        x = self.conv8(x)
        x = F.interpolate(x, size=[self.cfg.img_height,  self.cfg.img_width],
                           mode='bilinear', align_corners=False)

        output = {'seg': x}

        return output 