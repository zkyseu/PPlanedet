import paddle
from paddle import nn
import paddle.nn.functional as F

from ..builder import HEADS

@HEADS.register()
class ExistHead(nn.Layer):
    def __init__(self, cfg=None):
        super(ExistHead, self).__init__()
        self.cfg = cfg

        self.dropout = nn.Dropout2D(0.1)
        self.conv8 = nn.Conv2D(cfg.featuremap_out_channel, cfg.num_classes, 1)

        stride = cfg.featuremap_out_stride * 2
        self.fc9 = nn.Linear(
            int(cfg.exist_num_class * cfg.img_width / stride * cfg.img_height / stride), 128)
        self.fc10 = nn.Linear(128, cfg.exist_num_class-1)

    def forward(self, x):
        x = self.dropout(x)
        x = self.conv8(x)

        x = F.softmax(x, axis=1)
        x = F.avg_pool2d(x, 2, stride=2, padding=0)
        x = x.reshape((-1, x.numel() // x.shape[0]))
        x = self.fc9(x)
        x = F.relu(x)
        x = self.fc10(x)

        output = {'exist': x}

        return output