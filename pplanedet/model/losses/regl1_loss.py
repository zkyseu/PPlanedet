import paddle.nn as nn
import paddle.nn.functional as F

from ..builder import LOSSES

@LOSSES.register()
class RegL1KpLoss(nn.Layer):

    def __init__(self,cfg):
        super(RegL1KpLoss, self).__init__()

    def forward(self, output, target, mask):
        loss = F.l1_loss(output * mask, target * mask, reduction='sum')
        mask = mask.astype('bool').astype('float')
        loss = loss / (mask.sum() + 1e-4)
        return loss