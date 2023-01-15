import paddle.nn as nn
import paddle

from ..builder import LOSSES

def _neg_loss(pred, gt, channel_weights=None):
    ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
      Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
    '''
    compare_tensor = paddle.ones_like(gt)
    pos_inds = gt.equal(compare_tensor)
    neg_inds = gt.less_than(compare_tensor)

    neg_weights = paddle.pow(1 - gt, 4)

    loss = 0

    pos_loss = paddle.log(pred) * paddle.pow(1 - pred, 2) * pos_inds
    neg_loss = paddle.log(1 - pred) * paddle.pow(pred,
                                               2) * neg_weights * neg_inds

    num_pos = pos_inds.sum()
    if channel_weights is None:
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
    else:
        pos_loss_sum = 0
        neg_loss_sum = 0
        for i in range(len(channel_weights)):
            p = pos_loss[:, i, :, :].sum() * channel_weights[i]
            n = neg_loss[:, i, :, :].sum() * channel_weights[i]
            pos_loss_sum += p
            neg_loss_sum += n
        pos_loss = pos_loss_sum
        neg_loss = neg_loss_sum
    if num_pos > 2:
        loss = loss - (pos_loss + neg_loss) / num_pos
    else:
        loss = loss - (pos_loss + neg_loss) / 256
        loss = paddle.to_tensor(0, dtype=paddle.float32)
    return loss

@LOSSES.register()
class Condlane_focalLoss(nn.Layer):
    '''nn.Layer warpper for focal loss'''

    def __init__(self,cfg):
        super(Condlane_focalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target, weights_list=None):
        return self.neg_loss(out, target, weights_list)
