import paddle
import paddle.nn as nn
from ..builder import LOSSES

def Gline_iou(pred, target, img_w, length=15, aligned=True):
    '''
    Calculate the Gline iou value between predictions and targets
    Args:
        pred: lane predictions, shape: (num_pred, 72)
        target: ground truth, shape: (num_target, 72)
        img_w: image width
        length: extended radius
        aligned: True for iou loss calculation, False for pair-wise ious in assign
    '''
    px1 = pred - length
    px2 = pred + length
    tx1 = target - length
    tx2 = target + length
    if aligned:
        invalid_mask = target
        ovr = paddle.minimum(px2, tx2) - paddle.maximum(px1, tx1)
        union = paddle.maximum(px2, tx2) - paddle.minimum(px1, tx1)
        G = paddle.clip((union - 4*length) / (union + 1e-9),min=0.,max=1.)
    else:
        num_pred = pred.shape[0]
        invalid_mask = target.repeat(num_pred, 1, 1)
        ovr = (paddle.minimum(px2[:, None, :], tx2[None, ...]) -
               paddle.maximum(px1[:, None, :], tx1[None, ...]))
        union = (paddle.minimum(px2[:, None, :], tx2[None, ...]) -
                 paddle.maximum(px1[:, None, :], tx1[None, ...]))

    invalid_masks = (invalid_mask < 0) | (invalid_mask >= img_w)
    ovr[invalid_masks] = 0.
    union[invalid_masks] = 0.
    G[invalid_masks] = 0.
    iou = ovr.sum(axis=-1) / (union.sum(axis=-1) + 1e-9) - G.sum(axis=-1)
    # num = paddle.count_nonzero(ovr,dim=1)
    # weights = paddle.zeros(ovr.size())
    # def find_first_nonezero(x):
    #     index = paddle.arange(x.shape[1]).unsqueeze(0).repeat((x.shape[0],1))
    #     index[x==0] = x.shape[1]
    #     return paddle.min(index,dim=1)[0]
    # a = find_first_nonezero(ovr)
    return iou
def Gliou_loss(pred, target, img_w, length=15):
    return (1 - Gline_iou(pred, target, img_w, length)).mean()


@LOSSES.register()
class GLiou_loss(nn.Layer):
    def __init__(self,cfg):
        super().__init__()
    
    def forward(self,pred, target, img_w, length=15):
        return Gliou_loss(pred, target, img_w, length)
