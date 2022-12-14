from typing import Optional

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..builder import LOSSES

@LOSSES.register()
class SoftmaxFocalLoss(nn.Layer):
    """
    compare with multi-class softmax, this function need developer to decide final loss type(sum or mean)
    """
    def __init__(self, gamma, ignore_lb=255, loss_weight = 1., *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.nll = nn.NLLLoss(ignore_index=ignore_lb)
        self.loss_weight = loss_weight

    def forward(self, logits, labels):
        scores = F.softmax(logits, axis=1)
        factor = paddle.pow(1.-scores, self.gamma)
        log_score = F.log_softmax(logits, axis=1)
        log_score = factor * log_score
        loss = self.nll(log_score, labels)
        return loss*self.loss_weight

@LOSSES.register()
class FocalLoss(nn.Layer):
    """
    The implement of focal loss.
    The focal loss requires the label is 0 or 1 for now.
    Args:
        alpha (float, list, optional): The alpha of focal loss. alpha is the weight
            of class 1, 1-alpha is the weight of class 0. Default: 0.25
        gamma (float, optional): The gamma of Focal Loss. Default: 2.0
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, alpha=0.25, gamma=2.0, ignore_index=255,loss_weight = 1.,cfg = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.EPS = 1e-10
        self.loss_weight = loss_weight

    def forward(self, logit, label):
        """
        Forward computation.
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C, H, W), where C is number of classes.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, W, W),
                where each value is 0 <= label[i] <= C-1.
        Returns:
            (Tensor): The average loss.
        """
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert logit.shape[1] == 2, "The channel of logit should be 2."
        assert label.ndim == 3, "The ndim of label should be 3."

        class_num = logit.shape[1]  # class num is 2
        logit = paddle.transpose(logit, [0, 2, 3, 1])  # N,C,H,W => N,H,W,C

        mask = label != self.ignore_index  # N,H,W
        mask = paddle.unsqueeze(mask, 3)
        mask = paddle.cast(mask, 'float32')
        mask.stop_gradient = True

        label = F.one_hot(label, class_num)  # N,H,W,C
        label = paddle.cast(label, logit.dtype)
        label.stop_gradient = True

        loss = F.sigmoid_focal_loss(
            logit=logit,
            label=label,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction='none')
        loss = loss * mask
        avg_loss = paddle.sum(loss) / (
            paddle.sum(paddle.cast(mask != 0., 'int32')) * class_num + self.EPS)
        return avg_loss*self.loss_weight


@LOSSES.register()
class MultiClassFocalLoss(nn.Layer):
    """
    The implement of focal loss for multi class.
    Args:
        alpha (float, list, optional): The alpha of focal loss. alpha is the weight
            of class 1, 1-alpha is the weight of class 0. Default: 0.25
        gamma (float, optional): The gamma of Focal Loss. Default: 2.0
        ignore_index (int64, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. Default ``255``.
    """

    def __init__(self, num_class, alpha=1.0, gamma=2.0, ignore_index=255,loss_weight = 1.,cfg = None):
        super().__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.EPS = 1e-10
        self.loss_weight = loss_weight

    def forward(self, logit, label):
        """
        Forward computation.
        Args:
            logit (Tensor): Logit tensor, the data type is float32, float64. Shape is
                (N, C, H, W), where C is number of classes.
            label (Tensor): Label tensor, the data type is int64. Shape is (N, W, W),
                where each value is 0 <= label[i] <= C-1.
        Returns:
            (Tensor): The average loss.
        """
        assert logit.ndim == 4, "The ndim of logit should be 4."
        assert label.ndim == 3, "The ndim of label should be 3."

        logit = paddle.transpose(logit, [0, 2, 3, 1])
        label = label.astype('int64')
        ce_loss = F.cross_entropy(
            logit, label, ignore_index=self.ignore_index, reduction='none')

        pt = paddle.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt)**self.gamma) * ce_loss

        mask = paddle.cast(label != self.ignore_index, 'float32')
        focal_loss *= mask
        avg_loss = paddle.mean(focal_loss) / (paddle.mean(mask) + self.EPS)
        return avg_loss * self.loss_weight 

