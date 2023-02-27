import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ..builder import LOSSES

@LOSSES.register()
class ParsingRelationLoss(nn.Layer):
    def __init__(self,weight = 1.0,cfg = None):
        super(ParsingRelationLoss, self).__init__()
        self.weight = weight
    def forward(self,logits):
        n,c,h,w = logits.shape
        loss_all = []
        for i in range(0,h-1):
            loss_all.append(logits[:,:,i,:] - logits[:,:,i+1,:])
        #loss0 : n,c,w
        loss = paddle.concat(loss_all)
        return F.smooth_l1_loss(loss,paddle.zeros_like(loss))*self.weight

@LOSSES.register()
class ParsingRelationDis(nn.Layer):
    def __init__(self,weight = 1.0,cfg = None):
        super(ParsingRelationDis, self).__init__()
        self.l1 = nn.L1Loss()
        self.weight = weight
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        x = F.softmax(x[:,:dim-1,:,:],axis=1)
        embedding = paddle.to_tensor(np.arange(dim-1)).astype('float').reshape((1,-1,1,1))
        pos = paddle.sum(x*embedding,axis=1)

        diff_list1 = []
        for i in range(0,num_rows // 2):
            diff_list1.append(pos[:,i,:] - pos[:,i+1,:])

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l1(diff_list1[i],diff_list1[i+1])
        loss /= len(diff_list1) - 1
        return loss*self.weight