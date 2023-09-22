import math
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddleseg.cvlibs.param_init import normal_init

from ..common_model import LinearModule
from ..common_model.norm import linear_init_
from ..common_model.Transformer_module import inverse_sigmoid
from ..common_model.dynamic_assign import HungarianLaneAssigner
from ..lane import Lane

from ..builder import HEADS,build_loss

class MLP(nn.Layer):
    """This code is based on
        https://github.com/facebookresearch/detr/blob/main/models/detr.py
    """

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

        self._reset_parameters()

    def _reset_parameters(self):
        for l in self.layers:
            linear_init_(l)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
@HEADS.register()
class DETRHead(nn.Layer):
    __shared__ = ['num_classes', 'hidden_dim', 'use_focal_loss']
    __inject__ = ['loss']

    def __init__(self,
                 num_points=72,
                 num_priors=192,
                 hidden_dim=256,
                 sample_points=36,
                 num_mlp_layers=3,
                 cls_loss = None,
                 liou_loss = None,
                 cfg=None):
        super(DETRHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_width
        self.img_h = self.cfg.img_height
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points

        self.register_buffer(name='sample_x_indexs', tensor=(paddle.linspace(
            0, 1, num=self.sample_points, dtype=paddle.float32) *
                                self.n_strips).astype('int64'))
        self.register_buffer(name='prior_feat_ys', tensor=paddle.flip(
            (1 - self.sample_x_indexs.astype('float32') / self.n_strips), axis=[-1]))
        self.register_buffer(name='prior_ys', tensor=paddle.linspace(1,
                                       0,
                                       num=self.n_offsets,
                                       dtype=paddle.float32))

        for _ in range(num_mlp_layers):
            reg_modules += [*LinearModule(hidden_dim)]
            cls_modules += [*LinearModule(hidden_dim)]
        self.reg_modules = nn.LayerList(reg_modules)
        self.cls_modules = nn.LayerList(cls_modules)

        self.reg_layers = nn.Linear(
            hidden_dim, self.n_offsets + 1 + 2 +
            1,bias_attr=True)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(hidden_dim, 2, bias_attr=True)

        self.init_weights()
        self.focal_loss = build_loss(cls_loss,cfg)
        self.liou_loss = build_loss(liou_loss,cfg)

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            normal_init(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            normal_init(m, mean=0., std=1e-3)


    def forward(self, out_transformer, body_feats, inputs=None):
        r"""
        Args:
            out_transformer (Tuple): (feats: [num_levels, batch_size,
                                                num_queries, hidden_dim],
                            memory: [batch_size, hidden_dim, h, w],
                            src_proj: [batch_size, h*w, hidden_dim],
                            src_mask: [batch_size, 1, 1, h, w])
            body_feats (List(Tensor)): list[[B, C, H, W]]
            inputs (dict): dict(inputs)
        """
        feats, memory, src_proj, src_mask = out_transformer
        batch_size = memory.shape[0]

        for cls_layer in self.cls_modules:
            cls_features = cls_layer(feats)
        for reg_layer in self.reg_modules:
            reg_features = reg_layer(feats)

        cls_logits = self.cls_layers(cls_features)
        reg = self.reg_layers(reg_features)

        cls_logits = cls_logits.reshape((
            batch_size, -1, cls_logits.shape[1]))  # (B, num_priors, 2)
        reg = reg.reshape((batch_size, -1, reg.shape[1]))

        predictions = paddle.zeros((batch_size,self.num_priors,self.n_offsets + 1 + 2 + 1 + 2))

        predictions[:, :, :2] = cls_logits
        predictions[:, :,2:5] += reg[:, :, :3]  # also reg theta angle here
        predictions[:, :, 5] = reg[:, :, 3]  # length

        def tran_tensor(t):
            return t.unsqueeze(2).clone().tile((1, 1, self.n_offsets))

        predictions[..., 6:] = (
            tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
            ((1 - self.prior_ys.tile((batch_size, self.num_priors, 1)) -
                tran_tensor(predictions[..., 2])) * self.img_h /
                paddle.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

        predictions[..., 6:] += reg[..., 4:]

        return dict(predictions=predictions)

    def loss(self,output,batch,):
        cls_loss_weight = self.cfg.cls_loss_weight
        xyt_loss_weight = self.cfg.xyt_loss_weight
        iou_loss_weight = self.cfg.iou_loss_weight

        predictions_ = output['predictions']
        targets = batch['lane_line'].clone()
        cls_criterion = self.focal_loss
        cls_loss = paddle.to_tensor(0.0)
        reg_xytl_loss = paddle.to_tensor(0.0)
        iou_loss = paddle.to_tensor(0.0)

        for predictions, target in zip(predictions_, targets):  
            target = target[target[:, 1] == 1]

            if len(target) == 0:
                # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                cls_target = paddle.zeros((predictions.shape[0],)).astype("int64")
                cls_pred = predictions[:, :2]
                cls_loss = cls_loss + cls_criterion(
                    cls_pred, cls_target).sum()
                continue                  

            with paddle.no_grad():
                matched_row_inds, matched_col_inds = HungarianLaneAssigner(
                    predictions, target, self.img_w, self.img_h)

            # classification targets
            cls_target = paddle.zeros((predictions.shape[0],)).astype("int64")
            cls_target[matched_row_inds] = 1
            cls_pred = predictions[:, :2]

            # regression targets -> [start_y, start_x, theta] (all transformed to absolute values), only on matched pairs
            reg_yxtl = paddle.index_select(predictions,matched_row_inds,axis = 0)[...,2:6]
            reg_yxtl[:, 0] *= self.n_strips
            reg_yxtl[:, 1] *= (self.img_w - 1)
            reg_yxtl[:, 2] *= 180
            reg_yxtl[:, 3] *= self.n_strips

            target_yxtl = paddle.index_select(target,matched_col_inds,axis=0)[...,2:6].clone()

            reg_pred = paddle.index_select(predictions,matched_row_inds,axis=0)[...,6:]
            reg_pred *= (self.img_w - 1)
            reg_targets = paddle.index_select(target,matched_col_inds,axis = 0)[...,6:].clone()

            with paddle.no_grad():
                predictions_starts = paddle.clip(
                    (paddle.index_select(predictions,matched_row_inds,axis = 0)[...,2] *
                        self.n_strips).round().astype('int64'), 0, self.n_strips)  # ensure the predictions starts is valid
                target_starts = (paddle.index_select(target,matched_col_inds,axis = 0)[...,2] *
                                    self.n_strips).round().astype('int64')
                target_yxtl[:, -1] -= (predictions_starts - target_starts
                                        )  # reg length

            # Loss calculation
            cls_loss = cls_loss + cls_criterion(cls_pred, cls_target).sum(
            ) / target.shape[0]

            target_yxtl[:, 0] *= self.n_strips
            target_yxtl[:, 2] *= 180
            reg_xytl_loss = reg_xytl_loss + F.smooth_l1_loss(
                reg_yxtl, target_yxtl,
                reduction='none').mean()
            
            iou_loss = iou_loss + self.liou_loss(pred = reg_pred, target = reg_targets,img_w = self.img_w, length=15)

        cls_loss /= len(targets)
        reg_xytl_loss /= len(targets)
        iou_loss /= len(targets)

        loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'loss': loss,
            'cls_loss': cls_loss * cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
            'iou_loss': iou_loss * iou_loss_weight
        }

        return return_value
    
    def predictions_to_pred(self, predictions):
        '''
        Convert predictions to internal Lane structure for evaluation.
        '''
        prior_ys = self.prior_ys.astype('float64')
        lanes = []
        for lane in predictions:
            lane_xs = lane[6:]  # normalized value
            start = min(max(0, int(round(lane[2].item() * self.n_strips))),
                        self.n_strips)
            length = int(round(lane[5].item()))
            end = start + length - 1
            end = min(end, len(prior_ys) - 1)
            # end = label_end
            # if the prediction does not start at the bottom of the image,
            # extend its prediction until the x is outside the image
            if start > 0:
                mask = ((lane_xs[:start] >= 0.) &
                        (lane_xs[:start] <= 1.)).cpu().detach().numpy()[::-1]
                mask = ~((mask.cumprod()[::-1]).astype(bool))
                lane_xs[:start][mask] = -2
            if end < len(prior_ys) - 1:
                lane_xs[end + 1:] = -2

            lane_ys = prior_ys[lane_xs >= 0].clone()
            lane_xs = lane_xs[lane_xs >= 0]
            lane_xs = lane_xs.flip(axis=0).astype('float64')
            lane_ys = lane_ys.flip(axis=0)

            lane_ys = (lane_ys * (self.cfg.ori_img_h - self.cfg.cut_height) +
                       self.cfg.cut_height) / self.cfg.ori_img_h
            if len(lane_xs) <= 1:
                continue
            points = paddle.stack(
                (lane_xs.reshape((-1, 1)), lane_ys.reshape((-1, 1))),
                axis=1).squeeze(2)
            lane = Lane(points=points.cpu().numpy(),
                        metadata={
                            'start_x': lane[3],
                            'start_y': lane[2],
                            'conf': lane[1]
                        })
            lanes.append(lane)
        return lanes

    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        output = output['predictions']
#        print("output shape:",output.shape)
        softmax = nn.Softmax(axis=1)

        decoded = []
        for predictions in output:
            # filter out the conf lower than conf threshold
            threshold = self.cfg.test_parameters.conf_threshold
            scores = softmax(predictions[:, :2])[:, 1]
            keep_inds = scores >= threshold
            predictions = predictions[keep_inds]
            scores = scores[keep_inds]

            if predictions.shape[0] == 0:
                decoded.append([])
                continue

            predictions[:, 5] = paddle.round(predictions[:, 5] * self.n_strips)
            if as_lanes:
                pred = self.predictions_to_pred(predictions)
            else:
                pred = predictions
            decoded.append(pred)

        return decoded