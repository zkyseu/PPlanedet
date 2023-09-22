import math
import cv2
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ..common_model import ConvModule

from ..lane import Lane
from ..common_model import assign,ROIGather,LinearModule,lane_nms
from ..losses import line_iou
from ...utils import accuracy
from paddleseg.cvlibs.param_init import constant_init,normal_init

from ..builder import HEADS,build_head,build_loss

@HEADS.register()
class CLRHead(nn.Layer):
    def __init__(self,
                 num_points=72,
                 prior_feat_channels=64,
                 fc_hidden_dim=64,
                 num_priors=192,
                 num_fc=2,
                 refine_layers=3,
                 sample_points=36,
                 seg_decoder = None,
                 cls_loss = None,
                 liou_loss = None,
                 ce_loss = None,
                 cfg=None):
        super(CLRHead, self).__init__()
        self.cfg = cfg
        self.img_w = self.cfg.img_width
        self.img_h = self.cfg.img_height
        self.n_strips = num_points - 1
        self.n_offsets = num_points
        self.num_priors = num_priors
        self.sample_points = sample_points
        self.refine_layers = refine_layers
        self.fc_hidden_axis = fc_hidden_dim

        self.register_buffer(name='sample_x_indexs', tensor=(paddle.linspace(
            0, 1, num=self.sample_points, dtype=paddle.float32) *
                                self.n_strips).astype('int64'))
        self.register_buffer(name='prior_feat_ys', tensor=paddle.flip(
            (1 - self.sample_x_indexs.astype('float32') / self.n_strips), axis=[-1]))
        self.register_buffer(name='prior_ys', tensor=paddle.linspace(1,
                                       0,
                                       num=self.n_offsets,
                                       dtype=paddle.float32))

        self.prior_feat_channels = prior_feat_channels

        self._init_prior_embeddings()
        init_priors, priors_on_featmap = self.generate_priors_from_embeddings() #None, None
        self.register_buffer(name='priors', tensor=init_priors)
        self.register_buffer(name='priors_on_featmap', tensor=priors_on_featmap)

        # generate xys for feature map
        self.seg_decoder = build_head(seg_decoder,cfg)

        reg_modules = list()
        cls_modules = list()
        for _ in range(num_fc):
            reg_modules += [*LinearModule(self.fc_hidden_axis)]
            cls_modules += [*LinearModule(self.fc_hidden_axis)]
        self.reg_modules = nn.LayerList(reg_modules)
        self.cls_modules = nn.LayerList(cls_modules)

        self.roi_gather = ROIGather(self.prior_feat_channels, self.num_priors,
                                    self.sample_points, self.fc_hidden_axis,
                                    self.refine_layers)

        self.reg_layers = nn.Linear(
            self.fc_hidden_axis, self.n_offsets + 1 + 2 +
            1,bias_attr=True)  # n offsets + 1 length + start_x + start_y + theta
        self.cls_layers = nn.Linear(self.fc_hidden_axis, 2, bias_attr=True)

        # init the weights here
        self.init_weights()

        #build loss
        self.focal_loss = build_loss(cls_loss,cfg)
        self.liou_loss = build_loss(liou_loss,cfg)
        self.criterion = build_loss(ce_loss,cfg)

    # function to init layer weights
    def init_weights(self):
        # initialize heads
        for m in self.cls_layers.parameters():
            normal_init(m, mean=0., std=1e-3)

        for m in self.reg_layers.parameters():
            normal_init(m, mean=0., std=1e-3)

    def pool_prior_features(self, batch_features, num_priors, prior_xs):
        '''
        pool prior feature from feature map.
        Args:
            batch_features (Tensor): Input feature maps, shape: (B, C, H, W) 
        '''

        batch_size = batch_features.shape[0]

        prior_xs = prior_xs.reshape((batch_size, num_priors, -1, 1))
        prior_ys = self.prior_feat_ys.tile((batch_size * num_priors,)).reshape((
            batch_size, num_priors, -1, 1)).astype(prior_xs.dtype)

        prior_xs = prior_xs * 2. - 1.
        prior_ys = prior_ys * 2. - 1.
        grid = paddle.concat((prior_xs, prior_ys), axis=-1)
        feature = F.grid_sample(batch_features, grid,
                                align_corners=True).transpose((0, 2, 1, 3))

        feature = feature.reshape((batch_size * num_priors,
                                  self.prior_feat_channels, self.sample_points,
                                  1))
        return feature

    def generate_priors_from_embeddings(self):
        predictions = self.prior_embeddings.weight  # (num_prop, 3)

        # 2 scores, 1 start_y, 1 start_x, 1 theta, 1 length, 72 coordinates, score[0] = negative prob, score[1] = positive prob
        priors = paddle.zeros((self.num_priors, 2 + 2 + 2 + self.n_offsets)).astype(predictions.dtype)

        priors[:, 2:5] = predictions.clone()
        priors[:, 6:] = (
            priors[:, 3].unsqueeze(1).clone().tile((1, self.n_offsets)) *
            (self.img_w - 1) +
            ((1 - self.prior_ys.tile((self.num_priors, 1)) -
              priors[:, 2].unsqueeze(1).clone().tile((1, self.n_offsets))) *
             self.img_h / paddle.tan(priors[:, 4].unsqueeze(1).clone().tile((
                 1, self.n_offsets)) * math.pi + 1e-5))) / (self.img_w - 1)

        # init priors on feature map
        priors_on_featmap = paddle.index_select(priors.clone(),6 + self.sample_x_indexs,axis = -1)

        return priors, priors_on_featmap

    def _init_prior_embeddings(self):
        self.prior_embeddings = nn.Embedding(self.num_priors, 3)
        bottom_priors_nums = self.num_priors * 3 // 4
        left_priors_nums, _ = self.num_priors // 8, self.num_priors // 8
        strip_size = 0.5 / (left_priors_nums // 2 - 1)
        bottom_strip_size = 1 / (bottom_priors_nums // 4 + 1)

        with paddle.no_grad():
            for i in range(left_priors_nums):
                self.prior_embeddings.weight[i, 0] = i // 2 * strip_size
                self.prior_embeddings.weight[i, 1] = 0.0
                self.prior_embeddings.weight[i,
                                             2] = 0.16 if i % 2 == 0 else 0.32

            for i in range(left_priors_nums,
                           left_priors_nums + bottom_priors_nums):
                self.prior_embeddings.weight[i, 0] = 0.0
                self.prior_embeddings.weight[i, 1] = (
                    (i - left_priors_nums) // 4 + 1) * bottom_strip_size
                self.prior_embeddings.weight[i, 2] = 0.2 * (i % 4 + 1)

            for i in range(left_priors_nums + bottom_priors_nums,
                           self.num_priors):
                self.prior_embeddings.weight[i, 0] = (
                    i - left_priors_nums - bottom_priors_nums) // 2 * strip_size
                self.prior_embeddings.weight[i, 1] = 1.0
                self.prior_embeddings.weight[i,
                                             2] = 0.68 if i % 2 == 0 else 0.84

    # forward function here
    def forward(self, x, **kwargs):
        '''
        Take pyramid features as input to perform Cross Layer Refinement and finally output the prediction lanes.
        Each feature is a 4D tensor.
        Args:
            x: input features (list[Tensor])
        Return:
            prediction_list: each layer's prediction result
            seg: segmentation result for auxiliary loss
        '''
        batch_features = list(x[len(x) - self.refine_layers:])
        batch_features.reverse()
        batch_size = batch_features[-1].shape[0]

        if self.training:
            self.priors, self.priors_on_featmap = self.generate_priors_from_embeddings()

        priors, priors_on_featmap = self.priors.tile((batch_size, 1,
                                                  1)), self.priors_on_featmap.tile((
                                                      batch_size, 1, 1))

        predictions_lists = []

        # iterative refine
        prior_features_stages = []
        for stage in range(self.refine_layers):
            num_priors = priors_on_featmap.shape[1]
            prior_xs = paddle.flip(priors_on_featmap, axis=[2])

            batch_prior_features = self.pool_prior_features(
                batch_features[stage], num_priors, prior_xs)
            prior_features_stages.append(batch_prior_features)

            fc_features = self.roi_gather(prior_features_stages,
                                          batch_features[stage], stage)

            fc_features = fc_features.reshape((num_priors, batch_size,
                                           -1)).reshape((batch_size * num_priors,
                                                       self.fc_hidden_axis))

            cls_features = fc_features.clone()
            reg_features = fc_features.clone()
            for cls_layer in self.cls_modules:
                cls_features = cls_layer(cls_features)
            for reg_layer in self.reg_modules:
                reg_features = reg_layer(reg_features)

            cls_logits = self.cls_layers(cls_features)
            reg = self.reg_layers(reg_features)

            cls_logits = cls_logits.reshape((
                batch_size, -1, cls_logits.shape[1]))  # (B, num_priors, 2)
            reg = reg.reshape((batch_size, -1, reg.shape[1]))
            
            predictions = priors.clone()
            predictions[:, :, :2] = cls_logits
            predictions[:, :,2:5] += reg[:, :, :3]  # also reg theta angle here
            predictions[:, :, 5] = reg[:, :, 3]  # length

            def tran_tensor(t):
                return t.unsqueeze(2).clone().tile((1, 1, self.n_offsets))

            predictions[..., 6:] = (
                tran_tensor(predictions[..., 3]) * (self.img_w - 1) +
                ((1 - self.prior_ys.tile((batch_size, num_priors, 1)) -
                  tran_tensor(predictions[..., 2])) * self.img_h /
                 paddle.tan(tran_tensor(predictions[..., 4]) * math.pi + 1e-5))) / (self.img_w - 1)

            prediction_lines = predictions.clone()
            predictions[..., 6:] += reg[..., 4:]

            predictions_lists.append(predictions)

            if stage != self.refine_layers - 1:
                priors = prediction_lines.detach().clone()
                priors_on_featmap = paddle.index_select(priors,6 + self.sample_x_indexs,axis = -1)

        if self.training:
            seg = None
            seg_features = paddle.concat([
                F.interpolate(feature,
                              size=[
                                  batch_features[-1].shape[2],
                                  batch_features[-1].shape[3]
                              ],
                              mode='bilinear',
                              align_corners=False)
                for feature in batch_features
            ],axis=1)
            seg = self.seg_decoder(seg_features)
            output = {}
            output.update({'predictions_lists': predictions_lists})
            output.update(seg)
            return output

        return dict(output=predictions_lists[-1])

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

    def loss(self,
             output,
             batch,
             cls_loss_weight=2.,
             xyt_loss_weight=0.5,
             iou_loss_weight=2.,
             seg_loss_weight=1.):
        if self.cfg.haskey('cls_loss_weight'):
            cls_loss_weight = self.cfg.cls_loss_weight
        if self.cfg.haskey('xyt_loss_weight'):
            xyt_loss_weight = self.cfg.xyt_loss_weight
        if self.cfg.haskey('iou_loss_weight'):
            iou_loss_weight = self.cfg.iou_loss_weight
        if self.cfg.haskey('seg_loss_weight'):
            seg_loss_weight = self.cfg.seg_loss_weight

        predictions_lists = output['predictions_lists']
        targets = batch['lane_line'].clone()
        cls_criterion = self.focal_loss
        cls_loss = paddle.to_tensor(0.0)
        reg_xytl_loss = paddle.to_tensor(0.0)
        iou_loss = paddle.to_tensor(0.0)
        cls_acc = []
        cls_acc_stage = []

        for stage in range(self.refine_layers):
            predictions_list = predictions_lists[stage]
            for predictions, target in zip(predictions_list, targets):
                target = target[target[:, 1] == 1]

                if len(target) == 0:
                    # If there are no targets, all predictions have to be negatives (i.e., 0 confidence)
                    cls_target = paddle.zeros((predictions.shape[0],)).astype("int64")
                    cls_pred = predictions[:, :2]
                    cls_loss = cls_loss + cls_criterion(
                        cls_pred, cls_target).sum()
                    continue

                with paddle.no_grad():
                    matched_row_inds, matched_col_inds = assign(
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
                cls_accuracy = accuracy(cls_pred, cls_target)
                cls_acc_stage.append(cls_accuracy)
            cls_acc.append(sum(cls_acc_stage) / (len(cls_acc_stage) + 1e-5))

        # extra segmentation loss
        seg_loss = self.criterion(output["seg"],batch["seg"].astype('int64'))

        cls_loss /= (len(targets) * self.refine_layers)
        reg_xytl_loss /= (len(targets) * self.refine_layers)
        iou_loss /= (len(targets) * self.refine_layers)

        loss = cls_loss * cls_loss_weight + reg_xytl_loss * xyt_loss_weight \
            + seg_loss * seg_loss_weight + iou_loss * iou_loss_weight

        return_value = {
            'loss': loss,
            'cls_loss': cls_loss * cls_loss_weight,
            'reg_xytl_loss': reg_xytl_loss * xyt_loss_weight,
            'seg_loss': seg_loss * seg_loss_weight,
            'iou_loss': iou_loss * iou_loss_weight
        }
        for i in range(self.refine_layers):
            if not isinstance(cls_acc[i], paddle.Tensor):
                cls_acc[i] = paddle.to_tensor(cls_acc[i])
            return_value['stage_{}_acc'.format(i)] = cls_acc[i]


        return return_value

    def get_lanes(self, output, as_lanes=True):
        '''
        Convert model output to lanes.
        '''
        output = output['output']
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
            nms_predictions = predictions.detach().clone()
            nms_predictions = paddle.concat(
                [nms_predictions[..., :4], nms_predictions[..., 5:]], axis=-1)
                
            nms_predictions[..., 4] = nms_predictions[..., 4] * self.n_strips
            nms_predictions[...,
                            5:] = nms_predictions[..., 5:] * (self.img_w - 1)


            keep = lane_nms(
                nms_predictions[..., 5:],
                scores,
                nms_overlap_thresh=self.cfg.test_parameters.nms_thres,
                top_k=self.cfg.max_lanes,
                img_w=self.img_w)

            predictions = predictions.index_select(keep)

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

