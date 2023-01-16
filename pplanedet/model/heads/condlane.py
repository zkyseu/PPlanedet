import paddle
from paddle import nn
import paddle.nn.functional as F
from ..common_model import ConvModule
from ..lane import Lane
import numpy as np
import math
import random
import copy

from ..builder import HEADS,build_loss


class CondLaneLoss(paddle.nn.Layer):

    def __init__(self, 
                 weights, 
                 num_lane_cls,
                 crit_loss,
                 crit_kp_loss,
                 crit_ce_loss,
                 cfg):
        """
        Args:
            weights is a dict which sets the weight of the loss
            eg. {hm_weight: 1, kp_weight: 1, ins_weight: 1}
        """
        super(CondLaneLoss, self).__init__()
        self.crit = build_loss(crit_loss,cfg)
        self.crit_kp = build_loss(crit_kp_loss,cfg)
        self.crit_ce = build_loss(crit_ce_loss,cfg)

        hm_weight = 1.
        kps_weight = 0.4
        row_weight = 1.0
        range_weight = 1.0

        self.hm_weight = weights[
            'hm_weight'] if 'hm_weight' in weights else hm_weight
        self.kps_weight = weights[
            'kps_weight'] if 'kps_weight' in weights else kps_weight
        self.row_weight = weights[
            'row_weight'] if 'row_weight' in weights else row_weight
        self.range_weight = weights[
            'range_weight'] if 'range_weight' in weights else range_weight

    def forward(self, output, meta, **kwargs):
        hm, kps, mask, lane_range = output[:4]
        hm_loss, kps_loss, row_loss, range_loss = 0, 0, 0, 0
        hm = paddle.clip(F.sigmoid(hm), min=1e-4, max=1 - 1e-4)

        if self.hm_weight > 0:
            hm_loss += self.crit(hm, kwargs['gt_hm'])

        if self.kps_weight > 0:
            kps_loss += self.crit_kp(kps, kwargs['gt_reg'],
                                     kwargs['gt_reg_mask'])

        if self.row_weight > 0:
            mask_softmax = F.softmax(mask, axis=3)
            pos = compute_locations2(
                mask_softmax.shape,)
            row_pos = paddle.sum(pos * mask_softmax, axis=3) + 0.5
            #print(row_pos*kwargs['gt_row_masks'], kwargs['gt_rows']*kwargs['gt_row_masks'])
            row_loss += self.crit_kp(row_pos, kwargs['gt_rows'],
                                     kwargs['gt_row_masks'])
            #print(row_loss)

        if self.range_weight > 0:
            gt_ranges = kwargs['gt_ranges']
            lane_range = lane_range.transpose((0,2,1))
            range_loss = self.crit_ce(lane_range, gt_ranges)

        # Only non-zero losses are valid, otherwise multi-GPU training will report an error
        losses = {}
        loss = 0.
        if self.hm_weight:
            losses['hm_loss'] = self.hm_weight * hm_loss
        if self.kps_weight:
            losses['kps_loss'] = self.kps_weight * kps_loss
        if self.row_weight > 0:
            losses['row_loss'] = self.row_weight * row_loss
        if self.range_weight > 0:
            losses['range_loss'] = self.range_weight * range_loss

        for key, value in losses.items():
            loss += value
        losses['loss'] = loss
        return losses


def adjust_result(lanes, crop_bbox, img_shape, tgt_shape=(590, 1640)):

    def in_range(pt, img_shape):
        if pt[0] >= 0 and pt[0] < img_shape[1] and pt[1] >= 0 and pt[
                1] <= img_shape[0]:
            return True
        else:
            return False

    left, top, right, bot = crop_bbox
    h_img, w_img = img_shape[:2]
    crop_width = right - left
    crop_height = bot - top
    ratio_x = crop_width / w_img
    ratio_y = crop_height / h_img
    offset_x = (tgt_shape[1] - crop_width) / 2
    offset_y = top

    results = []
    if lanes is not None:
        for key in range(len(lanes)):
            pts = []
            for pt in lanes[key]['points']:
                pt[0] = float(pt[0] * ratio_x + offset_x)
                pt[1] = float(pt[1] * ratio_y + offset_y)
                pts.append(pt)
            if len(pts) > 1:
                results.append(pts)
    return results

class CondLanePostProcessor(object):

    def __init__(self,
                 mask_size,
                 hm_thr=0.5,
                 min_points=5,
                 hm_downscale=16,
                 mask_downscale=8,
                 use_offset=True,
                 **kwargs):
        self.hm_thr = hm_thr
        self.min_points = min_points
        self.hm_downscale = hm_downscale
        self.mask_downscale = mask_downscale
        self.use_offset = use_offset
        self.horizontal_id = [5]

        self.nms_groups = [[1]]
        if 'nms_thr' in kwargs:
            self.nms_thr = kwargs['nms_thr']
        else:
            self.nms_thr = 3
        self.pos = self.compute_locations(
            mask_size,).tile((100, 1, 1))

    def nms_seeds_tiny(self, seeds, thr):

        def cal_dis(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def search_groups(coord, groups, thr):
            for idx_group, group in enumerate(groups):
                for group_point in group:
                    group_point_coord = group_point[1]
                    if cal_dis(coord, group_point_coord) <= thr:
                        return idx_group
            return -1

        def choose_highest_score(group):
            highest_score = -1
            highest_idx = -1
            for idx, _, score in group:
                if score > highest_score:
                    highest_idx = idx
            return highest_idx

        def update_coords(points_info, thr=4):
            groups = []
            keep_idx = []
            for idx, (coord, score) in enumerate(points_info):
                idx_group = search_groups(coord, groups, thr)
                if idx_group < 0:
                    groups.append([(idx, coord, score)])
                else:
                    groups[idx_group].append((idx, coord, score))
            for group in groups:
                choose_idx = choose_highest_score(group)
                if choose_idx >= 0:
                    keep_idx.append(choose_idx)
            return keep_idx

        points = [(item['coord'], item['score']) for item in seeds]
        keep_idxes = update_coords(points, thr=thr)
        update_seeds = [seeds[idx] for idx in keep_idxes]
        return update_seeds

    def compute_locations(self, shape):
        pos = paddle.arange(
            0, shape[-1], step=1, dtype=paddle.float32,)
        pos = pos.reshape((1, 1, -1))
        pos = pos.tile((shape[0], shape[1], 1))
        return pos

    def lane_post_process_all(self, masks, regs, scores, ranges, downscale,
                              seeds):

        def get_range(ranges):
            max_rows = ranges.shape[1]
            lane_ends = []
            for idx, lane_range in enumerate(ranges):
                min_idx = max_idx = None
                for row_idx, valid in enumerate(lane_range):
                    if valid:
                        min_idx = row_idx - 1
                        break
                for row_idx, valid in enumerate(lane_range[::-1]):
                    if valid:
                        max_idx = len(lane_range) - row_idx
                        break
                if max_idx is not None:
                    max_idx = min(max_rows - 1, max_idx)
                if min_idx is not None:
                    min_idx = max(0, min_idx)
                lane_ends.append([min_idx, max_idx])
            return lane_ends

        lanes = []
        num_ins = masks.shape[0]
        mask_softmax = F.softmax(masks, axis=-1)
        row_pos = paddle.sum(
            self.pos[:num_ins] * mask_softmax,
            axis=2).detach().cpu().numpy().astype(np.int32)
        # row_pos = paddle.argmax(masks, -1).detach().cpu().numpy()
        ranges = paddle.argmax(ranges, 1).detach().cpu().numpy()
        lane_ends = get_range(ranges)
        regs = regs.detach().cpu().numpy()
        num_lanes, height, width = masks.shape
        # with Timer("post process time: %f"):

        for lane_idx in range(num_lanes):
            if lane_ends[lane_idx][0] is None or lane_ends[lane_idx][1] is None:
                continue
            selected_ys = np.arange(lane_ends[lane_idx][0],
                                    lane_ends[lane_idx][1] + 1)
            selected_col_idx = row_pos[lane_idx, :]
            selected_xs = selected_col_idx[selected_ys]
            if self.use_offset:
                selected_regs = regs[lane_idx, selected_ys, selected_xs]
            else:
                selected_regs = 0.5
            selected_xs = np.expand_dims(selected_xs, 1)
            selected_ys = np.expand_dims(selected_ys, 1)
            points = np.concatenate((selected_xs, selected_ys),
                                    1).astype(np.float32)
            points[:, 0] = points[:, 0] + selected_regs
            points *= downscale

            if len(points) > 1:
                lanes.append(
                    dict(
                        id_class=1,
                        points=points,
                        score=scores[lane_idx],
                        seed=seeds[lane_idx]))
        return lanes

    def collect_seeds(self, seeds):
        masks = []
        regs = []
        scores = []
        ranges = []
        for seed in seeds:
            masks.append(seed['mask'])
            regs.append(seed['reg'])
            scores.append(seed['score'])
            ranges.append(seed['range'])
        if len(masks) > 0:
            masks = paddle.concat(masks, 0)
            regs = paddle.concat(regs, 0)
            ranges = paddle.concat(ranges, 0)
            return masks, regs, scores, ranges
        else:
            return None

    def extend_line(self, line, dis=100):
        extended = copy.deepcopy(line)
        start = line[-2]
        end = line[-1]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        norm = math.sqrt(dx**2 + dy**2)
        dx = dx / norm
        dy = dy / norm
        extend_point = [start[0] + dx * dis, start[1] + dy * dis]
        extended.append(extend_point)
        return extended

    def __call__(self, output, downscale):
        lanes = []
        # with Timer("Elapsed time in tiny nms: %f"):
        seeds = self.nms_seeds_tiny(output, self.nms_thr)
        if len(seeds) == 0:
            return [], seeds
        collection = self.collect_seeds(seeds)
        if collection is None:
            return [], seeds
        masks, regs, scores, ranges = collection
        lanes = self.lane_post_process_all(masks, regs, scores, ranges,
                                           downscale, seeds)
        return lanes, seeds


def fill_fc_weights(layers):
    for m in layers.sublayers():
        if isinstance(m, nn.Conv2D):
            if m.bias is not None:
                zeros = nn.initializer.Constant(value=0)
                zeros(m.bias)


def fill_up_weights(up):
    w = up.weight
    f = math.ceil(w.shape[2] / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.shape[2]):
        for j in range(w.shape[3]):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.shape[0]):
        w[c, 0, :, :] = w[0, 0, :, :]

@HEADS.register()
class CtnetHead(nn.Layer):
    def __init__(self, 
                 heads, 
                 channels_in,
                 final_kernel=1, 
                 head_conv=256,
                 cfg = None):
        super(CtnetHead, self).__init__()
        
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head_conv > 0:
              fc = nn.Sequential(
                  nn.Conv2D(channels_in, head_conv,
                    kernel_size=3, padding=1, bias_attr=True),
                  nn.ReLU(),
                  nn.Conv2D(head_conv, classes,
                    kernel_size=final_kernel, stride=1,
                    padding=final_kernel // 2, bias_attr=True))
              if 'hm' in head:
                inits = nn.initializer.Constant(value = -2.19)
                inits(fc[-1].bias)
              else:
                fill_fc_weights(fc)
            else:
              fc = nn.Conv2D(channels_in, classes,
                  kernel_size=final_kernel, stride=1,
                  padding=final_kernel // 2, bias_attr=True)
              if 'hm' in head:
                inits = nn.initializer.Constant(value = -2.19)
                inits(fc[-1].bias)
              else:
                fill_fc_weights(fc)
            self.__setattr__(head, fc)

    def forward(self, x, **kwargs):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[0]
        z = {}
        for head in self.heads:
            z[head] = self.__getattr__(head)(x)
        return z
    
    def init_weights(self):
        # ctnet_head will init weights during building
        pass

def parse_dynamic_params(params,
                         channels,
                         weight_nums,
                         bias_nums,
                         out_channels=1,
                         mask=True):
    assert len(params.shape) == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.shape[1] == sum(weight_nums) + sum(bias_nums)
    # params: (num_ins x n_param)

    num_insts = params.shape[0]
    num_layers = len(weight_nums)

    params_splits = list(paddle.split(params,weight_nums+bias_nums,axis=1))

    # params_splits = list(
    #     paddle.split_with_sizes(params, weight_nums + bias_nums, dim=1))

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]
    if mask:
        bias_splits[-1] = bias_splits[-1] - 2.19

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape((
                num_insts * channels, -1, 1, 1))
            bias_splits[l] = bias_splits[l].reshape((num_insts * channels))
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape((
                num_insts * out_channels, -1, 1, 1))
            bias_splits[l] = bias_splits[l].reshape((num_insts * out_channels,))

    return weight_splits, bias_splits


def compute_locations(h, w, stride,):
    shifts_x = paddle.arange(
        0, w * stride, step=stride, dtype=paddle.float32)
    shifts_y = paddle.arange(
        0, h * stride, step=stride, dtype=paddle.float32)
    shift_y, shift_x = paddle.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape((-1,))
    shift_y = shift_y.reshape((-1,))
    locations = paddle.stack((shift_x, shift_y), axis=1) + stride // 2
    return locations

def compute_locations2(shape,):
    pos = paddle.arange(
        0, shape[-1], step=1, dtype=paddle.float32,)
    pos = pos.reshape((1, 1, 1, -1))
    pos = pos.tile((shape[0], shape[1], shape[2], 1))
    return pos


class DynamicMaskHead(nn.Layer):

    def __init__(self,
                 num_layers,
                 channels,
                 in_channels,
                 mask_out_stride,
                 weight_nums,
                 bias_nums,
                 disable_coords=False,
                 shape=(160, 256),
                 out_channels=1,
                 compute_locations_pre=True,
                 location_configs=None):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.in_channels = in_channels
        self.mask_out_stride = mask_out_stride
        self.disable_coords = disable_coords

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)
        self.out_channels = out_channels
        self.compute_locations_pre = compute_locations_pre
        self.location_configs = location_configs

        if compute_locations_pre and location_configs is not None:
            N, _, H, W = location_configs['size']
            locations = compute_locations(H, W, stride=1)

            locations = locations.unsqueeze(0).transpose((
                0, 2, 1)).reshape((1, 2, H, W))
            locations[...,0, :, :] /= H
            locations[...,1, :, :] /= W
            locations = locations.tile((N, 1, 1, 1))
            self.locations = locations

    def forward(self, x, mask_head_params, num_ins, idx=0, is_mask=True):

        N, _, H, W = x.shape
        if not self.disable_coords:
            if self.compute_locations_pre and self.location_configs is not None:
                if self.locations.shape[0] != N:
                    locations = self.locations[idx].unsqueeze(0)
                else:
                    locations = self.locations
            else:
                locations = compute_locations(
                    x.shape[2], x.shape[3], stride=1)
                locations = locations.unsqueeze(0).transpose((
                    0, 2, 1)).reshape((1, 2, H, W))
                locations[:0, :, :] /= H
                locations[:1, :, :] /= W
                locations = locations.tile((N, 1, 1, 1))

            #relative_coords = relative_coords.to(dtype=mask_feats.dtype)
            x = paddle.concat([locations, x], axis=1)
        mask_head_inputs = []
        for idx in range(N):
            mask_head_inputs.append(x[idx:idx + 1, ...].tile((
                1, num_ins[idx], 1, 1)))
        mask_head_inputs = paddle.concat(mask_head_inputs, 1)
        num_insts = sum(num_ins)
        mask_head_inputs = mask_head_inputs.reshape((1, -1, H, W))
        weights, biases = parse_dynamic_params(
            mask_head_params,
            self.channels,
            self.weight_nums,
            self.bias_nums,
            out_channels=self.out_channels,
            mask=is_mask)
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights,
                                              biases, num_insts)
        mask_logits = mask_logits.reshape((1, -1, H, W))
        return mask_logits

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert len(features.shape) == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=num_insts)
            if i < n_layers - 1:
                x = F.relu(x)
        return x


class MLP(nn.Layer):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.LayerList(
            nn.Conv1D(n, k, 1)
            for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


@HEADS.register()
class CondLaneHead(nn.Layer):

    def __init__(self,
                 heads,
                 in_channels,
                 num_classes,
                 head_channels=64,
                 head_layers=1,
                 reg_branch_channels = None,
                 disable_coords=False,
                 branch_in_channels=288,
                 branch_channels=64,
                 branch_out_channels=64,
                 branch_num_conv=1,
                 norm=True,
                 hm_idx=-1,
                 mask_idx=0,
                 compute_locations_pre=True,
                 location_configs=None,
                 mask_norm_act=True,
                 regression=True,
                 crit_loss = None,
                 crit_kp_loss = None,
                 crit_ce_loss =None,
		         cfg=None):
        super(CondLaneHead, self).__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.hm_idx = hm_idx
        self.mask_idx = mask_idx
        self.regression = regression
        if mask_norm_act:
            final_norm_cfg = True
            final_act_cfg = 'relu'
        else:
            final_norm_cfg = None
            final_act_cfg = None
        # mask branch
        mask_branch = []
        mask_branch.append(
            ConvModule(
                sum(in_channels),
                branch_channels,
                kernel_size=3,
                norm=norm))
        for i in range(branch_num_conv):
            mask_branch.append(
                ConvModule(
                    branch_channels,
                    branch_channels,
                    kernel_size=3,
                    norm=norm))
        mask_branch.append(
            ConvModule(
                branch_channels,
                branch_out_channels,
                kernel_size=3,
                norm=final_norm_cfg,
                act=final_act_cfg))
        self.add_sublayer('mask_branch', nn.Sequential(*mask_branch))

        self.mask_weight_nums, self.mask_bias_nums = self.cal_num_params(
            head_layers, disable_coords, head_channels, out_channels=1)

        # print('mask: ', self.mask_weight_nums, self.mask_bias_nums) # 66 1

        self.num_mask_params = sum(self.mask_weight_nums) + sum(
            self.mask_bias_nums)

        self.reg_weight_nums, self.reg_bias_nums = self.cal_num_params(
            head_layers, disable_coords, head_channels, out_channels=1)
        # print('reg: ', self.reg_weight_nums, self.reg_bias_nums) # 66 1

        self.num_reg_params = sum(self.reg_weight_nums) + sum(
            self.reg_bias_nums)
        if self.regression:
            self.num_gen_params = self.num_mask_params + self.num_reg_params
        else:
            self.num_gen_params = self.num_mask_params
            self.num_reg_params = 0

        self.mask_head = DynamicMaskHead(
            head_layers,
            branch_out_channels,
            branch_out_channels,
            1,
            self.mask_weight_nums,
            self.mask_bias_nums,
            disable_coords=False,
            compute_locations_pre=compute_locations_pre,
            location_configs=location_configs)
        if self.regression:
            self.reg_head = DynamicMaskHead(
                head_layers,
                branch_out_channels,
                branch_out_channels,
                1,
                self.reg_weight_nums,
                self.reg_bias_nums,
                disable_coords=False,
                out_channels=1,
                compute_locations_pre=compute_locations_pre,
                location_configs=location_configs)
        if 'params' not in heads:
            heads['params'] = num_classes * (
                self.num_mask_params + self.num_reg_params)

        self.ctnet_head = CtnetHead(
            heads,
            channels_in=branch_in_channels,
            final_kernel=1,
            # head_conv=64,)
            head_conv=branch_in_channels)

        self.feat_width = location_configs['size'][-1]
        self.mlp = MLP(self.feat_width, 64, 2, 2)

        self.post_process = CondLanePostProcessor(
                mask_size=self.cfg.mask_size, hm_thr=0.5, use_offset=True,
                nms_thr=4)

        self.loss_impl = CondLaneLoss(cfg.loss_weights, 
                                      1,
                                      crit_loss=crit_loss,
                                      crit_ce_loss=crit_ce_loss,
                                      crit_kp_loss=crit_kp_loss,
                                      cfg=cfg)
    
    def loss(self, output, batch):
        img_metas = batch.pop('img_metas')
        return self.loss_impl(output, img_metas, **batch)


    def cal_num_params(self,
                       num_layers,
                       disable_coords,
                       channels,
                       out_channels=1):
        weight_nums, bias_nums = [], []
        for l in range(num_layers):
            if l == num_layers - 1:
                if num_layers == 1:
                    weight_nums.append((channels + 2) * out_channels)
                else:
                    weight_nums.append(channels * out_channels)
                bias_nums.append(out_channels)
            elif l == 0:
                if not disable_coords:
                    weight_nums.append((channels + 2) * channels)
                else:
                    weight_nums.append(channels * channels)
                bias_nums.append(channels)

            else:
                weight_nums.append(channels * channels)
                bias_nums.append(channels)
        return weight_nums, bias_nums

    def parse_gt(self, gts, device=None):
        reg = (paddle.to_tensor(gts['reg'])).unsqueeze(0)
        reg_mask = (paddle.to_tensor(gts['reg_mask'])).unsqueeze(0)
        row = (paddle.to_tensor(
            gts['row'])).unsqueeze(0).unsqueeze(0)
        row_mask = (paddle.to_tensor(
            gts['row_mask'])).unsqueeze(0).unsqueeze(0)
        if 'range' in gts:
            lane_range = paddle.to_tensor(gts['range']) # new add: squeeze 
            #lane_range = (gts['range']).to(device).squeeze(0) # new add: squeeze 
        else:
            mask = None
            lane_range = paddle.zeros((1, mask.shape[-2]),
                                     dtype=paddle.int64)
        return reg, reg_mask, row, row_mask, lane_range

    def parse_pos(self, gt_masks, hm_shape, device, mask_shape=None):
        b = len(gt_masks)
        n = self.num_classes
        hm_h, hm_w = hm_shape[:2]
        if mask_shape is None:
            mask_h, mask_w = hm_shape[:2]
        else:
            mask_h, mask_w = mask_shape[:2]
        poses = []
        regs = []
        reg_masks = []
        rows = []
        row_masks = []
        lane_ranges = []
        labels = []
        num_ins = []
        for idx, m_img in enumerate(gt_masks):
            num = 0
            for m in m_img:
                gts = self.parse_gt(m, device=device)
                reg, reg_mask, row, row_mask, lane_range = gts
                label = m['label']
                num += len(m['points'])
                for p in m['points']:
                    pos = idx * n * hm_h * hm_w + label * hm_h * hm_w + p[
                        1] * hm_w + p[0]
                    poses.append(pos)
                for i in range(len(m['points'])):
                    labels.append(label)
                    regs.append(reg)
                    reg_masks.append(reg_mask)
                    rows.append(row)
                    row_masks.append(row_mask)
                    lane_ranges.append(lane_range)

            if num == 0:
                reg = paddle.zeros((1, 1, mask_h, mask_w))
                reg_mask = paddle.zeros((1, 1, mask_h, mask_w))
                row = paddle.zeros((1, 1, mask_h))
                row_mask = paddle.zeros((1, 1, mask_h))
                lane_range = paddle.zeros((1, mask_h),
                                         dtype=paddle.int64)
                label = 0
                pos = idx * n * hm_h * hm_w + random.randint(
                    0, n * hm_h * hm_w - 1)
                num = 1
                labels.append(label)
                poses.append(pos)
                regs.append(reg)
                reg_masks.append(reg_mask)
                rows.append(row)
                row_masks.append(row_mask)
                lane_ranges.append(lane_range)

            num_ins.append(num)

        if len(regs) > 0:
            regs = paddle.concat(regs, 1)
            reg_masks = paddle.concat(reg_masks, 1)
            rows = paddle.concat(rows, 1)
            row_masks = paddle.concat(row_masks, 1)
            lane_ranges = paddle.concat(lane_ranges, 0)

        gts = dict(
            gt_reg=regs,
            gt_reg_mask=reg_masks,
            gt_rows=rows,
            gt_row_masks=row_masks,
            gt_ranges=lane_ranges)

        return poses, labels, num_ins, gts


    def ctdet_decode(self, heat, thr=0.1):
        heat = heat.unsqueeze(0)
    
        def _nms(heat, kernel=3):
            pad = (kernel - 1) // 2

            hmax = nn.functional.max_pool2d(
                heat, (kernel, kernel), stride=1, padding=pad)
            keep = (hmax == heat).astype('float')
            return heat * keep

        def _format(heat, inds):
            ret = []
            for y, x, c in zip(inds[0], inds[1], inds[2]):
                id_class = c + 1
                coord = [x, y]
                score = heat[y, x, c]
                ret.append({
                    'coord': coord,
                    'id_class': id_class,
                    'score': score
                })
            return ret

        heat_nms = _nms(heat)
        heat_nms = heat_nms.squeeze(0)
        # print(heat.shape, heat_nms.shape)
        heat_nms = heat_nms.transpose((1, 2, 0)).detach().cpu().numpy()
        inds = np.where(heat_nms > thr)
        seeds = _format(heat_nms, inds)
        # heat_nms = heat_nms.permute(0, 2, 3, 1).detach().cpu().numpy()
        # inds = np.where(heat_nms > thr)
        # print(len(inds))
        # seeds = _format(heat_nms, inds)
        return seeds

    def forward_train(self, output, batch):
        img_metas = batch['img_metas']
        img_metas = img_metas._data[0]
        gt_batch_masks = [m['gt_masks'] for m in img_metas]
        hm_shape = img_metas[0]['hm_shape']
        mask_shape = img_metas[0]['mask_shape']
        inputs = output
        pos, labels, num_ins, gts = self.parse_pos(
            gt_batch_masks, hm_shape, inputs[0], mask_shape=mask_shape)
        batch.update(gts)
        
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]

        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.shape[0]

        # f_mask
        z = self.ctnet_head(f_hm)
        hm, params = z['hm'], z['params']
        h_hm, w_hm = hm.shape[2:]
        h_mask, w_mask = f_mask.shape[2:]
        params = params.reshape((m_batchsize, self.num_classes, -1, h_hm, w_hm))
        mask_branch = self.mask_branch(f_mask)
        reg_branch = mask_branch
        # reg_branch = self.reg_branch(f_mask)
        params = params.transpose((0, 1, 3, 4,
                                2)).reshape((-1, self.num_gen_params))

        pos_tensor = paddle.to_tensor(np.array(pos, dtype=np.float64)).astype('int64').unsqueeze(1)

        pos_tensor = pos_tensor.expand((-1, self.num_gen_params))
        mask_pos_tensor = pos_tensor[:, :self.num_mask_params]
        reg_pos_tensor = pos_tensor[:, self.num_mask_params:]
        if pos_tensor.shape[0] == 0:
            masks = None
            feat_range = None
        else:
            mask_params = params[:, :self.num_mask_params].take_along_axis(
                mask_pos_tensor,0)
            masks = self.mask_head(mask_branch, mask_params, num_ins)
            if self.regression:
                reg_params = params[:, self.num_mask_params:].take_along_axis(
                    reg_pos_tensor,0)
                regs = self.reg_head(reg_branch, reg_params, num_ins)
            else:
                regs = masks
            # regs = regs.view(sum(num_ins), 1, h_mask, w_mask)
            feat_range = masks.transpose((0, 1, 3,
                                       2)).reshape((sum(num_ins), w_mask, h_mask))
            feat_range = self.mlp(feat_range)
        batch.update(dict(mask_branch=mask_branch, reg_branch=reg_branch))
        return hm, regs, masks, feat_range, [mask_branch, reg_branch]

    def forward_test(
            self,
            inputs,
            hack_seeds=None,
            hm_thr=0.5,
    ):

        def parse_pos(seeds, batchsize, num_classes, h, w):
            pos_list = [[p['coord'], p['id_class'] - 1] for p in seeds]
            poses = []
            for p in pos_list:
                [c, r], label = p
                pos = label * h * w + r * w + c
                poses.append(pos)
            poses = paddle.to_tensor(np.array(
                poses, np.long)).astype('int64').unsqueeze(1)
            return poses

        # with Timer("Elapsed time in stage1: %f"):  # ignore
        x_list = list(inputs)
        f_hm = x_list[self.hm_idx]
        f_mask = x_list[self.mask_idx]
        m_batchsize = f_hm.shape[0]
        f_deep = f_mask
        m_batchsize = f_deep.shape[0]
        # with Timer("Elapsed time in ctnet_head: %f"):  # 0.3ms
        z = self.ctnet_head(f_hm)
        h_hm, w_hm = f_hm.shape[2:]
        h_mask, w_mask = f_mask.shape[2:]
        hms, params = z['hm'], z['params']
        hms = paddle.clip(F.sigmoid(hms), min=1e-4, max=1 - 1e-4)
        params = params.reshape((m_batchsize, self.num_classes, -1, h_hm, w_hm))
        # with Timer("Elapsed time in two branch: %f"):  # 0.6ms
        mask_branchs = self.mask_branch(f_mask)
        reg_branchs = mask_branchs
        # reg_branch = self.reg_branch(f_mask)
        params = params.transpose((0, 1, 3, 4,
                                2)).reshape((m_batchsize, -1, self.num_gen_params))

        batch_size, num_classes, h, w = hms.shape
        # with Timer("Elapsed time in ct decode: %f"):  # 0.2ms
        out_seeds, out_hm = [], []
        idx = 0
        for hm, param, mask_branch, reg_branch in zip(hms, params, mask_branchs, reg_branchs):
            mask_branch = mask_branch.unsqueeze(0)
            reg_branch = reg_branch.unsqueeze(0)
            seeds = self.ctdet_decode(hm, thr=hm_thr)
            if hack_seeds is not None:
                seeds = hack_seeds
            # with Timer("Elapsed time in stage2: %f"):  # 0.08ms
            pos_tensor = parse_pos(seeds, batch_size, num_classes, h, w)
            pos_tensor = pos_tensor.expand((-1, self.num_gen_params))
            num_ins = [pos_tensor.shape[0]]
            mask_pos_tensor = pos_tensor[:, :self.num_mask_params]
            if self.regression:
                reg_pos_tensor = pos_tensor[:, self.num_mask_params:]
            # with Timer("Elapsed time in stage3: %f"):  # 0.8ms
            if pos_tensor.shape[0] == 0:
                seeds = []
            else:
                mask_params = param[:, :self.num_mask_params].take_along_axis(
                    mask_pos_tensor,0)
                # with Timer("Elapsed time in mask_head: %f"):  #0.3ms
                masks = self.mask_head(mask_branch, mask_params, num_ins, idx)
                if self.regression:
                    reg_params = param[:, self.num_mask_params:].take_along_axis(
                        reg_pos_tensor,0)
                    # with Timer("Elapsed time in reg_head: %f"):  # 0.25ms
                    regs = self.reg_head(reg_branch, reg_params, num_ins, idx)
                else:
                    regs = masks
                feat_range = masks.transpose((0, 1, 3,
                                           2)).reshape((sum(num_ins), w_mask, h_mask))
                feat_range = self.mlp(feat_range)
                for i in range(len(seeds)):
                    seeds[i]['reg'] = regs[0, i:i + 1, :, :]
                    m = masks[0, i:i + 1, :, :]
                    seeds[i]['mask'] = m
                    seeds[i]['range'] = feat_range[i:i + 1]
            out_seeds.append(seeds)
            out_hm.append(hm)
            idx+=1
        output = {'seeds': out_seeds, 'hm': out_hm}
        return output


    def forward(
            self,
            x_list,
            **kwargs):
        if self.training:
            return self.forward_train(x_list, kwargs['batch'])
        return self.forward_test(x_list, )

    def get_lanes(self, output):
        out_seeds, out_hm = output['seeds'], output['hm']
        ret = []
        for seeds, hm in zip(out_seeds, out_hm):
            lanes, seed = self.post_process(seeds, self.cfg.mask_down_scale)
            result = adjust_result(
                    lanes=lanes,
                    crop_bbox=self.cfg.crop_bbox,
                    img_shape=(self.cfg.img_height, self.cfg.img_width),
                    tgt_shape=(self.cfg.ori_img_h, self.cfg.ori_img_w),
                    )
            lanes = []
            for lane in result:
                coord = []
                for x, y in lane:
                    coord.append([x, y])
                coord = np.array(coord)
                coord[:, 0] /= self.cfg.ori_img_w
                coord[:, 1] /= self.cfg.ori_img_h
                lanes.append(Lane(coord))
            ret.append(lanes)

        return ret

    def init_weights(self):
        # ctnet_head will init weights during building
        pass
