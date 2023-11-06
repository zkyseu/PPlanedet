from functools import cmp_to_key
import random
import copy
import PIL.Image
import PIL.ImageDraw

import cv2
import numpy as np
from shapely.geometry import Polygon, Point, LineString
import math

from ..data_container import DataContainer as DC
from ..builder import TRANSFORM 
from .transform import to_tensor 

def get_line_intersection(x, y, line):
    def in_line_range(val, start, end):
        s = min(start, end)
        e = max(start, end)
        if val >= s and val <= e and s != e:
            return True
        else:
            return False

    def choose_min_reg(val, ref):
        min_val = 1e5
        index = -1
        if len(val) == 0:
            return None
        else:
            for i, v in enumerate(val):
                if abs(v - ref) < min_val:
                    min_val = abs(v - ref)
                    index = i
        return val[index]

    reg_y = []
    reg_x = []

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(x, point_start[0], point_end[0]):
            k = (point_end[1] - point_start[1]) / (
                point_end[0] - point_start[0])
            reg_y.append(k * (x - point_start[0]) + point_start[1])
    reg_y = choose_min_reg(reg_y, y)

    for i in range(len(line) - 1):
        point_start, point_end = line[i], line[i + 1]
        if in_line_range(y, point_start[1], point_end[1]):
            k = (point_end[0] - point_start[0]) / (
                point_end[1] - point_start[1])
            reg_x.append(k * (y - point_start[1]) + point_start[0])
    reg_x = choose_min_reg(reg_x, x)
    return reg_x, reg_y

def extend_line(line, dis=10):
    extended = copy.deepcopy(line)
    start = line[1]
    end = line[0]
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    norm = math.sqrt(dx**2 + dy**2)
    dx = dx / norm
    dy = dy / norm
    extend_point = (start[0] + dx * dis, start[1] + dy * dis)
    extended.insert(0, extend_point)
    return extended

def select_mask_points(ct, r, shape, max_sample=5):

    def in_range(pt, w, h):
        if pt[0] >= 0 and pt[0] < w and pt[1] >= 0 and pt[1] < h:
            return True
        else:
            return False

    h, w = shape[:2]
    valid_points = []
    r = max(int(r // 2), 1)
    start_x, end_x = ct[0] - r, ct[0] + r
    start_y, end_y = ct[1] - r, ct[1] + r
    for x in range(start_x, end_x + 1):
        for y in range(start_y, end_y + 1):
            if x == ct[0] and y == ct[1]:
                continue
            if in_range((x, y), w, h) and cal_dis((x, y), ct) <= r + 0.1:
                valid_points.append([x, y])
    if len(valid_points) > max_sample - 1:
        valid_points = random.sample(valid_points, max_sample - 1)
    valid_points.append([ct[0], ct[1]])
    return valid_points

def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h

def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def draw_theta_headmap(heatmap, center, radius,theta, thre=0.5):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius -
                               left:radius + right]
    mask_idx = masked_gaussian.copy()
    masked_gaussian[mask_idx>=thre] = theta
    masked_gaussian[mask_idx<thre] = 0
    if min(masked_gaussian.shape) > 0 and min(
            masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian, out=masked_heatmap)

def convert_list(p, downscale=None):
    xy = list()
    if downscale is None:
        for i in range(len(p) // 2):
            xy.append((p[2 * i], p[2 * i + 1]))
    else:
        for i in range(len(p) // 2):
            xy.append((p[2 * i] / downscale, p[2 * i + 1] / downscale))
    return xy

def draw_label(mask,
               polygon_in,
               val,
               shape_type='polygon',
               width=3,
               convert=False):
    polygon = copy.deepcopy(polygon_in)
    mask = PIL.Image.fromarray(mask)
    xy = []
    if convert:
        for i in range(len(polygon) // 2):
            xy.append((polygon[2 * i], polygon[2 * i + 1]))
    else:
        for i in range(len(polygon)):
            xy.append((polygon[i][0], polygon[i][1]))

    if shape_type == 'polygon':
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=val, fill=val)
    else:
        PIL.ImageDraw.Draw(mask).line(xy=xy, fill=val, width=width)
    mask = np.array(mask, dtype=np.uint8)
    return mask

def clamp_line(line, box, min_length=0):
    left, top, right, bottom = box
    loss_box = Polygon([[left, top], [right, top], [right, bottom],
                        [left, bottom]])
    line_coords = np.array(line).reshape((-1, 2))
    if line_coords.shape[0] < 2:
        return None
    try:
        line_string = LineString(line_coords)
        I = line_string.intersection(loss_box)
        if I.is_empty:
            return None
        if I.length < min_length:
            return None
        if isinstance(I, LineString):
            pts = list(I.coords)
            return pts
        elif isinstance(I, MultiLineString):
            """
            MultiLineString has not been implemented
            """
            pts = []
            Istrings = list(I)
            for Istring in Istrings:
                pts += list(Istring.coords)
            return pts
    except:
        return None

def cal_dis(p1, p2):
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def dis(vec_1,vec_2):
    return ((vec_1[:,0]-vec_2[:,0]) ** 2 + (vec_1[:,1]-vec_2[:,1]) ** 2).reshape(vec_1.shape[0],1)
def cal_cross_point(point_theta):
    # point_theta (y,x,theta) in cv2 x&y 0~1 theta 0~180
    # points (y,x) in Cartesian coordinates then convert to cv2 by 1-*
    x0 = point_theta[:,1]
    y0 = point_theta[:,0]
    y0 = 1- y0
    n = point_theta.shape[0]
    k = np.tan(point_theta[:,2]/180 * math.pi)
    zeros = np.zeros((n,1))
    ones = np.ones((n,1))
    # point 1, x=0, y=? left
    y_out1 = k * 0 + y0 - k*x0
    points_1 = np.hstack((1-y_out1.reshape(n,1),zeros))
    # point 2, x=img_w, y=? right
    y_out2 = k * 1 + y0 - k*x0
    points_2 = np.hstack((1-y_out2.reshape(n,1),ones))
    # point 3, x= ? y=0 bottom
    x_out1 = (0 + k*x0-y0) / k
    points_3 = np.hstack((1-zeros,x_out1.reshape(n,1)))
    points = np.vstack((points_1[np.newaxis,...],points_2[np.newaxis,...],points_3[np.newaxis,...]))
    dis_all = np.hstack((dis(point_theta,points_1),dis(point_theta,points_2),dis(point_theta,points_3)))
    idx = np.argmin(dis_all,axis=1)
    row_idx = np.linspace(0,n-1,n).astype(int)
    point_theta[...,:2] = points[idx,row_idx]
    return point_theta

@TRANSFORM.register()
class CollectLane(object):
    def __init__(
            self,
            down_scale,
            keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg'),
            hm_down_scale=None,
            line_width=3,
            max_mask_sample=5,
            perspective=False,
            radius=2,
            cfg=None,
    ):
        self.meta_keys = meta_keys
        self.keys = keys
        self.cfg = cfg
        self.down_scale = down_scale
        self.hm_down_scale = hm_down_scale if hm_down_scale is not None else down_scale
        self.line_width = line_width
        self.max_mask_sample = max_mask_sample
        self.radius = radius

    def target(self, results):
        def min_dis_one_point(points, idx):
            min_dis = 1e6
            for i in range(len(points)):
                if i == idx:
                    continue
                else:
                    d = cal_dis(points[idx], points[i])
                    if d < min_dis:
                        min_dis = d
            return min_dis

        output_h = self.cfg.img_height
        output_w = self.cfg.img_width
        mask_h = int(output_h // self.down_scale)
        mask_w = int(output_w // self.down_scale)
        hm_h = int(output_h // self.hm_down_scale)
        hm_w = int(output_w // self.hm_down_scale)
        results['hm_shape'] = [hm_h, hm_w]
        results['mask_shape'] = [mask_h, mask_w]
        ratio_hm_mask = self.down_scale / self.hm_down_scale

        # gt init
        gt_hm = np.zeros((1, hm_h, hm_w), np.float32)
        gt_masks = []

        # gt heatmap and ins of bank
        gt_points = results['gt_points']
        # gt_points = results['lanes']
        valid_gt = []
        for pts in gt_points:
            id_class = 1
            pts = convert_list(pts, self.down_scale)
            pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))
            pts = clamp_line(
                pts, box=[0, 0, mask_w - 1, mask_h - 1], min_length=1)
            if pts is not None and len(pts) > 1:
                valid_gt.append([pts, id_class - 1])

        # draw gt_hm_lane
        gt_hm_lane_ends = []
        for l in valid_gt:
            label = l[1]
            point = (l[0][0][0] * ratio_hm_mask, l[0][0][1] * ratio_hm_mask)
            gt_hm_lane_ends.append([point, l[0]])
        radius = [self.radius for _ in range(len(gt_hm_lane_ends))]


        for (end_point, line), r in zip(gt_hm_lane_ends, radius):
            pos = np.zeros((mask_h), np.float32)
            pos_mask = np.zeros((mask_h), np.float32)
            pt_int = [int(end_point[0]), int(end_point[1])]
            draw_umich_gaussian(gt_hm[0], pt_int, r)
            line_array = np.array(line)
            y_min, y_max = int(np.min(line_array[:, 1])), int(
                np.max(line_array[:, 1]))
            mask_points = select_mask_points(
                pt_int, r, (hm_h, hm_w), max_sample=self.max_mask_sample)
            reg = np.zeros((1, mask_h, mask_w), np.float32)
            reg_mask = np.zeros((1, mask_h, mask_w), np.float32)

            extended_line = extend_line(line)
            line_array = np.array(line)
            y_min, y_max = np.min(line_array[:, 1]), np.max(line_array[:, 1])
            # regression
            m = np.zeros((mask_h, mask_w), np.uint8)
            lane_range = np.zeros((1, mask_h), np.int64)
            line_array = np.array(line)

            polygon = np.array(extended_line)
            polygon_map = draw_label(
                m, polygon, 1, 'line', width=self.line_width + 9) > 0
            for y in range(polygon_map.shape[0]):
                for x in np.where(polygon_map[y, :])[0]:
                    reg_x, _ = get_line_intersection(x, y, line)
                    # kps and kps_mask:
                    if reg_x is not None:
                        offset = reg_x - x
                        reg[0, y, x] = offset
                        if abs(offset) < 10:
                            reg_mask[0, y, x] = 1
                        if y >= y_min and y <= y_max:
                            pos[y] = reg_x
                            pos_mask[y] = 1
                        lane_range[:, y] = 1


            gt_masks.append({
                'reg': reg,
                'reg_mask': reg_mask,
                'points': mask_points,
                'row': pos,
                'row_mask': pos_mask,
                'range': lane_range,
                'label': 0
            })

        results['gt_hm'] = gt_hm
        results['gt_masks'] = gt_masks
        results['down_scale'] = self.down_scale
        results['hm_down_scale'] = self.hm_down_scale
        return True

    def __call__(self, results):
        data = {}
        img_meta = {}

        valid = self.target(results)
        if not valid:
            return None
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta,cpu_only = True)
        for key in self.keys:
            data[key] = results[key]
        return data

@TRANSFORM.register()
class CollectHm(object):
    def __init__(
            self,
            down_scale,
            keys,
            meta_keys=('filename', 'ori_shape', 'img_shape', 'img_norm_cfg'),
            hm_down_scale=None,
            line_width=3,
            max_mask_sample=5,
            perspective=False,
            radius=2,
            theta_thr=None,
            cfg=None,
    ):
        self.meta_keys = meta_keys
        self.keys = keys
        self.cfg = cfg
        self.theta_thr = theta_thr
        self.down_scale = down_scale
        self.hm_down_scale = hm_down_scale if hm_down_scale is not None else down_scale
        self.line_width = line_width
        self.max_mask_sample = max_mask_sample
        self.radius = radius

    def target(self, results):
        def min_dis_one_point(points, idx):
            min_dis = 1e6
            for i in range(len(points)):
                if i == idx:
                    continue
                else:
                    d = cal_dis(points[idx], points[i])
                    if d < min_dis:
                        min_dis = d
            return min_dis

        output_h = self.cfg.img_h
        output_w = self.cfg.img_w
        mask_h = int(output_h // self.down_scale)
        mask_w = int(output_w // self.down_scale)
        hm_h = int(output_h // self.hm_down_scale)
        hm_w = int(output_w // self.hm_down_scale)
        results['hm_shape'] = [hm_h, hm_w]
        results['mask_shape'] = [mask_h, mask_w]
        ratio_hm_mask = self.down_scale / self.hm_down_scale

        # gt init
        gt_hm = np.zeros((1, hm_h, hm_w), np.float32)
        shape_hm = np.zeros((1, hm_h, hm_w), np.float32)
        shape_hm_mask = np.zeros((1, hm_h, hm_w), np.float32)
        # gt heatmap and ins of bank
        gt_points = results['gt_points']
        thetas = results['lane_line'][:,4]
        lengths = results['lane_line'][:,5]
        valid_lane_mask = thetas>0
        # gt_points = results['lanes']
        valid_gt = []
        for lane_id,pts in enumerate(gt_points):
            id_class = 1
            pts = convert_list(pts, self.down_scale)
            pts = sorted(pts, key=cmp_to_key(lambda a, b: b[-1] - a[-1]))
            pts = clamp_line(pts, box=[0, 0, mask_w - 1, mask_h - 1], min_length=1)

            if pts is not None and len(pts) >= 1:
                valid_gt.append([pts, id_class - 1])

        gt_hm_lane_ends = []
        if self.cfg.haskey('do_mask'):
            do_mask = self.cfg.do_mask
        else:
            do_mask = False
        if do_mask:
        #================================================
        #                edge_start_points
        #================================================
            thetas = results['lane_line'][:,4][valid_lane_mask]
            lengths = results['lane_line'][:,5][valid_lane_mask]
            edge_start_pts = results['lane_line'][:,2:4][valid_lane_mask]
            edge_start_pts = cal_cross_point(np.hstack((edge_start_pts,thetas.reshape(-1,1))))
            for lane_id,start_pts in enumerate(edge_start_pts):
                # 0～99
                point = (start_pts[1]*mask_w-1 if start_pts[1] == 1 else start_pts[1]*mask_w,start_pts[0]*mask_h-1)
                theta = thetas[lane_id]
                length = lengths[lane_id]
                gt_hm_lane_ends.append([point,theta,length,start_pts[...,[1,0]]*np.array([self.cfg.img_w,self.cfg.img_h])])        
        else:
            # draw gt_hm_lane
            true_pts = []
            id_range = min(len(valid_gt),len(thetas))
            for lane_id in range(id_range):
                l = valid_gt[lane_id]
                point = (l[0][0][0] * ratio_hm_mask, l[0][0][1] * ratio_hm_mask)
                try:
                    theta = thetas[lane_id]
                except:
                    print(lane_id,len(thetas))
                length = lengths[lane_id]
                # if self.cfg.dataset_type == 'VIL':
                org_x,org_y = l[0][0][0]*self.down_scale,l[0][0][1]*self.down_scale
                true_pts.append([org_x,org_y])
                results['lane_line'][lane_id,2] = org_y / self.cfg.img_h
                results['lane_line'][lane_id,3] = org_x / self.cfg.img_w
                gt_hm_lane_ends.append([point,theta,length,[org_x,org_y]])
        radius = [self.radius for _ in range(len(gt_hm_lane_ends))]
        
        # if self.cfg.dataset_type == 'VIL' and len(results['lane_line'][valid_lane_mask]) > 0:
        #     new_lane = relocate2mid(proposals=results['lane_line'][valid_lane_mask],sps=np.array(true_pts),cfg=self.cfg)
        #     results['lane_line'][valid_lane_mask] = new_lane

        for (end_point,theta,length,pt), r in zip(gt_hm_lane_ends, radius):
            pt_int = [int(end_point[0]), int(end_point[1])]
            draw_umich_gaussian(gt_hm[0], pt_int, r)
            draw_theta_headmap(shape_hm[0],pt_int,r,theta=theta,thre=self.theta_thr)
        shape_hm_mask[shape_hm[0].reshape(shape_hm_mask.shape)>0]=1
        results['gt_hm'] = gt_hm
        results['shape_hm'] = shape_hm
        results['shape_hm_mask'] = shape_hm_mask
        return True

    def __call__(self, results):
        data = {}
        img_meta = {}

        valid = self.target(results)
        if not valid:
            return None
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)
        for key in self.keys:
            data[key] = results[key]
        return data