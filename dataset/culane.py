import os
import os.path as osp
import numpy as np
import cv2
from tqdm import tqdm
import logging
from .base_dataset import BaseDataset
from .culane_metric import eval_predictions

LIST_FILE = {
    'train': 'list/train_gt.txt',
    'val': 'list/test.txt',
    'test': 'list/test.txt',
} 

class CULane(BaseDataset):
    def __init__(self, 
                data_root, 
                split, 
                cut_height,
                sample_y,
                ori_w_h,
                processes=None):
        super().__init__(data_root, split, cut_height,transform=processes)
        self.list_path = osp.join(data_root, LIST_FILE[split])
        self.load_annotations()
        self.sample_y = sample_y
        self.ori_img_h,self.ori_img_w = ori_w_h

    def load_annotations(self):
        self.logger.info('Loading CULane annotations...')
        self.data_infos = []
        with open(self.list_path) as list_file:
            for line in list_file:
                infos = self.load_annotation(line.split())
                self.data_infos.append(infos)

    def load_annotation(self, line):
        infos = {}
        img_line = line[0]
        img_line = img_line[1 if img_line[0] == '/' else 0::]
        img_path = os.path.join(self.data_root, img_line)
        infos['img_name'] = img_line 
        infos['img_path'] = img_path
        if len(line) > 1:
            mask_line = line[1]
            mask_line = mask_line[1 if mask_line[0] == '/' else 0::]
            mask_path = os.path.join(self.data_root, mask_line)
            infos['mask_path'] = mask_path

        if len(line) > 2:
            exist_list = [int(l) for l in line[2:]]
            infos['lane_exist'] = np.array(exist_list)

        anno_path = img_path[:-3] + 'lines.txt'  # remove sufix jpg and add lines.txt
        with open(anno_path, 'r') as anno_file:
            data = [list(map(float, line.split())) for line in anno_file.readlines()]
        lanes = [[(lane[i], lane[i + 1]) for i in range(0, len(lane), 2) if lane[i] >= 0 and lane[i + 1] >= 0]
                 for lane in data]
        lanes = [list(set(lane)) for lane in lanes]  # remove duplicated points
        lanes = [lane for lane in lanes if len(lane) > 3]  # remove lanes with less than 2 points

        lanes = [sorted(lane, key=lambda x: x[1]) for lane in lanes]  # sort by y
        infos['lanes'] = lanes

        return infos

    def get_prediction_string(self, pred):
        ys = np.array(list(self.sample_y))[::-1] / self.ori_img_h
        out = []
        for lane in pred:
            xs = lane(ys)
            valid_mask = (xs >= 0) & (xs < 1)
            xs = xs * self.ori_img_w
            lane_xs = xs[valid_mask]
            lane_ys = ys[valid_mask] * self.ori_img_h
            lane_xs, lane_ys = lane_xs[::-1], lane_ys[::-1]
            lane_str = ' '.join(['{:.5f} {:.5f}'.format(x, y) for x, y in zip(lane_xs, lane_ys)])
            if lane_str != '':
                out.append(lane_str)

        return '\n'.join(out)

    def evaluate(self, predictions, output_basedir):
        print('Generating prediction output...')
        for idx, pred in enumerate(tqdm(predictions)):
            output_dir = os.path.join(output_basedir, os.path.dirname(self.data_infos[idx]['img_name']))
            output_filename = os.path.basename(self.data_infos[idx]['img_name'])[:-3] + 'lines.txt'
            os.makedirs(output_dir, exist_ok=True)
            output = self.get_prediction_string(pred)
            with open(os.path.join(output_dir, output_filename), 'w') as out_file:
                out_file.write(output)
        result = eval_predictions(output_basedir, self.data_root, self.list_path, official=True)
        self.logger.info(result)
        return result['F1']