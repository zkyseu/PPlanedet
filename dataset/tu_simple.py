import os.path as osp
import numpy as np
import cv2
import os
import json
import random
from .base_dataset import BaseDataset
from .tu_simple_metric import LaneEval
from paddleseg.cvlibs import manager

SPLIT_FILES = {
    'trainval': ['label_data_0313.json', 'label_data_0601.json', 'label_data_0531.json'],
    'train': ['label_data_0313.json', 'label_data_0601.json'],
    'val': ['label_data_0531.json'],
    'test': ['test_label.json'],
}

@manager.DATASETS.add_component
class TuSimple(BaseDataset):
    def __init__(self, 
                data_root, 
                split, 
                cut_height,
                ori_w_h,
                num_classes,
                test_json_file = None,
                transforms=None):
        super().__init__(data_root, split, cut_height,transforms)
        self.anno_files = SPLIT_FILES[split] 
        self.load_annotations()
        self.h_samples = list(range(160, 720, 10))
        self.ori_img_h,self.ori_img_w = ori_w_h
        self.exist_lane_num = np.zeros((num_classes -1))
        self.test_json_file = test_json_file

    def load_annotations(self):
        self.logger.info('Loading TuSimple annotations...')
        self.data_infos = []
        max_lanes = 0
        for anno_file in self.anno_files:
            anno_file = osp.join(self.data_root, anno_file)
            with open(anno_file, 'r') as anno_obj:
                lines = anno_obj.readlines()
            for line in lines:
                data = json.loads(line)
                y_samples = data['h_samples']
                gt_lanes = data['lanes']
                mask_path = data['raw_file'].replace('clips', 'seg_label')[:-3] + 'png'
                lanes = [[(x, y) for (x, y) in zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
                lanes = [lane for lane in lanes if len(lane) > 0]
                max_lanes = max(max_lanes, len(lanes))
                self.data_infos.append({
                    'img_path': osp.join(self.data_root, data['raw_file']),
                    'img_name': data['raw_file'],
                    'mask_path': osp.join(self.data_root, mask_path),
                    'lanes': lanes,
                })

        if self.training:
            random.shuffle(self.data_infos)
        self.max_lanes = max_lanes

    def __getitem__(self, idx):
        data_info = self.data_infos[idx]
        if not osp.isfile(data_info['img_path']):
            raise FileNotFoundError('cannot find file: {}'.format(data_info['img_path']))

        img = cv2.imread(data_info['img_path'])

        img = img[self.cut_height:, :, :]
        sample = data_info.copy()
        sample.update({'img': img})
        sample.update({'gt_fields':[]})

        if self.training:
            label = cv2.imread(sample['mask_path'], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cut_height:, :]
            lane_exist_id = np.unique(label)
            for idxs in lane_exist_id:
                if idxs == 0:
                    continue
                lane_idxs = idxs -1 
                self.exist_lane_num[lane_idxs] = 1
            sample.update({'label': label})
            sample['gt_fields'].append('label')
        
        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}

        data = {}
        data.update({"img":sample["img"]})
        if self.training:
            data.update({'label':sample["label"]})
            data.update({'exist_lane_num':self.exist_lane_num})
        # print(sample)
        # meta = DC(meta, cpu_only=True)
        # sample.update({'meta': meta})


        return data 

    def pred2lanes(self, pred):
        ys = np.array(self.h_samples) / self.ori_img_h
        lanes = []
        for lane in pred:
            xs = lane(ys)
            invalid_mask = xs < 0
            lane = (xs * self.ori_img_w).astype(int)
            lane[invalid_mask] = -2
            lanes.append(lane.tolist())

        return lanes

    def pred2tusimpleformat(self, idx, pred, runtime):
        runtime *= 1000.  # s to ms
        img_name = self.data_infos[idx]['img_name']
        lanes = self.pred2lanes(pred)
        output = {'raw_file': img_name, 'lanes': lanes, 'run_time': runtime}
        return json.dumps(output)

    def save_tusimple_predictions(self, predictions, filename, runtimes=None):
        if runtimes is None:
            runtimes = np.ones(len(predictions)) * 1.e-3
        lines = []
        for idx, (prediction, runtime) in enumerate(zip(predictions, runtimes)):
            line = self.pred2tusimpleformat(idx, prediction, runtime)
            lines.append(line)
        with open(filename, 'w') as output_file:
            output_file.write('\n'.join(lines))

    def evaluate(self, predictions, output_basedir, runtimes=None):
        if not os.path.exists(output_basedir):
            os.mkdir(output_basedir)
        pred_filename = os.path.join(output_basedir, 'tusimple_predictions.json')
        self.save_tusimple_predictions(predictions, pred_filename, runtimes)
        result, acc = LaneEval.bench_one_submit(pred_filename, self.test_json_file)
        self.logger.info(result)
        return acc