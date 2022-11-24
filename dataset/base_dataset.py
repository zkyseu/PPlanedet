import os.path as osp
import os
import numpy as np
import cv2
from paddle.io import Dataset
from paddleseg.utils import logger
from paddleseg.transforms import Compose
from .data_container import DataContainer as DC
from .visualization import imshow_lanes


class BaseDataset(Dataset):
    def __init__(self, data_root, split, cut_height,transform=None,ignore_index = 255):
        self.logger = logger
        self.data_root = data_root
        self.training = 'train' in split 
        self.processes = Compose(transform)
        self.cut_height = cut_height
        self.ignore_index = ignore_index


    def view(self, predictions, img_metas):
        img_metas = [item for img_meta in img_metas.data for item in img_meta]
        for lanes, img_meta in zip(predictions, img_metas):
            img_name = img_meta['img_name']
            img = cv2.imread(osp.join(self.data_root, img_name))
            out_file = osp.join(self.cfg.work_dir, 'visualization',
                                img_name.replace('/', '_'))
            lanes = [lane.to_array(self.cfg) for lane in lanes]
            imshow_lanes(img, lanes, out_file=out_file)

    def __len__(self):
        return len(self.data_infos)

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
            sample.update({'label': label})
            sample['gt_fields'].append('label')
        
        sample = self.processes(sample)
        meta = {'full_img_path': data_info['img_path'],
                'img_name': data_info['img_name']}

        data = {}
        data.update({"img":sample["img"]})
        if self.training:
            data.update({'label':sample["label"]})
            if 'lane_exist' in sample.keys():
                data.update({'exist_lane_num':sample['lane_exist']})

        # print(sample)
        # meta = DC(meta, cpu_only=True)
        # sample.update({'meta': meta})


        return data 
