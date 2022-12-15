import numpy as np
import os
import paddle
import os.path as osp
import cv2
import glob
import argparse
from pplanedet.datasets.preprocess.compose import Compose
from pplanedet.model.builder import build_model
from pplanedet.utils.py_config import Config
from pplanedet.datasets.visualization import imshow_lanes
from pathlib import Path
from tqdm import tqdm

def load(model,weight_path, export=False):
    state_dict = paddle.load(weight_path)

    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']

    if export:
        state_dict_ = dict()
        for k, v in state_dict.items():
            state_dict_['model.' + k] = v
        state_dict = state_dict_
    model.set_state_dict(state_dict) 


class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.processes = Compose(cfg.val_process, cfg)
        self.net = build_model(self.cfg)
        self.net.eval() 
        load(self.net,cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        data = {'img': img, 'lanes': []}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img})
        return data

    def inference(self, data):
        with paddle.no_grad():
            data = self.net(data)
            data = self.net.get_lanes(data)
        return data

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)

    def run(self, data):
        data = self.preprocess(data)
        data['lanes'] = self.inference(data)[0]
        if self.cfg.show or self.cfg.savedir:
            self.show(data)
        return data

def get_img_paths(path):
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        paths = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        paths = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        paths = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')
    return paths 

def process(args):
    cfg = Config.fromfile(args.config)
    cfg.show = args.show
    cfg.savedir = args.savedir
    cfg.load_from = args.load_from
    detect = Detect(cfg)
    paths = get_img_paths(args.img)
    for p in tqdm(paths):
        detect.run(p)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='The path of config file')
    parser.add_argument('--img',  help='The path of the img (img file or img_folder), for example: data/*.png')
    parser.add_argument('--show', action='store_true', 
            help='Whether to show the image')
    parser.add_argument('--savedir', type=str, default=None, help='The root of save directory')
    parser.add_argument('--load_from', type=str, default='best.pth', help='The path of model')
    args = parser.parse_args()
    process(args)