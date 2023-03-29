import numpy as np
import os
import paddle
import os.path as osp
import cv2
import glob
import argparse
from copy import copy
from pplanedet.datasets.preprocess.compose import Compose
from pplanedet.model.builder import build_model
from pplanedet.utils.py_config import Config
from pplanedet.datasets.visualization import imshow_lanes
from pathlib import Path
from tqdm import tqdm
from paddleseg.core.infer import reverse_transform
from paddleseg import utils
from paddleseg.utils import logger, progbar, visualize
import paddle.nn.functional as F


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

def mkdir(path):
    sub_dir = os.path.dirname(path)
    if not os.path.exists(sub_dir):
        os.makedirs(sub_dir)

def visualizes(im, result, color_map, save_dir=None, weight=0.6):
    """
    Convert predict result to color image, and save added image.

    Args:
        image (str): The path of origin image.
        result (np.ndarray): The predict result of image.
        color_map (list): The color used to save the prediction results.
        save_dir (str): The directory for saving visual image. Default: None.
        weight (float): The image weight of visual image, and the result weight is (1 - weight). Default: 0.6

    Returns:
        vis_result (np.ndarray): If `save_dir` is None, return the visualized result.
    """

    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    color_map = np.array(color_map).astype("uint8")
    # Use OpenCV LUT for color mapping
    c1 = cv2.LUT(result, color_map[:, 0])
    c2 = cv2.LUT(result, color_map[:, 1])
    c3 = cv2.LUT(result, color_map[:, 2])
    pseudo_img = np.dstack((c3, c2, c1))

#    im = cv2.imread(image)
    vis_result = cv2.addWeighted(im, weight, pseudo_img, 1 - weight, 0)

    if save_dir is not None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image_name = os.path.split(image)[-1]
        out_path = os.path.join(save_dir, image_name)
        cv2.imwrite(out_path, vis_result)
    else:
        return vis_result

class Detect(object):
    def __init__(self, cfg):
        self.cfg = cfg
        if cfg.get('val_transform',False):
            self.processes = Compose(cfg.val_transform, cfg)
        else:
            self.processes = Compose(cfg.val_process, cfg)
        self.net = build_model(self.cfg)
        self.net.eval() 
        load(self.net,cfg.load_from)

    def preprocess(self, img_path):
        ori_img = cv2.imread(img_path)
        img = ori_img[self.cfg.cut_height:, :, :].astype(np.float32)
        vis_img = copy(img)
        data = {'img': img, 'lanes': [],}
        data = self.processes(data)
        data['img'] = data['img'].unsqueeze(0)
        data.update({'img_path':img_path, 'ori_img':ori_img,'vis_img': vis_img})
        return data

    def inference(self, data):
        with paddle.no_grad():
            seg = self.net(data)
#            data = self.net.get_lanes(seg)
        return data,seg

    def show(self, data):
        out_file = self.cfg.savedir 
        if out_file:
            out_file = osp.join(out_file, osp.basename(data['img_path']))
        lanes = [lane.to_array(self.cfg) for lane in data['lanes']]
        imshow_lanes(data['ori_img'], lanes, show=self.cfg.show, out_file=out_file)

    def show_seg(self,logit,data,image_dir,im_path,custom_color = None):

        added_saved_dir = os.path.join(self.cfg.savedir , 'added_prediction')
        pred_saved_dir = os.path.join(self.cfg.savedir , 'pseudo_color_prediction')

        # get the saved name
        if image_dir is not None:
            im_file = im_path.replace(image_dir, '')
        else:
            im_file = os.path.basename(im_path)
        if im_file[0] == '/' or im_file[0] == '\\':
            im_file = im_file[1:]
            
        h = self.cfg.ori_img_h
        w = self.cfg.ori_img_w
#        print(data)
        color_map = visualize.get_color_map_list(256, custom_color=custom_color)
#        logit = F.interpolate(logit, (h, w), mode='bilinear')
        logit = F.softmax(logit,axis = 1)
        conf = paddle.max(logit, axis=1, keepdim=True)
#        logit = reverse_transform(logit, data['trans_info'], mode='bilinear')
        pred = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
        pred = paddle.squeeze(pred)
        pred = pred.cpu().numpy().astype('uint8')

        H,W = pred.shape

        im = cv2.imread(im_path)
        im = cv2.resize(data['vis_img'].astype('uint8'),(W,H))

        added_image = visualizes(
            im, pred, color_map, weight=0.6)
        added_image_path = os.path.join(added_saved_dir, im_file)
        mkdir(added_image_path)
        cv2.imwrite(added_image_path, added_image)

        # save pseudo color prediction
        pred_mask = utils.visualize.get_pseudo_color_map(pred, color_map)
        pred_saved_path = os.path.join(
            pred_saved_dir, os.path.splitext(im_file)[0] + ".png")
        mkdir(pred_saved_path)
        pred_mask.save(pred_saved_path)

    def run(self, data,image_dir):
        img_path = copy(data)
        data = self.preprocess(data)
        out = self.inference(data)
#        data['lanes'] = out[0][0]
#        data['lanes'],seg = self.inference(data)[0],self.inference(data)[1]
        if self.cfg.show or self.cfg.savedir:
            if not self.cfg.seg:
                data['lanes'] = self.net.get_lanes(out[1])[0]
                self.show(data)
            else:
                seg = out[1]['seg']
                self.show_seg(seg,data,image_dir,img_path,)
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
        detect.run(p,args.img)

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
