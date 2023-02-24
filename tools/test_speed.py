import time
import argparse
import math

import paddle
from pplanedet.utils.py_config import Config
from pplanedet.model.builder import build_model
from pplanedet.datasets.preprocess.compose import Compose

paddle.device.cuda.synchronize()

def parse_args():
    parser = argparse.ArgumentParser(description="Tool to measure a model's speed")
    parser.add_argument('config', help='The path of config file')
    parser.add_argument("--model_path", help="Model checkpoint path (optional)")
    parser.add_argument('--iters', default=100, type=int, help="Number of times to run the model and get the average")

    return parser.parse_args()


if __name__ == "__main__":
    """
    This file is modified from https://github.com/lucastabelini/LaneATT/blob/main/utils/speed.py
    """
    args = parse_args()
    cfg = Config.fromfile(args.config)
    model = build_model(cfg)
    height, width = cfg.img_height,cfg.img_width

    n_parameters = sum(p.numel() for p in model.parameters()
                        if not p.stop_gradient).item()
    i = int(math.log(n_parameters, 10) // 3)
    size_unit = ['', 'K', 'M', 'B', 'T', 'Q']
    test_param = n_parameters / math.pow(1000, i)
    print("Number of Parameters is {:.2f}{}.".format(test_param, size_unit[i]))

    if args.model_path is not None:
        ckpt = paddle.load(args.model_path)
        if 'state_dict' in ckpt.keys():
            model.set_state_dict(ckpt['state_dict'])
        else:
            model.set_state_dict(ckpt)
    
    model.eval()

    x = paddle.zeros((1, 3, height, width)) + 1

    # Benchmark latency and FPS
    t_all = 0
    for _ in range(args.iters):
        t1 = time.time()
        model(dict(img = x))
        t2 = time.time()
        t_all += t2 - t1

    print('Average latency (ms): {:.2f}'.format(t_all * 1000 / args.iters))
    print('Average FPS: {:.2f}'.format(args.iters / t_all))
