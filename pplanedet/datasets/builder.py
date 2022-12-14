import copy
import numpy as np
import math
import paddle
import random
from paddle.io import DistributedBatchSampler
from functools import partial

from ..utils.registry import Registry, build_from_config

DATASETS = Registry("DATASET")
TRANSFORM = Registry('transform')

def worker_init_fn(worker_id, seed):
    worker_seed = worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_dataset(split_cfg, cfg):
    return build_from_config(split_cfg, DATASETS, default_args=dict(cfg=cfg))

def build_dataloader(split_cfg, cfg, is_train=True,device = None,drop_last = True):

    if is_train:
        shuffle = True
    else:
        shuffle = False
    
    dataset = build_dataset(split_cfg, cfg)

    init_fn = partial(
            worker_init_fn, seed=cfg.seed)    

    sampler = paddle.io.DistributedBatchSampler(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle, drop_last=drop_last)

    dataloader = paddle.io.DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=cfg.num_workers,
        return_list=True,
        places=device,
        worker_init_fn=init_fn,)
    
    return dataloader