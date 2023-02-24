import os
import math
import random
import logging
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.distributed.fleet.meta_parallel import get_rng_state_tracker
from paddleseg.utils import logger

from ..hooks import build_hook, Hook
from ..utils import ModelEMA
from ..utils.misc import AverageMeter
from ..datasets.builder import build_dataloader
from ..model import build_model
from ..solver import build_lr_scheduler, build_optimizer
from ..datasets import IterLoader

def set_hyrbid_parallel_seed(basic_seed,
                             dp_rank,
                             mp_rank,
                             pp_rank,
                             device="cuda"):
    if not basic_seed:
        return
    assert device != "cpu"
    # local_seed/ global_seed is used to control dropout in ModelParallel
    local_seed = basic_seed + 123 + mp_rank * 10 + pp_rank * 1000
    global_seed = basic_seed + dp_rank
    tracker = get_rng_state_tracker()
    tracker.add('global_seed', global_seed)
    tracker.add('local_seed', local_seed)

class BaseTrainer:
    """
    Trainer class should contain following functions
    """
    def __init__(self,):
        pass

    def add_train_hooks(self,):
        pass

    def add_custom_hooks(self,):
        pass

    def add_hook(self,):
        pass

    def call_hook(self,):
        pass

    def train(self,):
        pass

    def val(self,):
        pass

    def resume(self,):
        pass

    def load(self,):
        pass

class Trainer(BaseTrainer):
    r"""
    # trainer calling logic:
    #
    #                build_model                               ||    model(BaseModel)
    #                     |                                    ||
    #               build_dataloader                           ||    dataloader
    #                     |                                    ||
    #               build_lr_scheduler                         ||    lr_scheduler
    #                     |                                    ||
    #               build_optimizer                            ||    optimizers
    #                     |                                    ||
    #               build_train_hooks                          ||    train hooks
    #                     |                                    ||
    #               build_custom_hooks                         ||    custom hooks
    #                     |                                    ||
    #                 train loop                               ||    train loop
    #                     |                                    ||
    #      hook(print log, checkpoint, evaluate, ajust lr)     ||    call hook
    #                     |                                    ||
    #                    end                                   \/

    """

    def __init__(self, cfg):
        #self.logger = logger
        cfg.num_gpus = dist.get_world_size()
        self.logger = logging.getLogger(__name__)
        self.cfg = cfg
        self.output_dir = cfg.output_dir
        self.best_dir = cfg.best_dir

        dp_rank = dist.get_rank()
        self.log_interval = cfg.log_config.interval

        # set seed
        seed = cfg.get('seed', False)
        if seed:
            seed += dp_rank
            paddle.seed(seed)
            np.random.seed(seed)
            random.seed(seed)

        # set device
        assert cfg['device'] in ['cpu', 'gpu', 'xpu', 'npu']
        self.device = paddle.set_device(cfg['device'])
        self.logger.info('train with paddle {} on {} device'.format(
            paddle.__version__, self.device))

        self.start_epoch = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.inner_iter = 0
        self.batch_id = 0
        self.global_steps = 0
        self.metric = 0
        self.best_metric = 0
        self.best_epoch = 0

        self.epochs = cfg.get('epochs', None)
        self.timestamp = cfg.timestamp
        self.logs = OrderedDict()

        self.model = build_model(cfg)
        self.logger.info(self.model)
    
        n_parameters = sum(p.numel() for p in self.model.parameters()
                           if not p.stop_gradient).item()

        i = int(math.log(n_parameters, 10) // 3)
        size_unit = ['', 'K', 'M', 'B', 'T', 'Q']
        self.logger.info("Number of Parameters is {:.2f}{}.".format(
            n_parameters / math.pow(1000, i), size_unit[i]))

        # build train dataloader
        self.train_dataloader = build_dataloader(
            self.cfg.dataset.train, self.cfg, is_train=True, device=self.device)
        self.iters_per_epoch = len(self.train_dataloader)

        # build learning rate
        self.lr_scheduler = build_lr_scheduler(cfg.lr_scheduler,
                                                self.iters_per_epoch)
        
        # build optimizer
        self.optimizer = build_optimizer(cfg.optimizer, self.lr_scheduler,
                                         [self.model])

        # distributed settings
        if dist.get_world_size() > 1:
            strategy = fleet.DistributedStrategy()
            ## Hybrid Parallel Training
            strategy.hybrid_configs = cfg.pop(
                'hybrid') if 'hybrid' in cfg else {}
            fleet.init(is_collective=True, strategy=strategy)
            hcg = fleet.get_hybrid_communicate_group()
            mp_rank = hcg.get_model_parallel_rank()
            pp_rank = hcg.get_stage_id()
            dp_rank = hcg.get_data_parallel_rank()
            set_hyrbid_parallel_seed(
                seed, 0, mp_rank, pp_rank, device=self.device)


        # amp training
        self.use_amp = cfg.get('use_amp',
                               False)  #if 'use_amp' in cfg else False

        if self.use_amp:
            amp_cfg = cfg.pop('AMP')
            self.auto_cast = amp_cfg.pop('auto_cast')
            scale_loss = amp_cfg.pop('scale_loss')
            self.scaler = paddle.amp.GradScaler(init_loss_scaling=scale_loss)
            amp_cfg['models'] = self.model
            if amp_cfg['level'] == 'O2':
                self.model = paddle.amp.decorate(**amp_cfg)  # decorate for level O2

        #whether use ema model
        self.use_ema = cfg.get('use_ema',False)
        if self.use_ema:
            self.logger.info("EMA model is adopted")
            ema_cfg = cfg.pop("ema")
            ema_decay = ema_cfg.get('ema_decay', 0.9998)
            ema_decay_type = ema_cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = ema_cfg.get('cycle_epoch', -1)
            ema_black_list = ema_cfg.get('ema_black_list', None)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list)


        # ZeRO
        self.sharding_strategies = cfg.get('sharding', False)
        if self.sharding_strategies:
            from paddle.distributed.fleet.meta_parallel.sharding.sharding_utils import ShardingScaler
            from paddle.distributed.fleet.meta_parallel.sharding.sharding_stage2 import ShardingStage2
            from paddle.distributed.fleet.meta_optimizers.dygraph_optimizer.sharding_optimizer_stage2 import ShardingOptimizerStage2
            self.sharding_stage = self.sharding_strategies['sharding_stage']
            accumulate_grad = self.sharding_strategies['accumulate_grad']
            offload = self.sharding_strategies['offload']
            if self.sharding_stage == 2:
                self.optimizer = ShardingOptimizerStage2(
                    params=self.model.parameters(),
                    optim=self.optimizer,
                    offload=offload)
                self.model = ShardingStage2(
                    self.model,
                    self.optimizer,
                    accumulate_grads=accumulate_grad)
                self.scaler = ShardingScaler(self.scaler)
            else:
                raise NotImplementedError()
        # data parallel
        elif dist.get_world_size() > 1:
            self.model = fleet.distributed_model(self.model)

        # build hooks
        self.hooks = []

        self.add_train_hooks()
        self.add_custom_hooks()
        self.hooks = sorted(self.hooks, key=lambda x: x.priority)

        if self.epochs:
            self.total_iters = self.epochs * self.iters_per_epoch
            self.by_epoch = True
        else:
            self.by_epoch = False
            self.total_iters = cfg.total_iters

    def add_train_hooks(self):
        optim_cfg = self.cfg.get('optimizer_config', None)
        if optim_cfg is not None:
            self.add_hook(build_hook(optim_cfg))
        else:
            self.add_hook(build_hook({'name': 'OptimizerHook'}))

        timer_cfg = self.cfg.get('timer_config', None)
        if timer_cfg is not None:
            self.add_hook(build_hook(timer_cfg))
        else:
            self.add_hook(build_hook({'name': 'IterTimerHook'}))
        ckpt_cfg = self.cfg.get('checkpoint', None)
        if ckpt_cfg is not None:
            self.add_hook(build_hook(ckpt_cfg))
        else:
            self.add_hook(build_hook({'name': 'CheckpointHook'}))

        log_cfg = self.cfg.get('log_config', None)
        if log_cfg is not None:
            self.add_hook(build_hook(log_cfg))
        else:
            self.add_hook(build_hook({'name': 'LogHook'}))

        lr_cfg = self.cfg.get('lr_config', None)
        if lr_cfg is not None:
            self.add_hook(build_hook(lr_cfg))
        else:
            self.add_hook(build_hook({'name': 'LRSchedulerHook'}))


    def add_custom_hooks(self):
        custom_cfgs = self.cfg.get('custom_config', None)
        if custom_cfgs is None:
            return

        for custom_cfg in custom_cfgs:
            cfg_ = custom_cfg.copy()
            insert_index = cfg_.pop('insert_index', None)
            self.add_hook(build_hook(cfg_), insert_index)

    def add_hook(self, hook, insert_index=None):
        assert isinstance(hook, Hook)

        if insert_index is None:
            self.hooks.append(hook)
        elif isinstance(insert_index, int):
            self.hooks.insert(insert_index, hook)

    def call_hook(self, fn_name):
        for hook in self.hooks:
            getattr(hook, fn_name)(self)

    def train(self):
        self.mode = 'train'
        self.model.train()
        iter_loader = IterLoader(self.train_dataloader, self.current_epoch)
        self.call_hook('run_begin')

        while self.current_iter < (self.total_iters):
            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_begin')
            self.inner_iter = self.current_iter % self.iters_per_epoch
            self.current_iter += 1
            self.current_epoch = iter_loader.epoch

            data = next(iter_loader)      

            self.call_hook('train_iter_begin')  

            if self.use_amp:
                with paddle.amp.auto_cast(**self.auto_cast):
                    self.outputs = self.model(data)

            else:
                self.outputs = self.model(data)
            
            self.call_hook('train_iter_end')
            if self.use_ema:
                self.ema.update()

            if self.current_iter % self.iters_per_epoch == 0:
                self.call_hook('train_epoch_end')
                self.current_epoch += 1

        self.call_hook('run_end')            


    def val(self, **kargs):
        if not hasattr(self, 'val_loader'):
            self.val_loader = build_dataloader(self.cfg.dataset.val, self.cfg, is_train=False,device=self.device,drop_last = False)        

        self.logger.info('start evaluate on epoch {} ..'.format(
            self.current_epoch + 1))
        rank = dist.get_rank()
        world_size = dist.get_world_size()    

        total_samples = len(self.val_loader.dataset)
        self.logger.info('Evaluate total samples {}'.format(total_samples))
        
        self.model.eval()
        predictions = []

        for i, data in enumerate(tqdm(self.val_loader, desc=f'Validate')):
            with paddle.no_grad():
                output = self.model(data,mode = 'test')
                if world_size > 1:
                    seg_list = []
                    if 'seg' in output.keys():
                        seg = output['seg']
                        dist.all_gather(seg_list, seg)
                        seg = paddle.concat(seg_list, 0)
                        output['seg'] = seg   
                    else:
                        seg = output['cls']
                        dist.all_gather(seg_list, seg)
                        seg = paddle.concat(seg_list, 0)
                        output['cls'] = seg                      
                    if 'exist' in output:
                        exists = output['exist']
                        exist_list = []
                        dist.all_gather(exist_list, exists)
                        exists = paddle.concat(exist_list, 0)
                        output['exist'] = exists
                    output = self.model._layers.get_lanes(output)  
                else:
                    output = self.model.get_lanes(output)             
                predictions.extend(output)
            if self.cfg.view:
                self.val_loader.dataset.view(output, data['meta'])

        out = self.val_loader.dataset.evaluate(predictions, self.cfg.pred_save_dir) 
 

        if out > self.best_metric:
            self.best_metric = out
            self.best_epoch = self.current_epoch + 1
        self.logger.info(f"best accuracy is {self.best_metric}") 
        self.logger.info(f"The epoch of best accuracy is {self.best_epoch}")

        self.model.train()

        self.metric = out

    def resume(self, checkpoint_path):
        checkpoint = paddle.load(checkpoint_path)
        ema_model = checkpoint['ema_model']
        if checkpoint.get('epoch', None) is not None:
            self.start_epoch = checkpoint['epoch']
            self.current_epoch = checkpoint['epoch']
            self.current_iter = (self.start_epoch - 1) * self.iters_per_epoch

        self.model.set_state_dict(checkpoint['state_dict'])
        self.optimizer.set_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.set_state_dict(checkpoint['lr_scheduler'])
        #resume ema training
        if self.use_ema and ema_model is not None:
            self.ema.resume(ema_model,self.start_epoch-1)            

        self.logger.info('Resume training from {} success!'.format(
            checkpoint_path))

    def load(self, weight_path, export=False):
        state_dict = paddle.load(weight_path)

        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        if export:
            state_dict_ = dict()
            for k, v in state_dict.items():
                state_dict_['model.' + k] = v
            state_dict = state_dict_
        self.model.set_state_dict(state_dict)   
