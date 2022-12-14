import paddle
from .hook import Hook
from .builder import HOOKS
from visualdl import LogWriter
import paddle.distributed as dist
import os


@HOOKS.register()
class VisualHook(Hook):
    def __init__(self, priority=1):
        self.priority = priority

    def run_begin(self, trainer):
        rank = dist.get_rank()
        if rank != 0:
            return
        logdir = os.path.join(trainer.output_dir, 'visual_dl')
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        self.writer = LogWriter(logdir=logdir)

    def train_epoch_end(self, trainer):
        rank = dist.get_rank()
        if rank != 0:
            return
        outputs = trainer.outputs
        for k in outputs.keys():
            v = trainer.logs[k].avg
            self.writer.add_scalar(tag='train/{}'.format(k),
                                   step=trainer.current_epoch,
                                   value=v)
        with paddle.no_grad():
            if dist.get_world_size() > 1:
                for name, param in trainer.model._layers.named_parameters():
                    if 'bn' not in name:
                        self.writer.add_histogram(name, param.numpy(),
                                                  trainer.current_epoch)
            else:
                for name, param in trainer.model.named_parameters():
                    if 'bn' not in name:
                        self.writer.add_histogram(name, param.numpy(),
                                                  trainer.current_epoch)

    def run_end(self, trainer):
        rank = dist.get_rank()
        if rank != 0:
            return
        self.writer.close()