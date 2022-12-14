from .hook import Hook
from .builder import HOOKS


@HOOKS.register()
class LRSchedulerHook(Hook):
    def __init__(self, unit='iter', priority=1):
        self.priority = priority
        assert unit in ['iter', 'epoch']
        self.unit = unit

    def train_iter_end(self, trainer):
        if self.unit == 'iter':
            trainer.lr_scheduler.step()

    def train_epoch_end(self, trainer):
        if self.unit == 'epoch':
            trainer.lr_scheduler.step()