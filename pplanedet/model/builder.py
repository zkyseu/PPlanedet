from ..utils.registry import Registry, build_from_config
import paddle.nn as nn

BACKBONES = Registry('backbones')
AGGREGATORS = Registry('aggregators')
HEADS = Registry('heads')
NECKS = Registry('necks')
MODELS = Registry('models')
LOSSES = Registry('losses')

def build_backbones(cfg):
    return build_from_config(cfg.backbone, BACKBONES, default_args=dict(cfg=cfg))

def build_aggregator(cfg):
    return build_from_config(cfg.aggregator, AGGREGATORS, default_args=dict(cfg=cfg))

def build_heads(cfg):
    return build_from_config(cfg.heads, HEADS, default_args=dict(cfg=cfg))

def build_head(split_cfg, cfg):
    return build_from_config(split_cfg, HEADS, default_args=dict(cfg=cfg))

def build_model(cfg):
    return build_from_config(cfg.model, MODELS, default_args=dict(cfg=cfg))

def build_necks(cfg):
    return build_from_config(cfg.neck, NECKS, default_args=dict(cfg=cfg))

def build_losses(cfg):
    return build_from_config(cfg.loss,LOSSES,default_args=dict(cfg=cfg))

def build_loss(split_cfg,cfg):
    return build_from_config(split_cfg,LOSSES,default_args=dict(cfg=cfg))
