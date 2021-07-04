#!/usr/bin/env python
# encoding: utf-8
import os
import argparse
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from fastreid.layers import BatchNorm, CircleSoftmax, GeneralizedMeanPoolingP
from train_net import setup

args = default_argument_parser().parse_args()
print("Command Line Args:", args)

cfg = setup(args)
cfg.defrost()
cfg.MODEL.BACKBONE.PRETRAIN = False
cfg.PRUNE.CFG_CHANNEL = []
# cfg.OUTPUT_DIR = cfg.PRUNE.SAVE_DIR # prune阶段的log都输出在PRUNE.SAVE_DIR目录下
model = DefaultTrainer.build_model(cfg)


backbone_parameters = 0
agg_parameters = 0
bn_parameters = 0
for named_m in model.named_modules():
    m = named_m[1]
    if 'backbone' in named_m[0] and (isinstance(m, nn.Conv2d) or isinstance(m, BatchNorm) or isinstance(m, nn.MaxPool2d)):
        backbone_parameters += sum([param.nelement() for param in m.parameters()])
    elif 'heads' in named_m[0] and isinstance(m, GeneralizedMeanPoolingP):
        agg_parameters += sum([param.nelement() for param in m.parameters()])
    elif 'heads' in named_m[0] and isinstance(m, BatchNorm):
        bn_parameters += sum([param.nelement() for param in m.parameters()])

print('backbone_parameters:',backbone_parameters)
print('agg_parameters:',agg_parameters)
print('bn_parameters:',bn_parameters)