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

from thop import profile


args = default_argument_parser().parse_args()
print("Command Line Args:", args)

cfg = setup(args)
cfg.defrost()
cfg.MODEL.BACKBONE.PRETRAIN = False
# cfg.OUTPUT_DIR = cfg.PRUNE.SAVE_DIR # prune阶段的log都输出在PRUNE.SAVE_DIR目录下
# checkpoint = torch.load('./logs/veri/sbs_R50-ibn/pruned/pruned_0.3.pth')
# cfg.PRUNE.CFG_CHANNEL = checkpoint['cfg']
model = DefaultTrainer.build_model(cfg)

input=torch.randn(1,3,256,256)
flops, params = profile(model, inputs=(input, ))
print(flops)


# # image size(3,256,256)
# dataloader = DefaultTrainer.build_train_loader(cfg)
# data = iter(dataloader)
# data = next(data)
# data = data['images']
# print(data.size())