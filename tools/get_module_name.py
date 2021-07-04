#!/usr/bin/env python
# encoding: utf-8
import sys
sys.path.append('.')
from fastreid.config import get_cfg
from fastreid.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from fastreid.utils.checkpoint import Checkpointer
from train_net import setup


import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

args = default_argument_parser().parse_args()
print("Command Line Args:", args)

cfg = setup(args)
cfg.defrost()
cfg.MODEL.BACKBONE.PRETRAIN = False
##
checkpoint_cfg = torch.load('./logs/vehicleid/sbs_Dense40/pruned/pruned_0.3.pth')
cfg.PRUNE.CFG_CHANNEL = checkpoint_cfg['cfg']
cfg.OUTPUT_DIR = cfg.PRUNE.SAVE_DIR # prune阶段的log都输出在PRUNE.SAVE_DIR目录下

model = DefaultTrainer.build_model(cfg)

mylog = open('module_name.log', mode = 'w',encoding='utf-8')
for i in model.named_modules():
    print(i, file=mylog)
    print("==================================================", file=mylog)
mylog.close()