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

cslayer = CircleSoftmax(cfg, 1024, 10)
for param in cslayer.parameters():
    print(param)
print(cslayer.weight.data.shape)
