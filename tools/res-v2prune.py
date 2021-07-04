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
from fastreid.modeling.backbones import channel_selection


args = default_argument_parser().parse_args()
print("Command Line Args:", args)

cfg = setup(args)
cfg.defrost()
cfg.MODEL.BACKBONE.PRETRAIN = False
cfg.PRUNE.CFG_CHANNEL = []
cfg.OUTPUT_DIR = cfg.PRUNE.SAVE_DIR # prune阶段的log都输出在PRUNE.SAVE_DIR目录下
model = DefaultTrainer.build_model(cfg)

if not os.path.exists(cfg.PRUNE.SAVE_DIR):
    os.makedirs(cfg.PRUNE.SAVE_DIR)

print("=> loading checkpoint '{}'".format(cfg.MODEL.WEIGHTS))
Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
res = []
#res.append(DefaultTrainer.test(cfg, model))

total = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for m in model.modules():
    if isinstance(m, nn.BatchNorm2d):
        size = m.weight.data.shape[0]
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * cfg.PRUNE.PERCENT)
thre = y[thre_index]

pruned = 0
cfg_channel = []
cfg_mask = []
first_bn = True
for k, m in enumerate(model.modules()):
    if isinstance(m, nn.BatchNorm2d):
        if first_bn == True:
            first_bn = False
            continue
        if isinstance(m, BatchNorm):
            continue
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg_channel.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
            format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total
print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
#res.append(DefaultTrainer.test(cfg, model))

print("Cfg:")
print(cfg_channel)

cfg.defrost()
cfg.PRUNE.CFG_CHANNEL = cfg_channel
# cfg.PRUNE.NEW_FEAT_DIM =True
newmodel = DefaultTrainer.build_model(cfg)

old_num_parameters = sum([param.nelement() for param in model.parameters()])
new_num_parameters = sum([param.nelement() for param in newmodel.parameters()])

savepath = os.path.join(cfg.PRUNE.SAVE_DIR, "prune.txt")
fp_prune=open(savepath, "a")
fp_prune.write("\n\nPrune Ratio: "+str(pruned_ratio)+"\n")
fp_prune.write("Prune Configuration(channel numbers in each BN): \n"+str(cfg_channel)+"\n")
fp_prune.write("Number of parameters: \n"+"\tOriginal: "+str(old_num_parameters)+"\n"+"\tAfter Prune: "+str(new_num_parameters)+"\n")

old_modules = list(model.modules())
new_modules = list(newmodel.modules())

layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]
first_bn = True
conv_count = 0

m0parameter = os.path.join(cfg.PRUNE.SAVE_DIR, "m0.txt")
m1parameter = os.path.join(cfg.PRUNE.SAVE_DIR, "m1.txt")
fp0 = open(m0parameter, "w")
fp1 = open(m1parameter, "w")

for layer_id in range(len(old_modules)):
    m0 = old_modules[layer_id]
    m1 = new_modules[layer_id]
    if isinstance(m0, nn.BatchNorm2d) and not(isinstance(m0, BatchNorm)):
        print('into nn.batchnorm2d')
        idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
        if idx1.size == 1:
            idx1 = np.resize(idx1,(1,))
        
        if first_bn:
            # We don't change the first bn layer.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            first_bn = False
            continue
        if isinstance(old_modules[layer_id + 1], channel_selection):
            print('bn is before channel selection')
            # If the next layer is the channel selection layer, then the current batch normalization layer won't be pruned.
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()
            # We need to set the mask parameter `indexes` for the channel selection layer.
            m2 = new_modules[layer_id + 1]
            m2.indexes.data.zero_()
            m2.indexes.data[idx1.tolist()] = 1.0
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):
                end_mask = cfg_mask[layer_id_in_cfg]
            continue
        else:
            print('bn is not before channel selection')
            m1.weight.data = m0.weight.data[idx1.tolist()].clone()
            m1.bias.data = m0.bias.data[idx1.tolist()].clone()
            m1.running_mean = m0.running_mean[idx1.tolist()].clone()
            m1.running_var = m0.running_var[idx1.tolist()].clone()
            layer_id_in_cfg += 1
            start_mask = end_mask.clone()
            if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                end_mask = cfg_mask[layer_id_in_cfg]

    elif isinstance(m0, nn.Conv2d):
        print('into nn.conv2d')
        if conv_count==0:
            # We don't change the first convolution layer.
            m1.weight.data = m0.weight.data.clone()
            conv_count += 1
            continue
        
        if isinstance(old_modules[layer_id-1], channel_selection) or isinstance(old_modules[layer_id-1], nn.BatchNorm2d):
            # This convers the convolutions in the residual block.
            # The convolutions are either after the channel selection layer or after the batch normalization layer.
            conv_count += 1
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))

            # If the current convolution is not the last convolution in the residual block, then we can change the 
            # number of output channels. Currently we use `conv_count` to detect whether it is such convolution.
            w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            if conv_count % 3 != 1:
                w1 = w1[idx1.tolist(), :, :, :].clone()
            m1.weight.data = w1.clone()
            continue
        
        # We need to consider the case where there are downsampling convolutions. 
        # For these convolutions, we just copy the weights.
        m1.weight.data = m0.weight.data.clone()

    # elif isinstance(m0, BatchNorm): # last bn in head
    #     print('into batchnorm')
    #     idx1 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    #     if idx1.size == 1:
    #         idx1 = np.resize(idx1,(1,))
    #     m1.weight.data = m0.weight.data[idx1.tolist()].clone()
    #     m1.bias.data = m0.bias.data[idx1.tolist()].clone()
    #     m1.running_mean = m0.running_mean[idx1.tolist()].clone()
    #     m1.running_var = m0.running_var[idx1.tolist()].clone()

    # elif isinstance(m0, CircleSoftmax):
    #     print('into circlesoftmax')
    #     idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
    #     if idx0.size == 1:
    #         idx0 = np.resize(idx0, (1,))

    #     m1.weight.data = m0.weight.data[:, idx0].clone()
    #     m1.bias.data = m0.bias.data.clone()  # ???

torch.save({'cfg': cfg_channel, 'state_dict': newmodel.state_dict()}, os.path.join(cfg.PRUNE.SAVE_DIR, 'pruned_{}.pth'.format(cfg.PRUNE.PERCENT)))

model = newmodel
res.append(DefaultTrainer.test(cfg, model))
fp_prune.write("Test result: \n"+"\tOriginal Model: "+str(res[0])+"\n\tSet BN channel to 0: "+str(res[1])+"\n\tPruned Model: "+str(res[2])+"\n")