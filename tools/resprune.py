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

if not os.path.exists(cfg.PRUNE.SAVE_DIR):
    os.makedirs(cfg.PRUNE.SAVE_DIR)

print("=> loading checkpoint '{}'".format(cfg.MODEL.WEIGHTS))
Checkpointer(model).load(cfg.MODEL.WEIGHTS)  # load trained model
res = []
res.append(DefaultTrainer.test(cfg, model))

total = 0

for named_m in model.named_modules():
    if 'bn' in named_m[0]:
        m = named_m[1]
        total += m.weight.data.shape[0]

bn = torch.zeros(total)
index = 0
for named_m in model.named_modules():
    if 'bn' in named_m[0]:
        m = named_m[1]
        size = m.weight.data.shape[0]
        print(m,m.weight.data.abs()) ###
        bn[index:(index+size)] = m.weight.data.abs().clone()
        index += size

y, i = torch.sort(bn)
thre_index = int(total * cfg.PRUNE.PERCENT)
thre = y[thre_index]


pruned = 0
cfg_channel = []
cfg_mask = []
for k, named_m in enumerate(model.named_modules()):
    if 'bn' in named_m[0]:
        m = named_m[1]
        weight_copy = m.weight.data.abs().clone()
        mask = weight_copy.gt(thre).float().cuda()
        pruned = pruned + mask.shape[0] - torch.sum(mask)
        m.weight.data.mul_(mask)
        m.bias.data.mul_(mask)
        cfg_channel.append(int(torch.sum(mask)))
        cfg_mask.append(mask.clone())
        # print('layer index: {:d} \t total channel: {:d} \t remaining channel: {:d}'.
        #    format(k, mask.shape[0], int(torch.sum(mask))))

pruned_ratio = pruned/total

print('Pre-processing Successful!')

# simple test model after Pre-processing prune (simple set BN scales to zeros)
res.append(DefaultTrainer.test(cfg, model))

print("Cfg:")
print(cfg_channel)

cfg.defrost()
cfg.PRUNE.CFG_CHANNEL = cfg_channel
newmodel = DefaultTrainer.build_model(cfg)

# old_num_parameters = sum([param.nelement() for param in model.parameters()])
# new_num_parameters = sum([param.nelement() for param in newmodel.parameters()])
old_num_parameters = 0
new_num_parameters = 0
for named_m in model.named_modules():
    if 'bn' in named_m[0] or 'conv' in named_m[0]:
        m = named_m[1]
        old_num_parameters += sum([param.nelement() for param in m.parameters()])

for named_m in newmodel.named_modules():
    if 'bn' in named_m[0] or 'conv' in named_m[0]:
        m = named_m[1]
        new_num_parameters += sum([param.nelement() for param in m.parameters()])


savepath = os.path.join(cfg.PRUNE.SAVE_DIR, "prune.txt")
fp_prune=open(savepath, "a")
fp_prune.write("\n\nPrune Ratio: "+str(pruned_ratio)+"\n")
fp_prune.write("Prune Configuration(channel numbers in each BN): \n"+str(cfg_channel)+"\n")
fp_prune.write("Number of parameters: \n"+"\tOriginal: "+str(old_num_parameters)+"\n"+"\tAfter Prune: "+str(new_num_parameters)+"\n")

named_old_modules = list(model.named_modules())
named_new_modules = list(newmodel.named_modules())
layer_id_in_cfg = 0
start_mask = torch.ones(3)
end_mask = cfg_mask[layer_id_in_cfg]


m0parameter = os.path.join(cfg.PRUNE.SAVE_DIR, "m0.txt")
m1parameter = os.path.join(cfg.PRUNE.SAVE_DIR, "m1.txt")
fp0 = open(m0parameter, "w")
fp1 = open(m1parameter, "w")
for layer_id in range(len(named_old_modules)):
    m0_name,m0 = named_old_modules[layer_id]
    m1_name,m1 = named_new_modules[layer_id]
    if isinstance(m0, BatchNorm):
        if 'bn' in m0_name: # 'bn0-3'
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            if idx1.size == 1:
                idx1 = np.resize(idx1,(1,))

            if 'bn1' in m0_name or 'bn2' in m0_name:
                m1.weight.data = m0.weight.data[idx1.tolist()].clone()
                m1.bias.data = m0.bias.data[idx1.tolist()].clone()
                m1.running_mean = m0.running_mean[idx1.tolist()].clone()
                m1.running_var = m0.running_var[idx1.tolist()].clone()
                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):  # do not change in Final FC
                    end_mask = cfg_mask[layer_id_in_cfg]
            else: # 'bn0' or 'bn3'
                m1.weight.data = m0.weight.data.clone()
                m1.bias.data = m0.bias.data.clone()
                m1.running_mean = m0.running_mean.clone()
                m1.running_var = m0.running_var.clone()

                layer_id_in_cfg += 1
                start_mask = end_mask.clone()
                if layer_id_in_cfg < len(cfg_mask):
                    end_mask = cfg_mask[layer_id_in_cfg]
        else: # other bn(in shortcut, non_local, heads)
            m1.weight.data = m0.weight.data.clone()
            m1.bias.data = m0.bias.data.clone()
            m1.running_mean = m0.running_mean.clone()
            m1.running_var = m0.running_var.clone()

        # fp0.write(m0_name+"::\n")
        # for i in m0.named_parameters():
        #     fp0.write(str(i)+"\n")
        # fp1.write(m1_name+"::\n")
        # for i in m1.named_parameters():
        #     fp1.write(str(i)+"\n")
    elif isinstance(m0, nn.Conv2d):
        if 'conv1' in m0_name or 'conv2' in m0_name or 'conv3' in m0_name:
            idx0 = np.squeeze(np.argwhere(np.asarray(start_mask.cpu().numpy())))
            idx1 = np.squeeze(np.argwhere(np.asarray(end_mask.cpu().numpy())))
            # print('In shape: {:d}, Out shape {:d}.'.format(idx0.size, idx1.size))
            if idx0.size == 1:
                idx0 = np.resize(idx0, (1,))
            if idx1.size == 1:
                idx1 = np.resize(idx1, (1,))
            
            if 'conv1' in m0_name: # [c_out:c_in]
                w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
            elif 'conv2' in m0_name:
                w1 = m0.weight.data[idx1.tolist(), :, :, :].clone()
                w1 = w1[:, idx0.tolist(), :, :].clone()
            else: # 'conv3'
                w1 = m0.weight.data[:, idx0.tolist(), :, :].clone()
            m1.weight.data = w1.clone()
        else: # 'conv0' or other conv model(in shortcut, non_local)
            m1.weight.data = m0.weight.data.clone()
        if 'NL' in m0_name and m0.bias is not None: # NL中的conv都有bias，其他位置的conv没有bias
            m1.bias.data = m0.bias.data.clone()

    elif isinstance(m0, CircleSoftmax):
    # fastreid的模型与最后的bn或者conv相连的是不是linear？？？
        m1.weight.data = m0.weight.data.clone()
        fp0.write(m0_name+"::\n")
        for i in m0.named_parameters():
            fp0.write(str(i)+"\n")
        fp1.write(m1_name+"::\n")
        for i in m1.named_parameters():
            fp1.write(str(i)+"\n")
    elif isinstance(m0, GeneralizedMeanPoolingP):
        m1.p.data = m0.p.data.clone()

torch.save({'cfg': cfg_channel, 'state_dict': newmodel.state_dict()}, os.path.join(cfg.PRUNE.SAVE_DIR, 'pruned_{}.pth'.format(cfg.PRUNE.PERCENT)))


model = newmodel
res.append(DefaultTrainer.test(cfg, model))
fp_prune.write("Test result: \n"+"\tOriginal Model: "+str(res[0])+"\n\tSet BN channel to 0: "+str(res[1])+"\n\tPruned Model: "+str(res[2])+"\n")