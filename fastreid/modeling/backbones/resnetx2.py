# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import logging
import math

import torch
from torch import nn
import numpy as np

from fastreid.layers import (
    IBN,
    SELayer,
    Non_local,
    get_norm,
)
from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm


logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'ibn_18x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet18_ibn_a-2f571257.pth',
    'ibn_34x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet34_ibn_a-94bc1577.pth',
    'ibn_50x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet50_ibn_a-d9d0bb7b.pth',
    'ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnet101_ibn_a-59ea0ac6.pth',
    'se_ibn_101x': 'https://github.com/XingangPan/IBN-Net/releases/download/v1.0/se_resnet101_ibn_a-fabed4e2.pth',
}

class channel_selection(nn.Module):
    """
    Select channels from the output of BatchNorm2d layer. It should be put directly after BatchNorm2d layer.
    The output shape of this layer is determined by the number of 1 in `self.indexes`.
    """
    def __init__(self, num_channels):
        """
        Initialize the `indexes` with all one vector with the length same as the number of channels.
        During pruning, the places in `indexes` which correpond to the channels to be pruned will be set to 0.
        """
        super(channel_selection, self).__init__()
        self.indexes = nn.Parameter(torch.ones(num_channels))

    def forward(self, input_tensor):
        """
        Parameter
        ---------
        input_tensor: (N,C,H,W). It should be the output of BatchNorm2d layer.
        """
        selected_index = np.squeeze(np.argwhere(self.indexes.data.cpu().numpy()))
        if selected_index.size == 1:
            selected_index = np.resize(selected_index, (1,)) 
        output = input_tensor[:, selected_index, :, :]
        return output

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, cfg, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.select = channel_selection(inplanes)
        self.conv1 = nn.Conv2d(cfg[0], cfg[1], kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(cfg[1])
        self.conv2 = nn.Conv2d(cfg[1], cfg[2], kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(cfg[2])
        self.conv3 = nn.Conv2d(cfg[2], planes * 4, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.count = 0
        self.inplanes = inplanes

    def forward(self, x):
        print('==========')
        residual = x
        self.count +=1
        # out = self.bn1(x)
        # out = self.select(out)
        # out = self.relu(out)
        # out = self.conv1(out)

        out = self.bn1(x)
        out = self.relu(out)
        print('A.out.shape',out.shape)
        if self.downsample is not None:
            residual = self.downsample(out)   
        print('B.out.shape',out.shape)
        out = self.select(out)
        out = self.conv1(out)
        print('C.out.shape',out.shape)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)
        print('D.out.shape',out.shape)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)
        print('conv3',self.conv3)
        print('residual.shape',residual.shape)
        print('E.out.shape',out.shape)
        out += residual

        return out

class ResNet(nn.Module):
    def __init__(self, last_stride, bn_norm, with_ibn, with_se, with_nl, block, layers, cfg_channel=None):
        
        if cfg_channel is None:
            # Construct config variable.
            cfg_channel = [[64, 64, 64], [256, 64, 64]*(layers[0]-1), [256, 128, 128], [512, 128, 128]*(layers[1]-1), [512, 256, 256], [1024, 256, 256]*(layers[2]-1), [1024, 512, 512], [2048, 512, 512]*(layers[3]-1)]
            cfg_channel = [item for sub_list in cfg_channel for item in sub_list]
        assert len(cfg_channel) == 3*(layers[0]+layers[1]+layers[2]+layers[3]), 'wrong with cfg_channel element number'

        self.inplanes = 64
        super().__init__()
        self.conv0 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn0 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        
        indx1 = 3*layers[0]
        indx2 = indx1 + 3*layers[1]
        indx3 = indx2 + 3*layers[2]
        indx4 = indx3 + 3*layers[3]
        self.layer1 = self._make_layer(block, 64, layers[0], cfg_channel[0:indx1], 1, bn_norm, with_ibn, with_se)
        self.layer2 = self._make_layer(block, 128, layers[1], cfg_channel[indx1:indx2], 2, bn_norm, with_ibn, with_se)
        self.layer3 = self._make_layer(block, 256, layers[2], cfg_channel[indx2:indx3], 2, bn_norm, with_ibn, with_se)
        self.layer4 = self._make_layer(block, 512, layers[3], cfg_channel[indx3:indx4], last_stride, bn_norm, with_se=with_se)

        # self.bn5 = nn.BatchNorm2d(2048) # get_norm(bn_norm, 2048)
        # self.select = channel_selection(2048)

        self.random_init()

    def _make_layer(self, block, planes, blocks, cfg_channel, stride=1, bn_norm="BN", with_ibn=False, with_se=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                # get_norm(bn_norm, planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg_channel[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg_channel[3*i:3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # layer 1
        for i in range(len(self.layer1)):
            x = self.layer1[i](x)
        # layer 2
        for i in range(len(self.layer2)):
            x = self.layer2[i](x)

        # layer 3
        for i in range(len(self.layer3)):
            x = self.layer3[i](x)

        # layer 4
        for i in range(len(self.layer4)):
            x = self.layer4[i](x)

        #x = self.bn5(x)
        #x = self.select(x)
        #x = self.relu(x)
        return x

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def init_pretrained_weights(key):
    """Initializes model with pretrained weights.

    Layers that don't match with pretrained layers in name or size are kept unchanged.
    """
    import os
    import errno
    import gdown

    def _get_torch_home():
        ENV_TORCH_HOME = 'TORCH_HOME'
        ENV_XDG_CACHE_HOME = 'XDG_CACHE_HOME'
        DEFAULT_CACHE_DIR = '~/.cache'
        torch_home = os.path.expanduser(
            os.getenv(
                ENV_TORCH_HOME,
                os.path.join(
                    os.getenv(ENV_XDG_CACHE_HOME, DEFAULT_CACHE_DIR), 'torch'
                )
            )
        )
        return torch_home

    torch_home = _get_torch_home()
    model_dir = os.path.join(torch_home, 'checkpoints')
    try:
        os.makedirs(model_dir)
    except OSError as e:
        if e.errno == errno.EEXIST:
            # Directory already exists, ignore.
            pass
        else:
            # Unexpected OSError, re-raise.
            raise

    filename = model_urls[key].split('/')[-1]

    cached_file = os.path.join(model_dir, filename)

    if not os.path.exists(cached_file):
        if comm.is_main_process():
            gdown.download(model_urls[key], cached_file, quiet=False)

    comm.synchronize()

    logger.info(f"Loading pretrained model from {cached_file}")
    state_dict = torch.load(cached_file, map_location=torch.device('cpu'))

    return state_dict


@BACKBONE_REGISTRY.register()
def build_resnetx2_backbone(cfg):
    """
    Create a ResNet instance from config.
    Returns:
        ResNet: a :class:`ResNet` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    last_stride   = cfg.MODEL.BACKBONE.LAST_STRIDE
    bn_norm       = cfg.MODEL.BACKBONE.NORM
    with_ibn      = cfg.MODEL.BACKBONE.WITH_IBN
    with_se       = cfg.MODEL.BACKBONE.WITH_SE
    with_nl       = cfg.MODEL.BACKBONE.WITH_NL
    depth         = cfg.MODEL.BACKBONE.DEPTH  # '164x'
    cfg_channel   = cfg.PRUNE.CFG_CHANNEL
    # fmt: on

    num_blocks_per_stage = {
        '18x': [2, 2, 2, 2],
        '34x': [3, 4, 6, 3],
        '50x': [3, 4, 6, 3],
        '101x': [3, 4, 23, 3],
    }[depth]

    block = Bottleneck

    depth = int(cfg.MODEL.BACKBONE.DEPTH[:-1])  # 164

    if len(cfg_channel) == 0:
        model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, )
    else:    
        model = ResNet(last_stride, bn_norm, with_ibn, with_se, with_nl, block,
                   num_blocks_per_stage, cfg_channel)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
            if with_ibn: key = 'ibn_' + key
            if with_se:  key = 'se_' + key

            state_dict = init_pretrained_weights(key)

        incompatible = model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(
                get_missing_parameters_message(incompatible.missing_keys)
            )
        if incompatible.unexpected_keys:
            logger.info(
                get_unexpected_parameters_message(incompatible.unexpected_keys)
            )

    return model