# encoding: utf-8

import logging
import math

import torch
from torch import nn
import numpy as np

from fastreid.utils.checkpoint import get_missing_parameters_message, get_unexpected_parameters_message
from .build import BACKBONE_REGISTRY
from fastreid.utils import comm

logger = logging.getLogger(__name__)
model_urls = {
    '18x': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    '34x': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    '50x': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    '101x': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
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

# no pretrain
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

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.select(out)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out

class ResNetDeep(nn.Module):
    def __init__(self, depth=164, cfg=None):    
        super(ResNetDeep, self).__init__()
        assert (depth - 2) % 9 == 0, 'depth should be 9n+2'

        n = (depth - 2) // 9
        block = Bottleneck

        if cfg is None:
            # Construct config variable.
            # cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256]]  # original 101
            # cfg = [[64, 64, 64], [256, 64, 64]*(n-1), [256, 128, 128], [512, 128, 128]*(n-1), [512, 256, 256], [1024, 256, 256]*(n-1), [1024]] # original 50 with depth 101
            # (1)
            cfg = [[16, 16, 16], [64, 16, 16]*(n-1), [64, 32, 32], [128, 32, 32]*(n-1), [128, 64, 64], [256, 64, 64]*(n-1), [256,512,512], [2048,512,512]*2, [2048]] # front 101 back 50
            cfg = [item for sub_list in cfg for item in sub_list]

        # self.inplanes = 16

        # self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
        #                        bias=False)
        # self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
        # self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
        # self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
        # self.bn = nn.BatchNorm2d(64 * block.expansion)
        # self.select = channel_selection(64 * block.expansion)
        # self.relu = nn.ReLU(inplace=True)
        
        # (2)
        indx1 = 3*layers[0]
        indx2 = indx1 + 3*layers[1]
        indx3 = indx2 + 3*layers[2]
        indx4 = indx3 + 3*layers[3]

        self.inplanes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1,
                               bias=False)
        self.layer1 = self._make_layer(block, 16, n, cfg = cfg[0:3*n])
        self.layer2 = self._make_layer(block, 32, n, cfg = cfg[3*n:6*n], stride=2)
        self.layer3 = self._make_layer(block, 64, n, cfg = cfg[6*n:9*n], stride=2)
        self.layer4 = self._make_layer(block, 512, 3, cfg = cfg[9*n:9*n+9], stride=2)
        self.bn = nn.BatchNorm2d(512 * block.expansion)
        self.select = channel_selection(512 * block.expansion)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, cfg, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
            )

        layers = []
        layers.append(block(self.inplanes, planes, cfg[0:3], stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cfg[3*i: 3*(i+1)]))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        # (3)
        x = self.layer4(x) 
        x = self.bn(x)
        x = self.select(x)
        x = self.relu(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x

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
def build_resnetdeep_backbone(cfg):
    """
    Create a ResNetDeep instance from config.
    Returns:
        ResNetDeep: a :class:`ResNetDeep` instance.
    """

    # fmt: off
    pretrain      = cfg.MODEL.BACKBONE.PRETRAIN
    pretrain_path = cfg.MODEL.BACKBONE.PRETRAIN_PATH
    depth         = int(cfg.MODEL.BACKBONE.DEPTH[:-1])  # '164x'
    cfg_channel   = cfg.PRUNE.CFG_CHANNEL
    # fmt: on

    if len(cfg_channel) == 0:
        model = ResNetDeep(depth)
    else:    
        model = ResNetDeep(depth, cfg=cfg_channel)
    if pretrain:
        # Load pretrain path if specifically
        if pretrain_path:
            try:
                # state_dict = torch.load(pretrain_path, map_location=torch.device('cpu'))
                checkpoint = torch.load(pretrain_path, map_location=torch.device('cpu'))
                state_dict = checkpoint['state_dict']
                logger.info(f"Loading pretrained model from {pretrain_path}")
            except FileNotFoundError as e:
                logger.info(f'{pretrain_path} is not found! Please check this path.')
                raise e
            except KeyError as e:
                logger.info("State dict keys error! Please check the state dict.")
                raise e
        else:
            key = depth
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
