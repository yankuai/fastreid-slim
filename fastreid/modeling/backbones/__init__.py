# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from .build import build_backbone, BACKBONE_REGISTRY

from .resnet_bn import build_resnetbn_backbone
from .densenet import build_densenet_backbone
from .densenet import channel_selection
from .resnetx2 import build_resnetx2_backbone
from .resnetx import build_resnetx_backbone
from .resnet5 import build_resnetdeep5_backbone
from .resnet4 import build_resnetdeep4_backbone
from .resnet3 import build_resnetdeep3_backbone
from .resnet2 import build_resnetdeep2_backbone
from .resnetdeep import build_resnetdeep_backbone
from .resnet import build_resnet_backbone
from .osnet import build_osnet_backbone
from .resnest import build_resnest_backbone
from .resnext import build_resnext_backbone
from .regnet import build_regnet_backbone, build_effnet_backbone
from .shufflenet import build_shufflenetv2_backbone
