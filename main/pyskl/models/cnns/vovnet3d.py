# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import warnings
from mmcv.cnn import ConvModule, build_activation_layer, constant_init, kaiming_init
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.utils import _BatchNorm
from torch.nn.modules.utils import _ntuple, _triple

from ...utils import cache_checkpoint, get_root_logger
from ..builder import BACKBONES
