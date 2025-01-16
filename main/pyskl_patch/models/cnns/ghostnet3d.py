import math
import warnings
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _ntuple, _triple

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible
from _efficientnet_blocks import SqueezeExcite, ConvBnAct # timm

from mmcv.runner import _load_checkpoint, load_checkpoint # mmcv
from mmcv.cnn import constant_init, kaiming_init # mmcv

from ...utils import cache_checkpoint, get_root_logger # pyskl
from ..builder import BACKBONES # pyskl


__all__ = ['GhostNet3D']

_SE_LAYER = partial(
    SqueezeExcite,
    gate_layer='hard_sigmoid',
    rd_round_fn=partial(make_divisible, divisor=4)
)

class GhostModule3D(nn.Module):
    """
    3D version of GhostModule, replacing nn.Conv2d with nn.Conv3d
    """
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=1,
            ratio=2,
            dw_size=3,
            stride=1,
            use_act=True,
            act_layer=nn.ReLU,
    ):
        super().__init__()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm3d(init_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_chs, new_chs, dw_size, 1, dw_size // 2, groups=init_chs, bias=False),
            nn.BatchNorm3d(new_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        # out 形狀: (N, C, D, H, W)
        return out[:, :self.out_chs, ...]  # 截取需要的 channel 數量


class GhostModule3DV2(nn.Module):
    """
    3D version of GhostModuleV2
    Compare to GhostModuleV1, GhostModuleV2 has short_conv and attention (sigmoid).
    """
    def __init__(
            self,
            in_chs,
            out_chs,
            kernel_size=1,
            ratio=2,
            dw_size=3,
            stride=1,
            use_act=True,
            act_layer=nn.ReLU,
    ):
        super().__init__()
        self.gate_fn = nn.Sigmoid()
        self.out_chs = out_chs
        init_chs = math.ceil(out_chs / ratio)
        new_chs = init_chs * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv3d(in_chs, init_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm3d(init_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )
        self.cheap_operation = nn.Sequential(
            nn.Conv3d(init_chs, new_chs, dw_size, 1, dw_size // 2, groups=init_chs, bias=False),
            nn.BatchNorm3d(new_chs),
            act_layer(inplace=True) if use_act else nn.Identity(),
        )

        # short_conv 這裡簡單示範：用 stride=2 的 3D pool 後再經過一系列 conv3d
        self.short_conv = nn.Sequential(
            nn.Conv3d(in_chs, out_chs, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm3d(out_chs),
            nn.Conv3d(out_chs, out_chs, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1),
                      groups=out_chs, bias=False),
            nn.BatchNorm3d(out_chs),
            nn.Conv3d(out_chs, out_chs, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0),
                      groups=out_chs, bias=False),
            nn.BatchNorm3d(out_chs),
        )

    def forward(self, x):
        # downsample 使用 3D average pool，kernel_size=2, stride=2
        # 視需求調整 kernel_size
        res = self.short_conv(F.avg_pool3d(x, kernel_size=2, stride=2))

        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)  # (N, C, D, H, W)

        # 上採樣回原本深度、空間維度
        # 也可以改用 trilinear，但 nearest 在某些硬體上可能更快
        out_upsample = F.interpolate(self.gate_fn(res), size=out.shape[-3:], mode='nearest')
        return out[:, :self.out_chs, ...] * out_upsample


class GhostBottleneck3D(nn.Module):
    """
    Ghost Bottleneck 的 3D 版本。包含：
    1. Ghost module (V1 or V2)
    2. Depthwise 3D conv
    3. SE (Squeeze-and-Excitation)
    4. Shortcut
    """
    def __init__(
            self,
            in_chs,
            mid_chs,
            out_chs,
            dw_kernel_size=3,
            stride=1,
            act_layer=nn.ReLU,
            se_ratio=0.,
            mode='original',
    ):
        super().__init__()
        has_se = (se_ratio is not None) and (se_ratio > 0.)
        self.stride = stride

        # 1) Ghost module (Point-wise expansion)
        if mode == 'original':
            self.ghost1 = GhostModule3D(in_chs, mid_chs, use_act=True, act_layer=act_layer)
        else:
            self.ghost1 = GhostModule3DV2(in_chs, mid_chs, use_act=True, act_layer=act_layer)

        # 2) Depthwise 3D conv
        if self.stride > 1:
            self.conv_dw = nn.Conv3d(
                mid_chs, mid_chs, dw_kernel_size, stride=stride,
                padding=(dw_kernel_size - 1) // 2, groups=mid_chs, bias=False
            )
            self.bn_dw = nn.BatchNorm3d(mid_chs)
        else:
            self.conv_dw = None
            self.bn_dw = None

        # 3) Squeeze-and-excitation
        self.se = _SE_LAYER(mid_chs, rd_ratio=se_ratio) if has_se else None

        # 4) Ghost module (point-wise linear projection)
        self.ghost2 = GhostModule3D(mid_chs, out_chs, use_act=False)

        # shortcut
        if in_chs == out_chs and self.stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv3d(
                    in_chs, in_chs, dw_kernel_size, stride=stride,
                    padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False
                ),
                nn.BatchNorm3d(in_chs),
                nn.Conv3d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm3d(out_chs),
            )

    def forward(self, x):
        shortcut = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # depthwise conv
        if self.conv_dw is not None:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # SE
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        # shortcut
        x += self.shortcut(shortcut)
        return x

# #------------------------------------------------------------------------
# # 這裡就是實際要用 mmcv 的方式去做 backbone 註冊的 GhostNet3D
# #------------------------------------------------------------------------
@BACKBONES.register_module()
class GhostNet3D(nn.Module):
    """3D GhostNet backbone"""

    def __init__(self,
                 in_channels=17,
                 base_channels=32,
                 num_stages=4,
                 up_strides=(1, 2, 2, 2),
                 pretrained=None,
                 pool_strat='avg',
                 act_layer=[nn.ReLU, nn.ReLU],
                 **kwargs):
        super().__init__()
        
        from functools import reduce
        from operator import mul

        # Define basic parameters
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.up_strides = up_strides
        self.pool_strat = pool_strat
        self.pool_layer = nn.AvgPool3d if pool_strat == 'avg' else nn.MaxPool3d

        # Define the GhostNet3D backbone
        self.model = nn.Sequential()

        for i in range(num_stages):
            if i == 0:
                self.model.add_module(
                    f'layer{i}',
                    GhostModule3DV2(in_channels, base_channels, act_layer=act_layer[i % len(act_layer)])
                )
                self.model.add_module(
                    f'pool{i}',
                    self.pool_layer(kernel_size=(1, 2, 2), stride=(1, 2, 2))
                )
            else:
                self.model.add_module(
                    f'layer{i}',
                    GhostModule3DV2(base_channels * reduce(mul, self.up_strides[:i]), base_channels * reduce(mul, self.up_strides[:i+1]), act_layer=act_layer[i % len(act_layer)])
                )


    def init_weights(self):
        """Initiate the parameters either from existing checkpoint or from
        scratch."""
        pass

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data. The size of x is (num_batches, 17, 32, 56, 56).

        Returns:
            torch.Tensor: The feature of the input samples extracted by the backbone.
        """