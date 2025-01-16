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
    """範例：使用 mmcv 風格的 3D GhostNet backbone。"""

    def __init__(self,
                 cfgs=None,
                 width=1.0,
                 in_channels=3,
                 out_indices=(3,),
                 num_stages=5,
                 output_stride=32,
                 drop_rate=0.2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 pretrained=None,
                 init_cfg=None,  # mmcv 也支援 init_cfg，可自行擴充
                 norm_eval=False,
                 frozen_stages=-1,
                 zero_init_residual=False,
                 **kwargs):
        """
        Args:
            cfgs (list): 與 2D GhostNet 相似，定義各 stage 的 kernel/expansion/se_ratio 等。
            width (float): 通道寬度縮放係數。
            in_channels (int): 輸入通道數。
            out_indices (tuple): 輸出哪幾個 stage 的 feature map。
            num_stages (int): 有幾個 stage。
            output_stride (int): 只示範固定 32，不支援 dilation。
            drop_rate (float): dropout 率。
            act_cfg (dict): activation config, mmcv 風格。
            pretrained (str): 若有預先訓練權重檔路徑，可指定。
            norm_eval (bool): 是否將 BN 層設成 eval (不更新 mean/var)。
            frozen_stages (int): 冷凍到第幾個 stage。-1 表示不凍結。
            zero_init_residual (bool): 是否把最後的 BN 做 zero init。
            kwargs: 其他額外引數。
        """
        super().__init__()
        # 針對 mmcv backbone 常見的參數設定
        assert output_stride == 32, 'Only support output_stride=32'
        self.cfgs = cfgs if cfgs is not None else self._get_default_cfgs()
        self.width = width
        self.in_channels = in_channels
        self.num_stages = num_stages
        self.out_indices = out_indices
        self.drop_rate = drop_rate
        self.norm_eval = norm_eval
        self.frozen_stages = frozen_stages
        self.zero_init_residual = zero_init_residual
        self.pretrained = pretrained

        # stem
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv3d(in_channels, stem_chs, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)  # 這裡簡單示範

        prev_chs = stem_chs
        # 逐 stage 建構 GhostBottleneck3D
        self.blocks = nn.ModuleList()
        self.stage_idx_list = []  # 用於對應 out_indices

        net_stride = 2  # 已經經過 stem (stride=2)
        total_stages = len(self.cfgs)  # 預設 5 個 stage
        stage_count = 0
        for i, cfg in enumerate(self.cfgs):
            layers = []
            s = 1  # 預設 stride
            for (k, exp_size, c, se_ratio, s) in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layer = GhostBottleneck3D(
                    in_chs=prev_chs,
                    mid_chs=mid_chs,
                    out_chs=out_chs,
                    dw_kernel_size=k,
                    stride=s,
                    se_ratio=se_ratio,
                    mode='original',
                )
                layers.append(layer)
                prev_chs = out_chs

            # 若該 stage 的最後一個 layer stride>1，表示空間 stride 翻倍
            if s > 1:
                net_stride *= 2
                stage_count += 1

            self.blocks.append(nn.Sequential(*layers))
            self.stage_idx_list.append(i)

        # 最後可能還有一層
        out_chs = make_divisible(exp_size * width, 4)
        self.conv_last = nn.Conv3d(prev_chs, out_chs, kernel_size=1, bias=False)
        self.bn_last = nn.BatchNorm3d(out_chs)
        self.act_last = nn.ReLU(inplace=True)

        # 統一記錄可供輸出的層 (主要是 blocks + 最後一層)
        self._all_blocks = [f'blocks.{i}' for i in range(len(self.blocks))]
        self._all_blocks.append('conv_last')

        # 預設只輸出最後一層 (即 stage 5)，若 out_indices 需要中間層，得對照 self._all_blocks[i] 的 idx
        # 這裡只做簡單示範
        # 例如 out_indices=(0,1,4) => 表示輸出 blocks.0, blocks.1, conv_last
        self.init_weights()

    def _get_default_cfgs(self):
        """給個預設配置 (對應原 2D GhostNet)"""
        cfgs = [
            # stage0 (對應 stride=1)
            [[3, 16, 16, 0, 1]],
            # stage1
            [[3, 48, 24, 0, 2], [3, 72, 24, 0, 1]],
            # stage2
            [[3, 72, 40, 0.25, 2], [3, 120, 40, 0.25, 1]],
            # stage3
            [[3, 240, 80, 0, 2],
             [3, 200, 80, 0, 1],
             [3, 184, 80, 0, 1],
             [3, 480, 112, 0.25, 1],
             [3, 672, 112, 0.25, 1]],
            # stage4
            [[3, 672, 160, 0.25, 2],
             [3, 960, 160, 0, 1],
             [3, 960, 160, 0.25, 1],
             [3, 960, 160, 0, 1],
             [3, 960, 160, 0.25, 1]]
        ]
        return cfgs

    def init_weights(self):
        """mmcv 常用的 init_weights，可載入 pretrained 權重或初始化權重。"""
        # 1) 先用 kaiming / constant init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

        if self.zero_init_residual:
            # 若想把最後一層 BN init to 0，視需求而定
            for m in self.modules():
                if isinstance(m, GhostBottleneck3D):
                    # 與 ResNet3D 的 zero_init_residual 類似邏輯，可自行定義
                    pass

        # 2) 若 self.pretrained 有指定，載入 checkpoint
        if isinstance(self.pretrained, str):
            logger = get_root_logger()
            logger.info(f'Load GhostNet3D pretrained weights from: {self.pretrained}')
            # 這裡單純 load_checkpoint，不做 inflate (因為已是 3D)
            ckpt = cache_checkpoint(self.pretrained)
            load_checkpoint(self, ckpt, strict=False, logger=logger)

    def _freeze_stages(self):
        """依據 frozen_stages，凍結指定層。"""
        # 假設 conv_stem + bn1 是 stage 0
        if self.frozen_stages >= 0:
            self.conv_stem.eval()
            self.bn1.eval()
            for param in self.conv_stem.parameters():
                param.requires_grad = False
            for param in self.bn1.parameters():
                param.requires_grad = False

        # 依序凍結 self.blocks, self.conv_last, ...
        # 這裡簡單假設 blocks.x -> stage x+1
        for i in range(1, self.frozen_stages + 1):
            if i - 1 < len(self.blocks):
                m = self.blocks[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

        # 若 frozen_stages >= num_stages (即凍結最後層)
        # 可再對 conv_last 做凍結
        if self.frozen_stages >= self.num_stages:
            self.conv_last.eval()
            self.bn_last.eval()
            for param in self.conv_last.parameters():
                param.requires_grad = False
            for param in self.bn_last.parameters():
                param.requires_grad = False

    def forward(self, x):
        """forward 過程: stem -> blocks -> conv_last -> return"""
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)

        outs = []
        # 依序通過每個 blocks stage
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.out_indices:
                outs.append(x)
        # 最後的 conv
        x = self.conv_last(x)
        x = self.bn_last(x)
        x = self.act_last(x)
        if (len(self.blocks)) in self.out_indices:  # 若指定要輸出最後 conv
            outs.append(x)

        # 只回傳一個就直接回傳張量，否則 tuple
        if len(outs) == 1:
            return outs[0]
        return tuple(outs)

    def train(self, mode=True):
        """模型 train/val 模式切換時，需要依需求固定某些層的參數。"""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            # 將 BatchNorm 設為 eval
            for m in self.modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()