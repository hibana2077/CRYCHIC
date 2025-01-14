import math
from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import SelectAdaptivePool2d, Linear, make_divisible
from ._builder import build_model_with_cfg
from ._efficientnet_blocks import SqueezeExcite, ConvBnAct
from ._manipulate import checkpoint_seq
from ._registry import register_model, generate_default_cfgs

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


class GhostNet3D(nn.Module):
    """
    GhostNet 的 3D 版本，將所有 2D layer 改成 3D layer。
    建議依需求調整各層的 kernel_size、stride、padding 等參數。
    """
    def __init__(
            self,
            cfgs,
            num_classes=1000,
            width=1.0,
            in_chans=3,
            output_stride=32,
            global_pool='avg',
            drop_rate=0.2,
            version='v1',
    ):
        super().__init__()
        assert output_stride == 32, '目前僅示範固定輸出步幅為 32，不支援 dilation'
        self.cfgs = cfgs
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.feature_info = []

        # 首層輸入 (in_chans -> stem_chs)
        stem_chs = make_divisible(16 * width, 4)
        self.conv_stem = nn.Conv3d(in_chans, stem_chs, kernel_size=3, stride=2, padding=1, bias=False)
        self.feature_info.append(dict(num_chs=stem_chs, reduction=2, module=f'conv_stem'))
        self.bn1 = nn.BatchNorm3d(stem_chs)
        self.act1 = nn.ReLU(inplace=True)
        prev_chs = stem_chs

        # 建立中間的 GhostBottleneck3D
        stages = nn.ModuleList([])
        stage_idx = 0
        layer_idx = 0
        net_stride = 2
        for cfg in self.cfgs:
            layers = []
            s = 1
            for k, exp_size, c, se_ratio, s in cfg:
                out_chs = make_divisible(c * width, 4)
                mid_chs = make_divisible(exp_size * width, 4)
                layer_kwargs = {}
                if version == 'v2' and layer_idx > 1:
                    layer_kwargs['mode'] = 'attn'  # 僅示範使用 mode='attn' 分支
                layers.append(
                    GhostBottleneck3D(prev_chs, mid_chs, out_chs, k, s, se_ratio=se_ratio, **layer_kwargs)
                )
                prev_chs = out_chs
                layer_idx += 1

            if s > 1:
                net_stride *= 2
                self.feature_info.append(dict(
                    num_chs=prev_chs, reduction=net_stride, module=f'blocks.{stage_idx}'
                ))
            stages.append(nn.Sequential(*layers))
            stage_idx += 1

        # 最後一層
        out_chs = make_divisible(exp_size * width, 4)
        stages.append(nn.Sequential(
            ConvBnAct(prev_chs, out_chs, kernel_size=1, act_layer=nn.ReLU)
        ))
        self.pool_dim = prev_chs = out_chs
        self.blocks = nn.Sequential(*stages)

        # head
        self.num_features = prev_chs
        self.head_hidden_size = out_chs = 1280
        # 這裡為了簡易示範，仍沿用 2D AdaptivePool，但實際上要改成 3D 版本或自定義 pool
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.conv_head = nn.Conv3d(prev_chs, out_chs, kernel_size=1, stride=1, padding=0, bias=True)
        self.act2 = nn.ReLU(inplace=True)

        # 這裡因為 SelectAdaptivePool2d 會先把維度縮成 2D，所以 flatten 在 forward_head 前
        # 也可以根據需求，改寫成 nn.AdaptiveAvgPool3d(...) + Flatten。
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(out_chs, num_classes) if num_classes > 0 else nn.Identity()

    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        matcher = dict(
            stem=r'^conv_stem|bn1',
            blocks=[
                (r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)', None),
                (r'conv_head', (99999,))
            ]
        )
        return matcher

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        """
        x shape: (N, C, D, H, W)
        """
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x

    def forward_head(self, x, pre_logits: bool = False):
        # 先做 3D conv，再交給 2D 的 SelectAdaptivePool2d 縮成 (N, C, 1, 1)
        # 若要真正 3D Pool，可以換成 nn.AdaptiveAvgPool3d(output_size=(1,1,1)) 再 flatten
        # 這裡僅示範邏輯
        x = self.conv_head(x)  # (N, 1280, D, H, W)
        x = self.act2(x)
        # SelectAdaptivePool2d 只會偵測到最後兩維，因此要先壓掉深度 D。
        # 若不想壓深度，可自行改用 3D pool。
        # 這裡簡單示範：先做個簡單的 3D AvgPool，把 D, H, W 都壓成 1
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))  # => (N, 1280, 1, 1, 1)
        x = x.squeeze(-1).squeeze(-1).squeeze(-1)  # => (N, 1280)
        # dropout
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        return x if pre_logits else self.classifier(x)

    def forward(self, x):
        """
        x shape: (N, C, D, H, W)
        """
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def checkpoint_filter_fn(state_dict, model: nn.Module):
    out_dict = {}
    for k, v in state_dict.items():
        if 'total' in k:
            continue
        out_dict[k] = v
    return out_dict


def _create_ghostnet3d(variant, width=1.0, pretrained=False, **kwargs):
    """
    可以比照 2D GhostNet 寫法，透過 build_model_with_cfg 來產生模型。
    這裡僅提供示範，因此不針對預訓練權重做處理。
    """
    cfgs = [
        # k, t, c, SE, s
        # stage1
        [[3, 16, 16, 0, 1]],
        # stage2
        [[3, 48, 24, 0, 2]],
        [[3, 72, 24, 0, 1]],
        # stage3
        [[3, 72, 40, 0.25, 2]],
        [[3, 120, 40, 0.25, 1]],
        # stage4
        [[3, 240, 80, 0, 2]],
        [[3, 200, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 184, 80, 0, 1],
         [3, 480, 112, 0.25, 1],
         [3, 672, 112, 0.25, 1]],
        # stage5
        [[3, 672, 160, 0.25, 2]],
        [[3, 960, 160, 0, 1],
         [3, 960, 160, 0.25, 1],
         [3, 960, 160, 0, 1],
         [3, 960, 160, 0.25, 1]]
    ]
    model_kwargs = dict(
        cfgs=cfgs,
        width=width,
        **kwargs,
    )
    return build_model_with_cfg(
        GhostNet3D,
        variant,
        pretrained,
        pretrained_filter_fn=checkpoint_filter_fn,
        feature_cfg=dict(flatten_sequential=True),
        **model_kwargs,
    )


# 以下示範幾個註冊好的 model，可依需求調整
@register_model
def ghostnet3d_050(pretrained=False, **kwargs) -> GhostNet3D:
    """ GhostNet3D-0.5x """
    model = _create_ghostnet3d('ghostnet3d_050', width=0.5, pretrained=pretrained, **kwargs)
    return model

@register_model
def ghostnet3d_100(pretrained=False, **kwargs) -> GhostNet3D:
    """ GhostNet3D-1.0x """
    model = _create_ghostnet3d('ghostnet3d_100', width=1.0, pretrained=pretrained, **kwargs)
    return model

@register_model
def ghostnet3d_130(pretrained=False, **kwargs) -> GhostNet3D:
    """ GhostNet3D-1.3x """
    model = _create_ghostnet3d('ghostnet3d_130', width=1.3, pretrained=pretrained, **kwargs)
    return model
