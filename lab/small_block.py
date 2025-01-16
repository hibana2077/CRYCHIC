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