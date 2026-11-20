"""models/up.py — 上采样 + 跳跃拼接 + DoubleConv（same 模式）"""

import torch
from torch import nn

from configs import config as cfg          # 新增
from models.double_conv import DoubleConv



class Up(nn.Module):
    """(B,in,H,W) + (B,out,2H,2W) -> (B,out,2H,2W)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO 1: 如果 cfg.BILINEAR：
        #             self.up = nn.Sequential(
        #                 nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        #                 nn.Conv2d(in_channels, out_channels, kernel_size=1),
        #             )
        #         否则：
        #             self.up = nn.ConvTranspose2d(
        #                 in_channels, out_channels, kernel_size=2, stride=2
        #             )
        # TODO 2: self.conv = DoubleConv(out_channels * 2, out_channels)
        if cfg.BILINEAR:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
            )
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, out_channels, kernel_size=2, stride=2
            )
        self.conv = DoubleConv(out_channels * 2, out_channels)


    def forward(self, x1, x2):
        # TODO 3: x1 = self.up(x1)
        # TODO 4: x = torch.cat([x2, x1], dim=1)
        # TODO 5: return self.conv(x)
        x1= self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)
