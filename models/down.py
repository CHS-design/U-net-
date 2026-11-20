"""models/down.py — 下采样：MaxPool2d(2) + DoubleConv"""

from torch import nn

from models.double_conv import DoubleConv


class Down(nn.Module):
    """(B, C, H, W) -> (B, 2C, H/2, W/2)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO 1: self.pool = nn.MaxPool2d(2)
        self.pool = nn.MaxPool2d(2)
        # TODO 2: self.conv = DoubleConv(in_channels, out_channels)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        # TODO 3: 先池化
        x = self.pool(x)
        x = self.conv(x)
        # TODO 4: 再双卷积
        # TODO 5: 返回
        return x
