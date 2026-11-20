"""models/out_conv.py — 1×1 卷积输出 logits"""

from torch import nn


class OutConv(nn.Module):
    """(B, Cin, H, W) -> (B, Cout, H, W)"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO 1: self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # TODO 2: 返回 self.conv(x)
        return self.conv(x)
