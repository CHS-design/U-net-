"""models/double_conv.py — 两次 3×3 卷积 + 可选 BN + ReLU"""

from torch import nn

from configs import config as cfg


class DoubleConv(nn.Module):
    """(B, Cin, H, W) -> (B, Cout, H', W')"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # TODO 1: padding = 1 if cfg.PADDING_MODE == "same" else 0
        padding = 1 if cfg.PADDING_MODE == "same" else 0
        # TODO 2: bias = not cfg.USE_BATCH_NORM
        bias = not cfg.USE_BATCH_NORM
        # TODO 3: self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=padding, bias=bias)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=padding, bias=bias)
        # TODO 4: self.bn1 = nn.BatchNorm2d(out_channels) if cfg.USE_BATCH_NORM else nn.Identity()
        self.bn1 = nn.BatchNorm2d(out_channels) if cfg.USE_BATCH_NORM else nn.Identity()
        # TODO 5: self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=padding, bias=bias)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=padding, bias=bias
        )
        # TODO 6: self.bn2 = 同上
        self.bn2 = nn.BatchNorm2d(out_channels) if cfg.USE_BATCH_NORM else nn.Identity()
        # TODO 7: self.relu = nn.ReLU(inplace=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # TODO 8: 第一层 conv → bn → relu
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # TODO 9: 第二层 conv → bn → relu
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # TODO 10: 返回 x
        return x
