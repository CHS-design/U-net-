"""models/unet.py — U-Net 总装（same 模式）"""

from torch import nn

from configs import config as cfg
from models.double_conv import DoubleConv
from models.down import Down
from models.up import Up
from models.out_conv import OutConv


class UNet(nn.Module):
    """(B, 3, H, W) -> (B, 1, H, W)"""

    def __init__(self):
        super().__init__()
        # TODO 1: self.inc = DoubleConv(cfg.IN_CHANNELS, cfg.CHANNELS[0])
        # TODO 2: self.down1 = Down(cfg.CHANNELS[0], cfg.CHANNELS[1])
        # TODO 3: self.down2 = Down(cfg.CHANNELS[1], cfg.CHANNELS[2])
        # TODO 4: self.down3 = Down(cfg.CHANNELS[2], cfg.CHANNELS[3])
        # TODO 5: self.down4 = Down(cfg.CHANNELS[3], cfg.CHANNELS[4])
        # TODO 6: self.up1 = Up(cfg.CHANNELS[4], cfg.CHANNELS[3])
        # TODO 7: self.up2 = Up(cfg.CHANNELS[3], cfg.CHANNELS[2])
        # TODO 8: self.up3 = Up(cfg.CHANNELS[2], cfg.CHANNELS[1])
        # TODO 9: self.up4 = Up(cfg.CHANNELS[1], cfg.CHANNELS[0])
        # TODO 10: self.outc = OutConv(cfg.CHANNELS[0], cfg.OUT_CHANNELS)
        self.inc = DoubleConv(cfg.IN_CHANNELS, cfg.CHANNELS[0])
        self.down1 = Down(cfg.CHANNELS[0], cfg.CHANNELS[1])
        self.down2 = Down(cfg.CHANNELS[1], cfg.CHANNELS[2])
        self.down3 = Down(cfg.CHANNELS[2], cfg.CHANNELS[3])
        self.down4 = Down(cfg.CHANNELS[3], cfg.CHANNELS[4])
        self.up1 = Up(cfg.CHANNELS[4], cfg.CHANNELS[3])
        self.up2 = Up(cfg.CHANNELS[3], cfg.CHANNELS[2])
        self.up3 = Up(cfg.CHANNELS[2], cfg.CHANNELS[1])
        self.up4 = Up(cfg.CHANNELS[1], cfg.CHANNELS[0])
        self.outc = OutConv(cfg.CHANNELS[0], cfg.OUT_CHANNELS)  

    def forward(self, x):
        # TODO 11: x1 = self.inc(x)
        # TODO 12: x2 = self.down1(x1); x3 = self.down2(x2)
        #          x4 = self.down3(x3); x5 = self.down4(x4)
        # TODO 13: x = self.up1(x5, x4)
        #          x = self.up2(x, x3)
        #          x = self.up3(x, x2)
        #          x = self.up4(x, x1)
        # TODO 14: return self.outc(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)
