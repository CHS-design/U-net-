"""utils/losses.py — BCE + Dice 组合损失"""

import torch
from torch import nn

from configs import config as cfg


class DiceLoss(nn.Module):
    """输入 logits 与 {0,1} target，返回 1 - 平均 Dice"""

    def __init__(self, smooth=1.0):
        super().__init__()
        # TODO 1: self.smooth = smooth
        pass

    def forward(self, logits, target):
        # TODO 2: probs = torch.sigmoid(logits)
        # TODO 3: inter = (probs * target).sum(dim=(1, 2, 3))
        # TODO 4: union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        # TODO 5: dice = (2 * inter + self.smooth) / (union + self.smooth)
        # TODO 6: return 1 - dice.mean()
        pass


class BCEDiceLoss(nn.Module):
    """bce_weight * BCE + dice_weight * DiceLoss"""

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        # TODO 7: self.bce = nn.BCEWithLogitsLoss()
        # TODO 8: self.dice = DiceLoss()
        # TODO 9: 保存 bce_weight / dice_weight
        pass

    def forward(self, logits, target):
        # TODO 10: 返回加权和
        pass


def build_loss():
    """按 cfg.LOSS 返回可调用损失"""
    # TODO 11: 三个分支 + 非法值 raise ValueError
    pass
