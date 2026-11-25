"""utils/losses.py — BCE + Dice 组合损失"""

import torch
from torch import nn

from configs import config as cfg


class DiceLoss(nn.Module):
    """输入 logits 与 {0,1} target，返回 1 - 平均 Dice"""

    def __init__(self, smooth=1.0):
        super().__init__()
        # TODO 1: self.smooth = smooth
        self.smooth = smooth

    def forward(self, logits, target):
        # TODO 2: probs = torch.sigmoid(logits)
        # TODO 3: inter = (probs * target).sum(dim=(1, 2, 3))
        # TODO 4: union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        # TODO 5: dice = (2 * inter + self.smooth) / (union + self.smooth)
        # TODO 6: return 1 - dice.mean()
        probs = torch.sigmoid(logits)
        inter = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """bce_weight * BCE + dice_weight * DiceLoss"""

    def __init__(self, bce_weight=1.0, dice_weight=1.0):
        super().__init__()
        # TODO 7: self.bce = nn.BCEWithLogitsLoss()
        # TODO 8: self.dice = DiceLoss()
        # TODO 9: 保存 bce_weight / dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, target):
        # TODO 10: 返回加权和
        bce_loss = self.bce(logits, target)
        dice_loss = self.dice(logits, target)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def compute_class_weights(dataset, num_classes, power=0.5):
    """统计训练掩膜的类别频率，返回加权 CE 的类别权重。

    weight[c] = (1 / freq[c]) ** power，再归一化到均值 1 保持 loss 量级。
    直接读 dataset 的掩膜文件，不经过 transform，速度快。
    """
    from pathlib import Path

    import numpy as np
    from PIL import Image

    masks_dir = Path(dataset.masks_dir)
    fixes = getattr(dataset, "label_fixes", {})
    split = getattr(dataset, "split", "")
    counts = np.zeros(num_classes, dtype=np.float64)
    for image_path in dataset.image_paths:
        mask_path = masks_dir / f"{image_path.stem}.png"
        arr = np.asarray(Image.open(mask_path).convert("L"))
        # 与 Dataset 的加载时修正保持一致，否则少数样本的类别计数会落到错误的类
        fix = fixes.get(f"{split}/{mask_path.name}")
        if fix is not None:
            arr = np.where(arr == fix["from"], np.uint8(fix["to"]), arr).astype(np.uint8)
        cls, cnt = np.unique(arr, return_counts=True)
        valid = cls < num_classes
        counts[cls[valid]] += cnt[valid]

    freq = counts / counts.sum()
    freq = np.clip(freq, 1e-6, None)  # 防止除零
    weights = freq ** (-power)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=cfg.DEVICE)


def build_loss(class_weights=None):
    """按 cfg.LOSS 返回可调用损失"""
    if cfg.LOSS == "bce":
        return nn.BCEWithLogitsLoss()
    elif cfg.LOSS == "dice":
        return DiceLoss()
    elif cfg.LOSS == "bce_dice":
        return BCEDiceLoss(cfg.BCE_WEIGHT, cfg.DICE_WEIGHT)
    elif cfg.LOSS == "ce":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"未知损失类型 {cfg.LOSS}")
