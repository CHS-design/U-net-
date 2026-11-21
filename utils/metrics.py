"""utils/metrics.py — 分割指标：Dice / IoU / 像素准确率"""

import torch

from configs import config as cfg

EPS = 1e-6


def _pred_mask(logits, threshold=cfg.THRESHOLD):
    """logits -> 硬 0/1 掩膜"""
    # TODO 1: return (torch.sigmoid(logits) > threshold).float()
    return (torch.sigmoid(logits) > threshold).float()


def dice_score(logits, target, threshold=cfg.THRESHOLD, eps=EPS):
    """返回每张图的 Dice，形状 (B,)"""
    # TODO 2: pred = _pred_mask(logits, threshold)
    # TODO 3: inter = (pred * target).sum(dim=(1, 2, 3))
    # TODO 4: denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    # TODO 5: return (2 * inter + eps) / (denom + eps)
    pred = _pred_mask(logits, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (denom + eps)


def iou_score(logits, target, threshold=cfg.THRESHOLD, eps=EPS):
    """返回每张图的 IoU，形状 (B,)"""
    # TODO 6: pred = _pred_mask(...)
    # TODO 7: inter = ...
    # TODO 8: union = pred.sum + target.sum - inter
    # TODO 9: return (inter + eps) / (union + eps)
    pred = _pred_mask(logits, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return (inter + eps) / (union + eps)


def pixel_accuracy(logits, target, threshold=cfg.THRESHOLD):
    """返回每张图的像素准确率，形状 (B,)"""
    # TODO 10: pred = _pred_mask(...)
    # TODO 11: return (pred == target).float().mean(dim=(1, 2, 3))
    pred = _pred_mask(logits, threshold)
    return (pred == target).float().mean(dim=(1, 2, 3))


class MetricTracker:
    """跨 batch 累计全局统计量，避免小 batch 偏置"""

    def __init__(self):
        self.reset()

    def reset(self):
        # TODO 12: inter / pred_sum / true_sum / correct / total 全部清零
        self.inter = 0
        self.pred_sum = 0
        self.true_sum = 0
        self.correct = 0
        self.total = 0

    def update(self, logits, target, threshold=cfg.THRESHOLD):
        # TODO 13: pred = _pred_mask(logits, threshold)
        # TODO 14: 累加 inter、pred_sum、true_sum
        # TODO 15: 累加 correct（pred == target）与 total（target.numel()）
        pred = _pred_mask(logits, threshold)
        self.inter += (pred * target).sum().item()
        self.pred_sum += pred.sum().item()
        self.true_sum += target.sum().item()
        self.correct += (pred == target).float().sum().item()
        self.total += target.numel()

    def compute(self):
        # TODO 16: 用累计量算 dice / iou / acc，返回字典
        dice = (2 * self.inter + EPS) / (self.pred_sum + self.true_sum + EPS)
        iou = (self.inter + EPS) / (self.pred_sum + self.true_sum - self.inter + EPS)
        acc = self.correct / self.total
        return {"dice": dice, "iou": iou, "acc": acc}
