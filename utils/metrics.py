"""utils/metrics.py — 分割指标统一模块

集中管理本项目的两类分割指标及其中间统计逻辑：

- 二值分割（TASK="binary"）：Dice / IoU / 像素准确率 Acc
- 多类语义分割（TASK="multiclass"）：mIoU / mAcc（基于混淆矩阵）

对外接口分三层，便于训练、验证、测试、推理各处复用：
    1. 单图函数：dice_score / iou_score / pixel_accuracy / multiclass_scores
    2. 跨 batch 累计器：BinaryMetricTracker / MultiClassMetricTracker
    3. 工厂与元信息：build_metric_tracker / metric_keys / select_key

设计约定（与历史实验数值完全一致，重构不改变任何计算）：
    - 所有统计量在 GPU 上累加，只在 compute() 时同步一次到 CPU（.item()）
    - 多类指标只在"真实出现过的类"（gt > 0）上平均，跳过未出现的类
    - 越界标签（target < 0 或 target >= num_classes）在混淆矩阵构建时被忽略
    - 分母加 EPS 防止除零；不额外做预测侧的 ignore（argmax 不会产生越界预测）
"""

import torch

from configs import config as cfg

EPS = 1e-6


# --------------------------------------------------------------------------- #
# 一、二值分割指标
# --------------------------------------------------------------------------- #

def _pred_mask(logits, threshold=cfg.THRESHOLD):
    """二值 logits → 硬 0/1 掩膜（sigmoid + 阈值）"""
    return (torch.sigmoid(logits) > threshold).float()


def dice_score(logits, target, threshold=cfg.THRESHOLD, eps=EPS):
    """返回每张图的 Dice，形状 (B,)"""
    pred = _pred_mask(logits, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3))
    denom = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    return (2 * inter + eps) / (denom + eps)


def iou_score(logits, target, threshold=cfg.THRESHOLD, eps=EPS):
    """返回每张图的 IoU，形状 (B,)"""
    pred = _pred_mask(logits, threshold)
    inter = (pred * target).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return (inter + eps) / (union + eps)


def pixel_accuracy(logits, target, threshold=cfg.THRESHOLD):
    """返回每张图的像素准确率，形状 (B,)"""
    pred = _pred_mask(logits, threshold)
    return (pred == target).float().mean(dim=(1, 2, 3))


class BinaryMetricTracker:
    """二值分割的跨 batch 累计器

    累计全局交集 / 预测前景 / 真实前景 / 正确像素，compute() 算全局 Dice / IoU / Acc，
    避免逐图平均带来的小图偏置。累加量留在 GPU 上，只在 compute 时同步一次。
    """

    def __init__(self, device):
        self.device = device
        self.reset()

    def reset(self):
        self.inter = torch.zeros((), device=self.device, dtype=torch.float32)
        self.pred_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self.true_sum = torch.zeros((), device=self.device, dtype=torch.float32)
        self.correct = torch.zeros((), device=self.device, dtype=torch.float32)
        self.total = 0  # numel() 是 CPU 元数据读取，无 GPU 同步开销

    def update(self, logits, target, threshold=cfg.THRESHOLD):
        pred = _pred_mask(logits, threshold)
        self.inter += (pred * target).sum()
        self.pred_sum += pred.sum()
        self.true_sum += target.sum()
        self.correct += (pred == target).float().sum()
        self.total += target.numel()

    def compute(self):
        dice = (2 * self.inter + EPS) / (self.pred_sum + self.true_sum + EPS)
        iou = (self.inter + EPS) / (self.pred_sum + self.true_sum - self.inter + EPS)
        acc = self.correct / self.total
        return {"dice": dice.item(), "iou": iou.item(), "acc": acc.item()}


# 保留旧名称作为别名，避免外部历史代码引用失效
MetricTracker = BinaryMetricTracker


# --------------------------------------------------------------------------- #
# 二、多类语义分割指标（mIoU / mAcc）
# --------------------------------------------------------------------------- #

def validate_labels(target, num_classes, source=""):
    """校验类别索引图取值在 [0, num_classes) 内（多类任务的数据前置约束）

    data.py 加载掩膜、MultiClassMetricTracker 收到 target 时都遵循这一约定：
    越界标签会在混淆矩阵构建时被 ignore（见 multiclass_confusion），
    但数据源头就保证合法，能让指标统计与 CE 损失都可信。
    """
    max_cls = int(target.max())
    if max_cls >= num_classes or max_cls < 0:
        raise ValueError(
            f"掩膜类别索引越界{source}: max={max_cls}, NUM_CLASSES={num_classes}"
        )


def multiclass_confusion(logits, target, num_classes, device=None):
    """构建混淆矩阵 conf[true, pred]，形状 (num_classes, num_classes)

    步骤：
        1. pred = logits.argmax(dim=1)：多类 logits 直接取最大概率通道，无需 softmax
        2. valid = 0 <= target < num_classes：忽略越界标签（如 padding / 未标注像素）
        3. 把 (true, pred) 二维坐标压成一维索引 true * num_classes + pred
        4. bincount + view 还原成混淆矩阵

    参数：
        logits: (B, C, H, W) float
        target: (B, H, W) long，类别索引
        num_classes: 类别总数（含背景）
        device: 输出矩阵所在设备；默认跟随 target
    """
    pred = logits.argmax(dim=1)              # (B,H,W)，argmax 前不需要 softmax
    target = target.long()
    valid = (target >= 0) & (target < num_classes)
    idx = target[valid] * num_classes + pred[valid]  # 2D 坐标压成 1D
    conf = torch.bincount(idx, minlength=num_classes**2).view(num_classes, num_classes)
    return conf if device is None else conf.to(device)


def multiclass_scores(conf, eps=EPS):
    """从混淆矩阵计算逐类统计量与 mIoU / mAcc（仍在设备上，未同步 CPU）

    返回字典：
        inter:      逐类 TP（对角线），形状 (C,)
        gt:         逐类真实像素数（行和），形状 (C,)
        pred:       逐类预测像素数（列和），形状 (C,)
        union:      逐类并集 = gt + pred - inter，形状 (C,)
        present:    该类是否在真实标签中出现（gt > 0），形状 (C,) bool
        iou / acc:  逐类 IoU / Acc（已加 eps），形状 (C,)
        miou / macc: 标量，在 present 类上取均值
    """
    inter = torch.diag(conf)       # 每类 TP
    gt = conf.sum(dim=1)           # 真实为该类的像素数
    pred = conf.sum(dim=0)         # 预测为该类的像素数
    union = gt + pred - inter
    present = gt > 0               # 只在真实出现过的类上平均
    iou = (inter + eps) / (union + eps)
    acc = inter / (gt + eps)
    miou = iou[present].mean()
    macc = acc[present].mean()
    return {
        "inter": inter, "gt": gt, "pred": pred, "union": union,
        "present": present, "iou": iou, "acc": acc,
        "miou": miou, "macc": macc,
    }


class MultiClassMetricTracker:
    """多类语义分割的跨 batch 累计器

    累计混淆矩阵，compute() 时按 present 掩码对逐类 IoU / Acc 求平均。
    混淆矩阵的构建与分数计算分别委托给 multiclass_confusion / multiclass_scores，
    便于单测、可视化或自定义评估直接复用纯函数。
    """

    def __init__(self, device, num_classes=cfg.NUM_CLASSES):
        self.num_classes = num_classes
        self.conf = torch.zeros(
            (num_classes, num_classes), device=device, dtype=torch.float32
        )

    def update(self, logits, target):
        self.conf += multiclass_confusion(logits, target, self.num_classes, self.conf.device)

    def compute(self):
        scores = multiclass_scores(self.conf)
        return {"miou": scores["miou"].item(), "macc": scores["macc"].item()}


# --------------------------------------------------------------------------- #
# 三、工厂与元信息（统一任务 → 指标映射，供训练/评估/日志共用）
# --------------------------------------------------------------------------- #

def metric_keys(task=None):
    """返回该任务在日志/JSON 中的指标键名列表"""
    task = task or cfg.TASK
    if task == "multiclass":
        return ["miou", "macc"]
    elif task == "binary":
        return ["dice", "iou", "acc"]
    raise ValueError(f"未知 TASK: {task!r}")


def select_key(task=None):
    """返回该任务用于选优 / 早停的指标键（越大越好）"""
    task = task or cfg.TASK
    if task == "multiclass":
        return "miou"
    elif task == "binary":
        return "dice"
    raise ValueError(f"未知 TASK: {task!r}")


def build_metric_tracker(device, task=None, num_classes=None):
    """按任务创建指标累计器；num_classes 仅多类任务使用"""
    task = task or cfg.TASK
    num_classes = num_classes or cfg.NUM_CLASSES
    if task == "multiclass":
        return MultiClassMetricTracker(device, num_classes)
    elif task == "binary":
        return BinaryMetricTracker(device)
    raise ValueError(f"未知 TASK: {task!r}")
