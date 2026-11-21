"""plot_curve.py — 由 loss_history.json 独立画图

职责：
    1. 读 checkpoints/{RUN_NAME}/loss_history.json
    2. 画 train/val 的 loss / Dice / IoU / Acc 曲线
    3. 保存 loss_curve.png 到同一实验目录
    4. 与训练解耦：随时可重跑，不依赖训练进程

用法：
    python plot_curve.py
    python plot_curve.py --run-name smoke_toy
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # 必须在 import pyplot 之前：服务器无图形界面也能存图
import matplotlib.pyplot as plt

from configs import config as cfg


def _metric(record, split, name):
    """从一条 history 记录里取指标；优先读嵌套的 train_metrics/val_metrics"""
    metrics = record.get(f"{split}_metrics")
    if metrics is None:
        raise KeyError(
            f"history 缺少 {split}_metrics，旧格式请重新训练或手动转换"
        )
    return metrics[name]


def _plot_metric(ax, epochs, train_values, val_values, title, ylabel):
    ax.plot(epochs, train_values, marker="o", label="train")
    ax.plot(epochs, val_values, marker="o", label="val")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()


def main(run_name=None):
    run_name = run_name or cfg.RUN_NAME
    ckpt_dir = os.path.join("./checkpoints", run_name)
    hist_path = os.path.join(ckpt_dir, "loss_history.json")
    out_path = os.path.join(ckpt_dir, "loss_curve.png")

    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"找不到训练历史: {hist_path}")

    with open(hist_path, encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        raise ValueError(f"训练历史为空: {hist_path}")

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    train_dice = [_metric(h, "train", "dice") for h in history]
    val_dice = [_metric(h, "val", "dice") for h in history]
    train_iou = [_metric(h, "train", "iou") for h in history]
    val_iou = [_metric(h, "val", "iou") for h in history]
    train_acc = [_metric(h, "train", "acc") for h in history]
    val_acc = [_metric(h, "val", "acc") for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_metric(axes[0, 0], epochs, train_loss, val_loss, "Loss", "loss")
    _plot_metric(
        axes[0, 1], epochs, train_dice, val_dice, "Dice", "dice"
    )
    _plot_metric(axes[1, 0], epochs, train_iou, val_iou, "IoU", "iou")
    _plot_metric(
        axes[1, 1], epochs, train_acc, val_acc, "Pixel Accuracy", "accuracy"
    )

    fig.suptitle(f"{run_name} — training curves")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=cfg.RUN_NAME,
        help="实验名，对应 checkpoints/{run_name}/loss_history.json",
    )
    args = parser.parse_args()
    main(args.run_name)
