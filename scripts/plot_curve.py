"""plot_curve.py — 由 loss_history.json 独立画图

职责：
    1. 读 checkpoints/{RUN_NAME}/loss_history.json
    2. 画 train/val 的 loss 与 history 中实际记录的指标曲线
    3. 保存 loss_curve.png 到同一实验目录
    4. 与训练解耦：随时可重跑，不依赖训练进程

用法：
    python scripts/plot_curve.py
    python scripts/plot_curve.py --run-name smoke_toy
"""

import argparse
import json
import os
import sys

# 允许从项目根目录直接运行 python scripts/xxx.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    metric_names = list(history[0]["train_metrics"].keys())
    fig, axes = plt.subplots(
        1, 1 + len(metric_names), figsize=(4 * (1 + len(metric_names)), 4)
    )
    axes = axes if hasattr(axes, "__len__") else [axes]
    _plot_metric(axes[0], epochs, train_loss, val_loss, "Loss", "loss")
    for ax, name in zip(axes[1:], metric_names):
        _plot_metric(
            ax,
            epochs,
            [_metric(h, "train", name) for h in history],
            [_metric(h, "val", name) for h in history],
            name.upper(),
            name,
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
