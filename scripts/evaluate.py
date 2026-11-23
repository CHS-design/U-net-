"""evaluate.py — 测试集评估入口

职责：
    1. 从 checkpoints/{RUN_NAME}/best.pt 加载 state_dict
    2. model.eval() 后在 test 集上统计损失与指标（二值 Dice/IoU/acc 或多类 mIoU/mAcc）
    3. test 集只在最后用一次；训练期选优只用 val
    4. 输出总指标；test 集不参与训练或早停

用法：
    python scripts/evaluate.py
    python scripts/evaluate.py --run-name smoke_toy --smoke --max-batches 1
"""

import argparse
import os
import sys

# 允许从项目根目录直接运行 python scripts/xxx.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from configs import config as cfg
from models.unet import UNet
from train import evaluate as run_eval
from utils.data import build_datasets
from utils.losses import build_loss


def load_model(ckpt_path, device):
    """加载 best.pt，返回 eval 模式的模型"""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")

    model = UNet().to(device).to(memory_format=torch.channels_last)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def main(run_name=None, max_batches=None):
    run_name = run_name or cfg.RUN_NAME
    ckpt_path = os.path.join("./checkpoints", run_name, "best.pt")

    model = load_model(ckpt_path, cfg.DEVICE)

    _, _, test_set = build_datasets()
    test_loader = DataLoader(
        test_set,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
    )

    criterion = build_loss()
    loss, metrics = run_eval(model, test_loader, criterion, cfg.DEVICE, max_batches)

    print(f"run: {run_name}")
    print(f"checkpoint: {ckpt_path}")
    print(f"test samples: {len(test_set)}")
    print(f"loss: {loss:.4f}")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        default=None,
        help="实验名，默认 cfg.RUN_NAME；--smoke 时默认 smoke_toy",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="只评估前 N 个 batch（冒烟用）",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用 toy 小配置，配合 smoke_toy 权重做流程验收",
    )
    args = parser.parse_args()

    if args.smoke:
        cfg.DATASET = "toy"
        cfg.TASK = "binary"
        cfg.OUT_CHANNELS = 1
        cfg.LOSS = "bce_dice"
        cfg.BATCH_SIZE = 2
        cfg.IMAGE_SIZE = 64
        cfg.NUM_WORKERS = 0
        if args.run_name is None:
            args.run_name = "smoke_toy"
        if args.max_batches is None:
            args.max_batches = 1

    main(args.run_name, args.max_batches)
