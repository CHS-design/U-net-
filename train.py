"""train.py — 训练入口（阶段 5 实现）

职责：
    1. 读 configs/config.py 的超参
    2. 调 utils/data.py 构建 train/val DataLoader
    3. 建 UNet 并 .to(cfg.DEVICE)
    4. 每个 epoch 走五步：zero_grad → forward → loss → backward → step
    5. 验证：model.eval() + torch.no_grad()，统计 val loss / Dice / IoU
    6. val loss 创新低才存 best.pt；patience 到上限早停
    7. 每 epoch 指标写入 checkpoints/{RUN_NAME}/loss_history.json

状态：train_one_epoch / evaluate / main 已实现，支持 --smoke 冒烟。
"""

import argparse
import json
import os
import random

import numpy as np
import torch

from configs import config as cfg
from models.unet import UNet
from utils.data import build_dataloaders
from utils.losses import build_loss
from utils.metrics import MetricTracker


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None):
    """跑一个 epoch 的训练，返回 (平均 loss, 指标字典)"""
    model.train()
    tracker = MetricTracker()
    total_loss = 0.0
    total_samples = 0

    for i, (images, masks) in enumerate(loader):
        # TODO 1: max_batches 限制（冒烟用）
        if max_batches is not None and i >= max_batches:
            break

        # TODO 2: images / masks 搬到 device
        images = images.to(device)
        masks = masks.to(device)

        # TODO 3: optimizer.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)

        # TODO 4: logits = model(images)
        logits = model(images)
        # TODO 5: loss = criterion(logits, masks)
        loss = criterion(logits, masks)
        # TODO 6: loss.backward()
        loss.backward()
        # TODO 7: 如果 cfg.GRAD_CLIP > 0，调用 torch.nn.utils.clip_grad_norm_
        if cfg.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
        # TODO 8: optimizer.step()
        optimizer.step()
        # TODO 9: tracker.update(logits.detach(), masks)
        tracker.update(logits.detach(), masks)
        # TODO 10: 按 batch_size 加权累计 loss
        loss = loss.item() * images.size(0)
        total_loss += loss
        total_samples += images.size(0)

    # TODO 11: 返回平均 loss 和 tracker.compute()
    return total_loss / total_samples, tracker.compute()


def evaluate(model, loader, criterion, device, max_batches=None):
    """跑验证/测试，返回 (平均 loss, 指标字典)"""
    model.eval()
    tracker = MetricTracker()
    total_loss = 0.0
    total_samples = 0

    # TODO 12: with torch.no_grad():
    # TODO 13: 遍历 loader，max_batches 限制
    # TODO 14: 搬数据、forward、算 loss
    # TODO 15: tracker.update(logits, masks)
    # TODO 16: 按 batch_size 加权累计 loss
    # TODO 17: 返回平均 loss 和 tracker.compute()
    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device)
            masks = masks.to(device)
            logits = model(images)
            loss = criterion(logits, masks)
            tracker.update(logits, masks)
            loss = loss.item() * images.size(0)
            total_loss += loss
            total_samples += images.size(0)
    return total_loss / total_samples, tracker.compute()


def set_seed(seed):
    # TODO 1: random.seed / np.random.seed / torch.manual_seed / torch.cuda.manual_seed_all
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_optimizer(model):
    # TODO 2: adam / sgd / 非法值 ValueError
    if cfg.OPTIMIZER == "adam":
        return torch.optim.Adam(
            model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY
        )
    elif cfg.OPTIMIZER == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.LR,
            momentum=cfg.MOMENTUM,
            weight_decay=cfg.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.OPTIMIZER}")


def main(max_train_batches=None, max_val_batches=None):
    set_seed(cfg.SEED)
    # TODO 3: os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    # TODO 4: train_loader, val_loader, _ = build_dataloaders()
    train_loader, val_loader, _ = build_dataloaders()
    # TODO 5: model = UNet().to(cfg.DEVICE)
    model = UNet().to(cfg.DEVICE)
    # TODO 6: criterion = build_loss()；optimizer = build_optimizer(model)
    criterion = build_loss()
    optimizer = build_optimizer(model)

    best_loss, patience, history = float("inf"), 0, []

    for epoch in range(1, cfg.EPOCHS + 1):
        # TODO 7: train_loss, train_metrics = train_one_epoch(
        #             model, train_loader, criterion, optimizer, cfg.DEVICE, max_train_batches)
        # TODO 8: val_loss, val_metrics = evaluate(
        #             model, val_loader, criterion, cfg.DEVICE, max_val_batches)
        # TODO 9: history.append({...})
        # TODO 10: print epoch 行
        # TODO 11: 如果 val_loss < best_loss：保存 best.pt、patience=0
        #          否则 patience += 1，达到 cfg.PATIENCE 就 break
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE, max_train_batches
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, cfg.DEVICE, max_val_batches
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best_loss:
            best_loss = val_loss
            patience = 0
            torch.save(model.state_dict(), os.path.join(cfg.CKPT_DIR, "best.pt"))
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                break

    # TODO 12: json.dump(history, os.path.join(cfg.CKPT_DIR, "loss_history.json"), ...)
    history_path = os.path.join(cfg.CKPT_DIR, "loss_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # TODO 13: 添加 --smoke / --max-train-batches / --max-val-batches
    # TODO 14: 如果 args.smoke：覆盖 cfg 并设置 max batches
    # TODO 15: main(args.max_train_batches, args.max_val_batches)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="切到 toy 小配置，跑 2 个训练 batch / 1 个验证 batch",
    )
    parser.add_argument(
        "--max-train-batches", type=int, help="冒烟用，只跑 max-train-batches 个 batch"
    )
    parser.add_argument(
        "--max-val-batches", type=int, help="冒烟用，只跑 max-val-batches 个 batch"
    )
    args = parser.parse_args()

    if args.smoke:
        cfg.DATASET = "toy"
        cfg.BATCH_SIZE = 2
        cfg.IMAGE_SIZE = 64
        cfg.NUM_WORKERS = 0
        cfg.EPOCHS = 1
        cfg.RUN_NAME = "smoke_toy"
        cfg.CKPT_DIR = f"./checkpoints/{cfg.RUN_NAME}"
        if args.max_train_batches is None:
            args.max_train_batches = 2
        if args.max_val_batches is None:
            args.max_val_batches = 1

    main(args.max_train_batches, args.max_val_batches)
