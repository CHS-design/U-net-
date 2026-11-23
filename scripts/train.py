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
import functools
import json
import os
import random
import sys
import time
from contextlib import redirect_stderr, redirect_stdout

# 允许从项目根目录直接运行 python scripts/xxx.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.amp import GradScaler, autocast

from configs import config as cfg
from models.unet import UNet
from utils.data import build_dataloaders
from utils.losses import build_loss, compute_class_weights
from utils.metrics import build_metric_tracker, metric_keys, select_key


def train_one_epoch(model, loader, criterion, optimizer, device, max_batches=None, scaler=None):
    """跑一个 epoch 的训练，返回 (平均 loss, 指标字典)"""
    model.train()
    tracker = build_metric_tracker(device)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_samples = 0
    use_amp = scaler is not None

    for i, (images, masks) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break

        images = images.to(device, memory_format=torch.channels_last)
        masks = masks.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp:
            scaler.scale(loss).backward()
            if cfg.GRAD_CLIP > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP)
            optimizer.step()

        tracker.update(logits.detach(), masks)
        total_loss += loss.detach() * images.size(0)
        total_samples += images.size(0)

    return (total_loss / total_samples).item(), tracker.compute()


def evaluate(model, loader, criterion, device, max_batches=None):
    """跑验证/测试，返回 (平均 loss, 指标字典)"""
    model.eval()
    tracker = build_metric_tracker(device)
    total_loss = torch.zeros((), device=device, dtype=torch.float32)
    total_samples = 0
    use_amp = device == "cuda"

    with torch.no_grad():
        for i, (images, masks) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            images = images.to(device, memory_format=torch.channels_last)
            masks = masks.to(device)
            with autocast(device_type="cuda", enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)
            tracker.update(logits.detach(), masks)
            total_loss += loss.detach() * images.size(0)
            total_samples += images.size(0)
    return (total_loss / total_samples).item(), tracker.compute()


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


class Tee:
    """把 print 输出同时写到多个流（终端 + 日志文件）"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
        self.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def tee_to_ckpt(func):
    """装饰器：把函数内的 stdout 同时写入终端与 CKPT_DIR/train.log"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        os.makedirs(cfg.CKPT_DIR, exist_ok=True)
        log_path = os.path.join(cfg.CKPT_DIR, "train.log")
        with open(log_path, "w", encoding="utf-8") as log_file:
            with redirect_stdout(Tee(sys.__stdout__, log_file)):
                return func(*args, **kwargs)

    return wrapper


@tee_to_ckpt
def main(max_train_batches=None, max_val_batches=None, resume_from=None):
    set_seed(cfg.SEED)
    keys, key = metric_keys(), select_key()
    torch.backends.cudnn.benchmark = True
    os.makedirs(cfg.CKPT_DIR, exist_ok=True)
    train_loader, val_loader, _ = build_dataloaders()
    model = UNet().to(cfg.DEVICE).to(memory_format=torch.channels_last)
    model = torch.compile(model)
    if resume_from:
        state = torch.load(resume_from, map_location=cfg.DEVICE, weights_only=True)
        # best.pt 保存时剥掉了 _orig_mod. 前缀，加载回 compile 模型要加回去
        state = {k if k.startswith("_orig_mod.") else f"_orig_mod.{k}": v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
        print(f"已从 {resume_from} 加载权重，继续训练")
    class_weights = None
    if cfg.TASK == "multiclass" and cfg.USE_CLASS_WEIGHTS:
        class_weights = compute_class_weights(
            train_loader.dataset, cfg.NUM_CLASSES, cfg.CLASS_WEIGHT_POWER
        )
        print(
            f"类别权重: min={class_weights.min():.3f} max={class_weights.max():.3f} "
            f"(背景={class_weights[0]:.3f})"
        )
    criterion = build_loss(class_weights)
    optimizer = build_optimizer(model)
    scaler = GradScaler("cuda") if cfg.DEVICE == "cuda" else None

    header = (
        f"{'Epoch':>6}  {'Time':>7}  {'train_loss':>10}  "
        + "".join(f"  {'train_' + k:>10}" for k in keys)
        + f"  {'val_loss':>10}"
        + "".join(f"  {'val_' + k:>10}" for k in keys)
    )
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    best_score, patience, history = -1.0, 0, []  # -1 保证首轮即使指标为 0 也会保存 best.pt
    total_start = time.perf_counter()

    for epoch in range(1, cfg.EPOCHS + 1):
        epoch_start = time.perf_counter()
        train_loss, train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, cfg.DEVICE, max_train_batches, scaler
        )
        val_loss, val_metrics = evaluate(
            model, val_loader, criterion, cfg.DEVICE, max_val_batches
        )
        epoch_time = time.perf_counter() - epoch_start

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
            }
        )
        print(
            f"{epoch:>3d}/{cfg.EPOCHS:<2d}  {epoch_time:>6.1f}s  {train_loss:>10.4f}  "
            + "".join(f"  {train_metrics[k]:>10.4f}" for k in keys)
            + f"  {val_loss:>10.4f}"
            + "".join(f"  {val_metrics[k]:>10.4f}" for k in keys)
        )

        if val_metrics[key] > best_score:
            best_score = val_metrics[key]
            patience = 0
            state = {
                k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
            }
            torch.save(state, os.path.join(cfg.CKPT_DIR, "best.pt"))
        else:
            patience += 1
            if patience >= cfg.PATIENCE:
                print(f"早停：连续 {patience} 轮 val {key} 未创新高")
                break

    total_time = time.perf_counter() - total_start
    print("-" * len(header))
    print(
        f"训练完成 | 共 {len(history)} 轮 | 总耗时 {total_time:.1f}s ({total_time / 60:.2f} min)"
        f" | 平均 {total_time / max(len(history), 1):.1f}s/轮"
    )

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
    parser.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        help="续训：不带值=从当前 RUN_NAME/best.pt 加载；带路径=从指定权重加载",
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
        cfg.EPOCHS = 1
        cfg.RUN_NAME = "smoke_toy"
        cfg.CKPT_DIR = f"./checkpoints/{cfg.RUN_NAME}"
        if args.max_train_batches is None:
            args.max_train_batches = 2
        if args.max_val_batches is None:
            args.max_val_batches = 1

    resume_from = None
    if args.resume:
        resume_from = args.resume if args.resume != "auto" else os.path.join(
            cfg.CKPT_DIR, "best.pt"
        )
        if not os.path.exists(resume_from):
            raise FileNotFoundError(f"续训权重不存在: {resume_from}")
        # 换 RUN_NAME 避免覆盖原实验的 best.pt / loss_history / train.log
        cfg.RUN_NAME = cfg.RUN_NAME + "_resume"
        cfg.CKPT_DIR = os.path.join("./checkpoints", cfg.RUN_NAME)
        print(f"续训模式: 权重={resume_from} | 新目录={cfg.CKPT_DIR}")

    main(args.max_train_batches, args.max_val_batches, resume_from)
