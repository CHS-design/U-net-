"""predict.py — 单张 / 目录推理（阶段 6 实现）

职责：
    1. 加载 best.pt 并切到 eval 模式
    2. 读单张图或目录里的图，做与训练一致的预处理
    3. sigmoid → 阈值 0.5 → 二值掩膜，保存为 PNG
    4. 输出原图 / 掩膜 / 叠加图，便于肉眼检查推理效果

状态：已实现单图/目录推理、掩膜与红色叠加图保存。
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from configs import config as cfg
from evaluate import load_model
from utils.data import build_image_transform

def collect_images(input_path):
    """返回待预测图片的 Path 列表"""
    # TODO 1: p = Path(input_path)
    # TODO 2: if p.is_file(): return [p]
    # TODO 3: if p.is_dir(): 过滤后缀 .jpg/.jpeg/.png/.bmp/.tif/.tiff，排序后返回
    # TODO 4: 否则 raise FileNotFoundError
    p = Path(input_path)
    if p.is_file():
        return [p]
    if p.is_dir():
        suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        images = sorted(
            q for q in p.iterdir() if q.is_file() and q.suffix.lower() in suffixes
        )
        if not images:
            raise FileNotFoundError(f"目录下没有图片: {input_path}")
        return images
    raise FileNotFoundError(f"输入不存在: {input_path}")


def predict_one(model, transform, image_path, output_dir, device):
    """对单张图推理，保存 mask 与 overlay"""
    # TODO 5: image = Image.open(image_path).convert("RGB")
    # TODO 6: orig_w, orig_h = image.size
    # TODO 7: x = transform(image).unsqueeze(0).to(device)
    # TODO 8: with torch.no_grad(): logits = model(x)
    # TODO 9: probs = torch.sigmoid(logits)
    # TODO 10: mask = (probs > cfg.THRESHOLD).float()
    # TODO 11: mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    # TODO 12: mask_img = Image.fromarray(mask_np).resize((orig_w, orig_h), Image.NEAREST)
    # TODO 13: 保存 output_dir / f"{image_path.stem}_mask.png"
    # TODO 14: 原图数组 copy，掩膜为真的位置涂红，保存 overlay
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = torch.sigmoid(logits)
    mask = (probs > cfg.THRESHOLD).float()
    mask_np = (mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
    mask_img = Image.fromarray(mask_np).resize((orig_w, orig_h), Image.NEAREST)
    mask_img.save(output_dir / f"{image_path.stem}_mask.png")
    base = np.array(image).astype(np.float32)
    foreground = np.array(mask_img) > 127
    base[foreground] = 0.5 * base[foreground] + 0.5 * np.array([255, 0, 0], dtype=np.float32)
    overlay = Image.fromarray(base.astype(np.uint8))
    overlay.save(output_dir / f"{image_path.stem}_overlay.png")


def main(input_path, output_dir, run_name=None):
    run_name = run_name or cfg.RUN_NAME
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt = Path("./checkpoints") / run_name / "best.pt"
    model = load_model(str(ckpt), cfg.DEVICE)
    transform = build_image_transform()
    for p in collect_images(input_path):
        predict_one(model, transform, p, output_dir, cfg.DEVICE)
        print(f"processed: {p.name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="单张图 / 图目录")
    parser.add_argument("output_dir", help="输出目录")
    parser.add_argument(
        "--run-name", default=None, help="训练名，默认 cfg.RUN_NAME"
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="使用 64×64 预处理，默认指向 smoke_toy 权重",
    )
    args = parser.parse_args()

    if args.smoke:
        cfg.IMAGE_SIZE = 64
        if args.run_name is None:
            args.run_name = "smoke_toy"

    main(args.input_path, args.output_dir, args.run_name)

