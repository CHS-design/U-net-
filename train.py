"""train.py — 训练入口（阶段 5 实现）

职责：
    1. 读 configs/config.py 的超参
    2. 调 utils/data.py 构建 train/val DataLoader
    3. 建 UNet 并 .to(cfg.DEVICE)
    4. 每个 epoch 走五步：zero_grad → forward → loss → backward → step
    5. 验证：model.eval() + torch.no_grad()，统计 val loss / Dice / IoU
    6. val loss 创新低才存 best.pt；patience 到上限早停
    7. 每 epoch 指标写入 checkpoints/{RUN_NAME}/loss_history.json

状态：空壳，等阶段 5 亲手实现。
"""
