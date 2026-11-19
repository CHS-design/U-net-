"""plot_curve.py — 由 loss_history.json 独立画图（阶段 5/6 实现）

职责：
    1. 读 checkpoints/{RUN_NAME}/loss_history.json
    2. 画 train/val loss 与 Dice/IoU 随 epoch 变化的曲线
    3. 保存 loss_curve.png 到同一实验目录
    4. 与训练解耦：随时可重跑，不依赖训练进程

状态：空壳，等阶段 5/6 亲手实现。
"""
