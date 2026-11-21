# U-Net 复现（PyTorch）

从零复现 Ronneberger 等人 2015 年的 **U-Net**（MICCAI 2015），完成二值语义分割的训练、评估与推理链路。

- 论文：*U-Net: Convolutional Networks for Biomedical Image Segmentation*
- 任务：输入图像，输出同尺寸的前景/背景 logits
- 默认数据：Oxford-IIIT Pet（trimap `==1` 为前景）
- 结构：4 次下采样，通道 `64 → 128 → 256 → 512 → 1024`，4 次上采样 + 跳跃拼接
- 当前状态：**本地 toy 冒烟通过；全量训练待 5080 执行**

---

## 1. 项目结构

```
U-net复现/
├── configs/
│   ├── config.py            # 超参唯一真源（四分区 + 派生量 + assert）
│   └── README.md
├── data/
│   ├── README.md
│   └── oxford-iiit-pet/     # 数据（不入 git）
├── models/
│   ├── double_conv.py       # 两次 3×3 卷积 + 可选 BN + ReLU
│   ├── down.py              # MaxPool + DoubleConv
│   ├── up.py                # 上采样 + 跳跃拼接 + DoubleConv
│   ├── out_conv.py          # 1×1 卷积输出 logits
│   ├── unet.py              # U-Net 总装
│   └── README.md
├── utils/
│   ├── data.py              # Dataset / transform / DataLoader
│   ├── losses.py            # BCE + Dice
│   ├── metrics.py           # Dice / IoU / 像素准确率 + MetricTracker
│   └── README.md
├── checkpoints/             # 训练产物（不入 git）
├── train.py                 # 训练 + 验证 + 早停 + 保存
├── evaluate.py              # 测试集评估
├── predict.py               # 单图/目录推理
├── plot_curve.py            # 由 loss_history.json 画曲线
├── requirements.txt
└── README.md
```

数据、权重、曲线不进入版本控制；目录约定见各子目录的 `README.md`。

---

## 2. 环境与安装

- Python ≥ 3.10（开发环境 3.12）
- PyTorch ≥ 2.0；开发环境 `torch 2.14.0+cu126` / `torchvision 0.29.0`
- 本机：RTX 3050 Laptop 4GB，只跑 `--smoke`
- 服务器：RTX 5080 16GB，跑全量

先按 CUDA 版本安装 torch 全家，再装其余依赖：

```powershell
# 本机 3050（CUDA 12.6）
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu126

# 服务器 5080（CUDA 13.2）
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu132

pip install -r requirements.txt
```

---

## 3. 数据

### 3.1 Oxford-IIIT Pet（默认）

首次运行会自动下载到 `data/oxford-iiit-pet/`（约 800MB）。也可以手动放置：

```
data/oxford-iiit-pet/
├── images/          # 7390 张 jpg
└── annotations/
    ├── trimaps/     # 7390 张 png，值 1/2/3
    ├── trainval.txt # 3680 行
    └── test.txt     # 3669 行
```

- 官方划分只提供 `trainval` + `test`；本项目用 `VAL_RATIO=0.2` 从 `trainval` 再切验证集。
- trimap 规则：`== 1` 为前景，`2`（背景）与 `3`（边界）都算背景。
- 目录里可能带 macOS 的 `._xxx.png` 元数据，torchvision 按 ID 取文件，不会误用。

### 3.2 toy（冒烟）

`DATASET="toy"` 时程序生成随机图 + 中心方块掩膜，不读磁盘、不需要网络，用于验证代码链路。

### 3.3 folder（自定义）

```
data/raw/custom/
├── images/   000.png 001.png ...
└── masks/    000.png 001.png ...
```

文件名主干必须一致；掩膜按 8 位灰度处理，`> 127` 为前景。当前 `build_datasets()` 的 folder 分支尚未接入训练入口，属于后续扩展。

---

## 4. 配置

所有超参在 `configs/config.py`，其它文件只读 `cfg.XXX`。

当前关键值：

| 项 | 值 | 说明 |
|---|---|---|
| `DATASET` | `"oxford_pet"` | 默认 Pet；可切 `"toy"` |
| `IMAGE_SIZE` | `256` | 必须是 16 的倍数 |
| `BATCH_SIZE` | `16` | 5080 全量；本机冒烟覆盖为 2 |
| `NUM_WORKERS` | `4` | Linux 服务器；Windows 冒烟覆盖为 0 |
| `BASE_CHANNELS` | `64` | 论文首层通道，冒烟不降 |
| `NUM_DOWNS` | `4` | 4 次下采样，通道到 1024 |
| `PADDING_MODE` | `"same"` | 输出等大；`valid` 暂未实现 |
| `BILINEAR` | `True` | 双线性上采样；`False` 为论文转置卷积 |
| `USE_BATCH_NORM` | `True` | 当前开启；论文原版无 BN |
| `LOSS` | `"bce_dice"` | BCE + Dice |
| `EPOCHS` / `LR` | `20` / `1e-3` | Adam |
| `PATIENCE` | `5` | 连续 5 轮 val loss 不降则早停 |

**参数量对账**（输入 3 通道）：

| 配置 | 参数量 |
|---|---|
| 无 BN + 转置卷积（论文默认） | `31,031,745` |
| 有 BN + 双线性（当前 config） | `28,948,673` |

形状链（256 输入）：`256 → 128 → 64 → 32 → 16 → 32 → 64 → 128 → 256`。

---

## 5. 运行

### 5.1 本地冒烟（3050 4GB）

```powershell
python train.py --smoke
python plot_curve.py --run-name smoke_toy
python evaluate.py --smoke --max-batches 1
python predict.py data/oxford-iiit-pet/images/Abyssinian_1.jpg predictions_smoke --smoke
```

`--smoke` 自动切到：

```
DATASET=toy, BATCH_SIZE=2, IMAGE_SIZE=64, NUM_WORKERS=0,
EPOCHS=1, 训练 2 个 batch / 验证 1 个 batch,
RUN_NAME=smoke_toy
```

产物在 `checkpoints/smoke_toy/`，不会覆盖真实实验。

### 5.2 5080 全量训练

先把项目和 `data/oxford-iiit-pet/` 传到服务器（或让 torchvision 自动下载），然后：

```bash
cd ~/U-net复现
python -m pip install -r requirements.txt

# 先用冒烟确认服务器环境
python train.py --smoke

# 全量训练（建议放 tmux/nohup，避免断连）
tmux new -s unet
python train.py 2>&1 | tee train.log
```

训练结束后：

```bash
python plot_curve.py
python evaluate.py
python predict.py data/oxford-iiit-pet/images/Abyssinian_1.jpg predictions
```

如果显存不足：**先降 `BATCH_SIZE`（16 → 8 → 4），不要降 `BASE_CHANNELS`、不要砍层数**。改 batch 后同步改 `RUN_NAME`，避免不同配置的权重混在同一目录。

---

## 6. 产物

```
checkpoints/{RUN_NAME}/
├── best.pt            # 验证集 val loss 最优的 state_dict
├── loss_history.json  # 每 epoch 的 loss 与 train/val 指标
└── loss_curve.png     # 由 plot_curve.py 生成
```

`loss_history.json` 结构：

```json
[
  {
    "epoch": 1,
    "train_loss": 1.23,
    "val_loss": 1.30,
    "train_metrics": {"dice": 0.4, "iou": 0.25, "acc": 0.7},
    "val_metrics": {"dice": 0.35, "iou": 0.21, "acc": 0.68}
  }
]
```

`predict.py` 输出：

- `xxx_mask.png`：0/255 黑白掩膜；
- `xxx_overlay.png`：前景半透明红色叠加。

---

## 7. 关键约定（军规）

1. **config 唯一真源**：模型结构、数据尺寸、训练超参都从 `cfg` 读，不在别处写死。
2. **test 集只在最后用一次**：训练期用 val 选最优、早停。
3. **掩膜阈值分两侧**：
   - 标注灰度图：`> MASK_THRESHOLD`（127）；
   - 模型概率：`> THRESHOLD`（0.5）。
4. **Pet 用 `==1`，folder/toy 用 `>127`**：trimap 是类别编号，不是灰度强度。
5. **掩膜 Resize 用 NEAREST**：双线性会插出灰边，污染标签。
6. **`BCEWithLogitsLoss` 吃 logits**：不要先 sigmoid 再喂 BCE。
7. **固定 `RUN_NAME` 不用时间戳**：训练与推理是两个进程，时间戳目录会对不上。
8. **权重只存 `state_dict`**：跨环境加载更稳。

---

## 8. 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| 本机跑全量 OOM | 用 `--smoke`；或降 `BATCH_SIZE`，不要改通道序列 |
| mask 全 0、Dice 恒 0 | 检查 Pet 是否误用 `>127`；检查 `IMAGE_SIZE` 是否被 16 整除 |
| `Sizes of tensors must match` | 跳跃连接接错；same 模式下检查输入尺寸是否 16 的倍数 |
| `FileNotFoundError: best.pt` | `RUN_NAME` 与训练时不一致，或还没训练完 |
| Windows DataLoader 报 spawn 错误 | `NUM_WORKERS=0`，或用 `--smoke`；入口已有 `if __name__ == "__main__"` |
| `json` 报 Tensor 不可序列化 | history 里混入了张量；指标先 `.item()` 转 float |

---

## 9. 下一步实验

1. **数据增强**：随机水平/垂直翻转（图像与掩膜同步），再对比 val Dice。
2. **学习率调度**：`ReduceLROnPlateau` 监控 val Dice，或 cosine + warmup。
3. **valid 模式**：补 `_center_crop`，复现论文 572→388 的尺寸变化。
4. **ISBI 2012**：原论文电镜数据，30 张 512×512，需要弹性形变增强。
5. **消融**：BCE vs Dice vs BCE+Dice；BN on/off；转置卷积 vs 双线性。
6. **多分类/多通道**：把 `OUT_CHANNELS` 改大，损失换 `CrossEntropyLoss`。

---

## 10. 复现目标与边界

- 本项目的目标是**结构、数据管线、训练/评估流程**可运行、可解释、可复现，不追求复刻论文的具体 Dice 数字。
- 当前 config 是 `BN + 双线性` 变体；若要与论文默认对齐，把 `BILINEAR`、`USE_BATCH_NORM` 改成 `False`，并重新对账 `31,031,745`。
- 所有核心结构（DoubleConv / Down / Up / UNet）与损失数学（Dice）均手写；只使用 PyTorch 的基础层与数据工具。
