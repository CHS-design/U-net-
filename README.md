# U-Net 复现（PyTorch）

从零复现 Ronneberger 等人 2015 年的 **U-Net**（MICCAI 2015），完成语义分割的训练、评估与推理链路。

- 论文：*U-Net: Convolutional Networks for Biomedical Image Segmentation*
- 任务：二值前景/背景分割（Pet / toy / folder）或 116 类语义分割（PlantSeg 论文协议）
- 默认数据：PlantSeg v7（`0`=背景，`1~115`=病害类别）
- 结构：4 次下采样，通道 `64 → 128 → 256 → 512 → 1024`，4 次上采样 + 跳跃拼接
- 当前状态：**多类链路（CE + mIoU/mAcc）已接通；全量训练待 5080 执行**

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
├── scripts/                 # 可执行的阶段入口脚本
│   ├── train.py             # 训练 + 验证 + 早停 + 保存
│   ├── evaluate.py          # 测试集评估
│   ├── predict.py           # 单图/目录推理
│   └── plot_curve.py        # 由 loss_history.json 画曲线
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

### 3.4 PlantSeg（当前默认，2024 论文版）

训练集已在 `data/plantseg_v7/plantseg/`，共 7774 张带掩膜图像（图像为 jpg，掩膜为 png）：

```
data/plantseg_v7/plantseg/
├── images/       train/(5367)  val/(846)  test/(1561)
├── annotations/  train/(5367)  val/(846)  test/(1561)
├── Metadata.csv         # Name / Disease / Index / Split 等元信息
└── annotation_{train,val,test}.json   # COCO 多边形标注，本管线不用
```

- 图像与掩膜按同名主干配对（如 `images/train/apple_black_rot_1.jpg` ↔ `annotations/train/apple_black_rot_1.png`），已核对全部分支一一对应。
- 掩膜是**类别索引图**：`0` 为背景，前景像素值 = `Metadata.csv` 的 `Index + 1`（1~115）。多类协议直接把原值当 class id，配 `OUT_CHANNELS=116` 的交叉熵；只有 `TASK="binary"` 时才用 `raw_mask > 0` 压成"任意病害 vs 背景"。**不能沿用 Pet 的 `==1` 或 folder 的 `>127`。**
- 论文协议：SGD（lr `1e-3`、momentum `0.9`、weight decay `5e-4`）、batch size `16`、cross-entropy loss、mIoU/mAcc 逐类平均。本项目额外用官方 `val/` 做早停，`test/` 只在最后评估一次。
- 图像分辨率不固定，管线统一 resize 到 `IMAGE_SIZE`；掩膜用最近邻插值，类别值不被插值污染。
- 直接用官方 train/val/test 划分，不做二次切分；`VAL_RATIO` 只对 Pet 生效。论文原文是每类 80/20 切 train/test、没有 val，因此指标口径对齐但数字不可直接比较。

---

## 4. 配置

所有超参在 `configs/config.py`，其它文件只读 `cfg.XXX`。

当前关键值：

| 项 | 值 | 说明 |
|---|---|---|
| `DATASET` | `"plantseg"` | 当前默认 PlantSeg；可切 `"oxford_pet"` / `"toy"` / `"folder"` |
| `TASK` | `"multiclass"` | `"multiclass"`=论文协议；`"binary"`=旧 Dice/IoU 链路 |
| `NUM_CLASSES` / `OUT_CHANNELS` | `116` / `116` | 0 背景 + 115 病害；二值任务改为 1 |
| `PLANTSEG_ROOT` | `"./data/plantseg_v7/plantseg"` | PlantSeg 数据根目录 |
| `PLANTSEG_MASK_THRESHOLD` | `0` | 仅二值分支用：前景 > 0；多类分支保留原索引 |
| `IMAGE_SIZE` | `256` | 必须是 16 的倍数 |
| `BATCH_SIZE` | `16` | 论文值；本机冒烟覆盖为 2 |
| `EPOCHS` | `100` | 配合 `PATIENCE=5` 早停 |
| `NUM_WORKERS` | `4` | Linux 服务器；Windows 冒烟覆盖为 0 |
| `BASE_CHANNELS` | `64` | 论文首层通道，冒烟不降 |
| `NUM_DOWNS` | `4` | 4 次下采样，通道到 1024 |
| `PADDING_MODE` | `"same"` | 输出等大；`valid` 暂未实现 |
| `BILINEAR` | `True` | 双线性上采样；`False` 为论文转置卷积 |
| `USE_BATCH_NORM` | `True` | 当前开启；论文原版无 BN |
| `LOSS` | `"ce"` | 论文用交叉熵；二值任务用 `"bce_dice"` |
| `OPTIMIZER` / `LR` / `MOMENTUM` / `WEIGHT_DECAY` | `"sgd"` / `1e-3` / `0.9` / `0.0005` | 论文给出的优化器设置 |
| `PATIENCE` | `5` | 连续 5 轮 val 选优指标（多类 mIoU / 二值 Dice）不升则早停 |
| `RUN_NAME` | `"unet_plantseg_mc_s256_b64_bs16_bn_bilinear_ce_sgd"` | 换数据集/超参必须同步改名 |

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
python scripts/train.py --smoke
python scripts/plot_curve.py --run-name smoke_toy
python scripts/evaluate.py --smoke --max-batches 1
python scripts/predict.py data/oxford-iiit-pet/images/Abyssinian_1.jpg predictions_smoke --smoke
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
python scripts/train.py --smoke

# 全量训练（建议放 tmux/nohup，避免断连）
tmux new -s unet
python scripts/train.py 2>&1 | tee train.log
```

训练结束后：

```bash
python scripts/plot_curve.py
python scripts/evaluate.py
python scripts/predict.py data/oxford-iiit-pet/images/Abyssinian_1.jpg predictions
```

如果显存不足：**先降 `BATCH_SIZE`（16 → 8 → 4），不要降 `BASE_CHANNELS`、不要砍层数**。改 batch 后同步改 `RUN_NAME`，避免不同配置的权重混在同一目录。

### 5.3 PlantSeg 训练 / 评估 / 推理

数据已就位，`configs/config.py` 默认 `DATASET="plantseg"`、`TASK="multiclass"`、`LOSS="ce"`、实验名 `unet_plantseg_mc_...`，无需改代码即可按论文协议运行：

```bash
cd /home/chs123/code/unet

# 1) 先验数据管线：掩膜必须是 (H,W) int64 类别索引，最大值 <=115
python -c "from utils.data import build_datasets; tr,va,te=build_datasets(); print(len(tr),len(va),len(te)); x,y=tr[0]; print(x.shape,x.dtype,y.shape,y.dtype,int(y.max()))"

# 2) 训练（val mIoU 选优 + 早停；建议 tmux/nohup）
python scripts/train.py 2>&1 | tee train_plantseg_mc.log

# 3) 曲线与测试集评估（打印 miou / macc）
python scripts/plot_curve.py
python scripts/evaluate.py

# 4) 单图推理：*_mask.png 是类别索引图，*_overlay.png 把非背景涂红
python scripts/predict.py data/plantseg_v7/plantseg/images/test/apple_black_rot_143.jpg predictions_plantseg
```

切换回 Pet 等二值实验时，必须同时改回 `DATASET="oxford_pet"`、`TASK="binary"`、`OUT_CHANNELS=1`、`LOSS="bce_dice"` 和 `RUN_NAME`；`--smoke` 分支已自动做这套切换。只改 `DATASET` 会留下多类损失与二值掩膜的不一致配置。

---

## 6. 产物

```
checkpoints/{RUN_NAME}/
├── best.pt            # 验证集选优指标最优的 state_dict（多类 mIoU / 二值 Dice）
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
    "train_metrics": {"miou": 0.40, "macc": 0.70},
    "val_metrics": {"miou": 0.35, "macc": 0.68}
  }
]
```

`predict.py` 输出：

- `xxx_mask.png`：多类任务为类别索引图（0~115，与官方标注同格式），二值任务为 0/255 黑白掩膜；
- `xxx_overlay.png`：前景（多类时任意非背景类）半透明红色叠加。

---

## 7. 关键约定（军规）

1. **config 唯一真源**：模型结构、数据尺寸、训练超参都从 `cfg` 读，不在别处写死。
2. **test 集只在最后用一次**：训练期用 val 选最优、早停。
3. **损失与指标按 `TASK` 分流**：
   - 多类（PlantSeg 论文协议）：cross-entropy 吃 logits + `(B,H,W) int64` 类别索引；指标为逐类 mIoU/mAcc，按 val mIoU 选优；
   - 二值（Pet/toy/folder）：BCE/Dice，模型概率 `> THRESHOLD`（0.5），按 val Dice 选优。
4. **Pet 用 `==1`，folder/toy 用 `>127`，PlantSeg 多类保留 `0~115` 原值**：只有 PlantSeg 的二值分支才用 `>0`，不能按灰度强度混用。
5. **掩膜 Resize 用 NEAREST**：双线性会插出灰边，污染标签。
6. **`CrossEntropyLoss` / `BCEWithLogitsLoss` 都吃 logits**：不要先 sigmoid 再喂损失。
7. **固定 `RUN_NAME` 不用时间戳**：训练与推理是两个进程，时间戳目录会对不上。
8. **权重只存 `state_dict`**：跨环境加载更稳。
9. **`--smoke` 必须切回完整二值配置**：`train.py` / `evaluate.py` / `predict.py` 的冒烟分支都覆盖 `TASK`、`OUT_CHANNELS`、`LOSS`，只改 `DATASET` 会留下不一致配置。

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
| 交叉熵报 target dtype / shape 错 | 多类 target 必须是 `(B,H,W) int64`；检查 `PlantSegSegDataset` 的 `squeeze(0).long()` |
| `--smoke` 报交叉熵错误 | 冒烟分支没同步切回二值；确认 `TASK/OUT_CHANNELS/LOSS` 都被覆盖 |
| 多类指标打印成 `dice/iou/acc` | `TASK` 没切成 `"multiclass"`，或用的是旧权重/旧历史文件 |

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
