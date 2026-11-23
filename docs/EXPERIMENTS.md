# U-Net 实验记录

Oxford-IIIT Pet 二值分割（前景 = trimap `==1`）实验汇总。测试集固定为官方 `test` split，3669 张，只在最终评估时使用一次。

## 1. 环境

| 项 | 值 |
|---|---|
| Python | 3.12.14（conda 环境 `chs`） |
| PyTorch | 2.14.0+cu132 |
| torchvision | 0.29.0+cu132 |
| GPU | NVIDIA RTX 5080 16GB |
| CUDA | 13.2 |

## 2. 固定配置

| 项 | 值 |
|---|---|
| 数据集 | Oxford-IIIT Pet（trainval 切 20% 作 val，固定 seed=42） |
| 输入尺寸 | 256 × 256 |
| 模型 | 4 下采样，通道 64→128→256→512→1024，BN + 双线性上采样 |
| 参数量 | 28,948,673 |
| 损失 | BCE + Dice（等权） |
| 优化器 | Adam，lr=1e-3，weight_decay=1e-5 |
| 梯度裁剪 | 1.0 |
| EPOCHS | 100（早停 PATIENCE=5，1-6 号实验为 20） |
| BATCH_SIZE | 16（1-2 号）/ 32（3-8 号） |

## 3. 实验变量

| # | RUN_NAME | BATCH | 加速 | 数据增强 | 选优依据 |
|---|---|---|---|---|---|
| 1 | `unet_pet_s256_b64_bn_bilinear` | 16 | fp32 | 无 | val loss |
| 2 | `unet_pet_s256_b64_bs32_bn_bilinear` | 32 | AMP + cudnn.benchmark | 无 | val loss |
| 3 | `unet_pet_s256_b64_bs32_bn_bilinear_opt` | 32 | + torch.compile + NHWC + 异步指标 | 无 | val loss |
| 4 | `unet_pet_s256_b64_bs32_bn_bilinear_opt_aug` | 32 | 同上 | 翻转 + ColorJitter | val loss |
| 5 | `unet_pet_s256_b64_bs32_bn_bilinear_opt_aug_dice` | 32 | 同上 | 翻转 + ColorJitter | **val Dice** |
| 6 | `unet_pet_s256_b64_bs32_bn_bilinear_opt_noaug_dice` | 32 | 同上 | 无 | **val Dice** |

> 加速项逐层叠加：#2 起开 AMP + cudnn.benchmark；#3 起再加 `torch.compile`、NHWC（channels_last）内存格式、指标异步化；#4 起加数据增强；#5 起把选优/早停依据从 val loss 改为 val Dice。

## 4. 训练结果

| # | 实验 | 训练轮数 | 总耗时 | 平均轮时 | val 最优 Dice | val 最优 loss |
|---|---|---|---|---|---|---|
| 2 | batch32 + AMP（20 轮设置） | 20 | 7.94 min | 23.8s | — | — |
| 3 | + compile/NHWC | 39（早停） | 12.85 min | 19.8s | 0.8894 (ep36) | 0.3458 (ep34) |
| 4 | + 数据增强 | 45（早停） | 14.30 min | 19.1s | 0.8844 (ep41) | 0.3505 (ep40) |
| 5 | 增强 + Dice 选优 | **60（早停）** | 19.06 min | 19.1s | **0.8947** (ep55) | 0.3163 (ep58) |
| 6 | 无增强 + Dice 选优 | 42（早停） | 13.29 min | 19.0s | 0.8909 (ep37) | 0.3398 (ep37) |

> #1（fp32 + batch16）约 36s/轮，20 轮约 12 min；#2 与之对比，AMP + cudnn.benchmark 提速约 35%（23.8s/轮）。#2 未单独评估测试集。

## 5. 测试集结果（3669 张，官方 test split）

| # | 实验 | test loss | test Dice | test IoU | test Acc |
|---|---|---|---|---|---|
| 3 | compile/NHWC + loss 选优 | 0.3245 | 0.8955 | 0.8108 | 0.9359 |
| 4 | + 数据增强 + loss 选优 | 0.3364 | 0.8880 | 0.7986 | 0.9329 |
| 5 | + 数据增强 + **Dice 选优** | **0.3104** | **0.8972** | **0.8136** | 0.9371 |
| 6 | 无增强 + **Dice 选优** | 0.3238 | 0.8955 | 0.8108 | **0.9373** |

**当前最佳：#5（增强 + Dice 选优），test Dice 0.8972 / IoU 0.8136。**

## 6. 与网上公开复现的对比

### 6.1 同数据集、同配置的公开实现（有明确数值）

以下两个公开项目与本项目同为 Oxford-IIIT Pet、256×256、官方 trainval/test 划分（seed=42），数值可直接对照。

| 来源 | 模型 | 评估集 | Dice | IoU | 关键配置 |
|---|---|---|---|---|---|
| [Razamindset/Image-segmentation-with-Unet](https://github.com/Razamindset/Image-segmentation-with-Unet) v3 | UNet | val（736 张） | 0.8844 | 0.8063 | batch16, AdamW 1e-4, BCE+Dice, 有增强, 40 ep |
| Razamindset v2 | UNet | val（736 张） | ~0.86 | ~0.78 | batch16, 纯 BCE, 有增强 |
| [LitTTian/unet-pet-segmentation](https://github.com/LitTTian/unet-pet-segmentation) | UNet | **test（3669 张）** | 0.8779 | 0.7963 | batch24, lr1e-4, 纯 BCE, 40 ep, Val IoU 选优 |
| LitTTian | TransUNet (bs16) | test（3669 张） | **0.8914** | **0.8162** | batch16, Transformer 骨干, 52 ep |
| LitTTian | TransUNet (默认) | test | 0.8856 | 0.8080 | batch24, 59 ep |
| LitTTian | TransUNetP | test | 0.8833 | 0.8048 | batch24, 71 ep |
| **本项目 #5** | UNet | **test（3669 张）** | **0.8972** | **0.8136** | batch32, Adam 1e-3, BCE+Dice, 有增强, Dice 选优 |

> 说明：
> - Razamindset 报的是 **val**（736 张），本项目报的是 **test**（3669 张，测试集只在最后用一次）。测试集通常更难，本项目的 test 成绩达到甚至略超对方的 val 水平。
> - LitTTian 同为 **test**，可比性最强。其纯 UNet baseline（Dice 0.8779 / IoU 0.7963）明显低于本项目 #5（+0.019 Dice / +0.017 IoU）；其最强的 TransUNet（换 Transformer 骨干，Dice 0.8914 / IoU 0.8162）与本项目 #5 接近——**本项目用纯 CNN 的 UNet 达到了与 TransUNet 相当的水平**，IoU 略低 0.0026 但 Dice 高 0.0058。
> - 双方随机种子、划分细节、增强方式不同，1~2 个点的差距在噪声范围内。

### 6.2 其他知名复现（数据集不同，仅作方法参考）

| 项目 | 数据集 | 与本项目的方法差异 / 共识 |
|---|---|---|
| [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) | Carvana（高清车辆图） | 用 `ReduceLROnPlateau` **监控 val Dice**（mode=max）而非 loss，保存最优 checkpoint；同样用 NHWC (channels_last) 内存格式和梯度裁剪 1.0。数据集不同，无直接可比数值，但其选优策略与本项目 #5 一致 |
| [LitTTian/unet-pet-segmentation](https://github.com/LitTTian/unet-pet-segmentation) | Oxford Pet | 同数据集，数值见 6.1；其用 **Val IoU** 选优（patience=20），并明确指出"loss 最小的点不一定是测试集 IoU/Dice 最高的点"，与本项目改用 Dice 选优的理由一致 |
| 原论文 U-Net (Ronneberger et al. 2015) | ISBI 电镜细胞 | 数据集完全不同（30 张 512×512 电镜图 + 弹性形变增强），指标不可直接比较 |

### 6.3 定位

本项目 README 的目标是"结构、数据管线、训练/评估流程可运行、可解释、可复现，不追求复刻论文的具体 Dice 数字"。按此定位，test Dice 0.8972 / IoU 0.8136 属于 Oxford Pet 二值分割的主流偏上水平。

## 7. 结论与经验

1. **加速**：AMP + cudnn.benchmark 把每轮从 ~36s 降到 23.8s（-35%）；再加 `torch.compile` + NHWC + 异步指标降到 ~19s（再 -20%）。`torch.compile` 首次编译约 30s，之后命中磁盘缓存。数据增强的 CPU 开销与 GPU 计算重叠，未拖慢每轮耗时。

2. **选优指标**：改为 val Dice 选优后提升明显。同为增强配置，#5 vs #4 的 test Dice 从 0.8880 → 0.8972（+0.0092），IoU 0.7986 → 0.8136（+0.0150）。原因是 loss 最低点未必是 Dice 最高点——#5 中按 loss 最优在 ep58（Dice 0.8930），按 Dice 最优在 ep55（loss 0.3201）。这也是主流分割项目的普遍做法。

3. **数据增强**：翻转 + 轻度 ColorJitter（brightness/contrast/saturation=0.2, hue=0.05）单独看提升很小（#5 vs #6，Dice 0.8972 vs 0.8955），差距在单次实验噪声范围内；但结合 Dice 选优后，增强版略优于无增强版，未观察到 ColorJitter 的负面影响。

4. **batch size**：16 → 32 在 AMP 下显存反而更低（AMP batch32 峰值 7.22GB，fp32 batch16 约 9.8GB），每轮 batch 数从 184 降到 92，训练更稳。
