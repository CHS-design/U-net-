# data/ — 数据集落盘目录（不入库）

## 作用

存放三类数据来源。本目录**只放数据**，不放代码、不放模型权重。

## 将要有的内容

| 来源 | 触发方式 | 落盘位置 / 说明 |
|---|---|---|
| `toy` | 配置 `DATASET = "toy"` | 运行时合成，用于无网络冒烟；不依赖任何外部文件 |
| `oxford_pet`（默认） | torchvision `datasets.OxfordIIITPet(..., download=True)` | 自动下载到 `data/oxford-iiit-pet/`，首次需联网（约 800MB） |
| `folder` | 你手动放 `images/` 与 `masks/` | 约定放 `data/raw/custom/`，文件名主干需一一对应 |

## 掩膜约定

- Oxford Pet 的 trimap：值为 `1` 的像素是前景（宠物），`2`（背景）与 `3`（边界）都算背景。
- `folder` / `toy` 掩膜：像素值 `> 127` 视为前景，统一转成 `{0, 1}`。
- 送入网络的图像为 `float32` 张量 `[3, H, W]`（值域约 0~1），掩膜为 `float32` 张量 `[1, H, W]`（值域 `{0, 1}`）。
- 图像与掩膜必须同步做缩放/翻转；掩膜用最近邻插值，防止插值污染标签。

## 规矩与验收

- 数据文件不入库：`.gitignore` 用 `data/*` + `!data/README.md`，README 保留。
- 数据不手改；需要重下就删掉对应子目录再跑一次。
- 验收：实例化 Dataset 取单样本，打印图像与掩膜的 shape/dtype，二者空间尺寸必须一致。
