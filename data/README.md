# data/ — 数据集落盘目录（不入库）

## 作用

存放四类数据来源。本目录**只放数据**，不放代码、不放模型权重。

## 将要有的内容

| 来源 | 触发方式 | 落盘位置 / 说明 |
|---|---|---|
| `toy` | 配置 `DATASET = "toy"` | 运行时合成，用于无网络冒烟；不依赖任何外部文件 |
| `oxford_pet`（可选） | 切回 `DATASET="oxford_pet"` + `TASK="binary"` 等配置 | 自动下载到 `data/oxford-iiit-pet/`，首次需联网（约 800MB） |
| `folder` | 你手动放 `images/` 与 `masks/` | 约定放 `data/raw/custom/`，文件名主干需一一对应 |
| `plantseg`（当前默认） | 已落盘 | `plantsegv3/`：11458 张，`images/{train,val,test}` + `annotations/{train,val,test}`，发行版自带划分 |

## 掩膜约定

- Oxford Pet 的 trimap：值为 `1` 的像素是前景（宠物），`2`（背景）与 `3`（边界）都算背景。
- `folder` / `toy` 掩膜：像素值 `> 127` 视为前景，统一转成 `{0, 1}`。
- PlantSeg 掩膜：类别索引图，`0` 为背景，`1~115` 为病害类别（`89` = soybean rust，对应 CSV 的 Index 88）。多类协议保留原值作 class id；只有二值分支才用 `> 0` 转成"任意病害 vs 背景"，不要套 `> 127`。
- PlantSeg v3 的发行版元数据有两处已知问题，已由 `scripts/prepare_plantseg.py` 处理：CSV 缺 88 号、118 张 soybean rust 被写成 `bean rust / 13`；另有 16 张掩膜值与 CSV 不符。训练只读掩膜，类名以 `configs/plantseg_classes.json` 为准，16 张修正表在 `configs/plantseg_label_fixes.json`。
- 送入网络的图像为 `float32` 张量 `[3, H, W]`（值域约 0~1）。多类掩膜为 `int64` 张量 `[H, W]`（值域 `0~115`，已去掉通道维以匹配交叉熵）；二值掩膜为 `float32` 张量 `[1, H, W]`（值域 `{0, 1}`）。
- 图像与掩膜必须同步做缩放/翻转；掩膜用最近邻插值，防止插值污染标签。

## PlantSeg v3 的修正产物

- `metadata_fixed.csv`：修正后的元数据副本（118 条 soybean rust 归位到 88 号，并补 `License` 列）；原始 `Metadatav2.csv` 不动。
- `configs/plantseg_classes.json`：掩膜值 0~115 → 作物/病名，唯一类名表。
- `configs/plantseg_label_fixes.json`：16 张错标掩膜的加载时修正表（`from → to`），不改磁盘文件。
- 校验：`python scripts/prepare_plantseg.py --check`，与当前数据不一致时退出码为 1。

## 许可与合规

- PlantSeg 官方发布在 Zenodo（DOI `10.5281/zenodo.13293891`），许可为 **CC BY-NC-ND 4.0**：署名、非商业、禁止发布衍生版本。
- 图像来自 Google/Bing/Baidu/Reddit/Flickr 等第三方站点，`URL` 列仅用于溯源，不转移著作权；本目录数据仅限非商业研究，不要再分发。
- v3 的 `Metadatav2.csv` 没有 `License` 列；`metadata_fixed.csv` 按文件名从 v7 借了 `License`（CC-BY-NC / CC0），未命中的填 `unknown`。

## 规矩与验收

- 数据文件不入库：`.gitignore` 用 `data/*` + `!data/README.md`，README 保留。
- 数据不手改；需要重下就删掉对应子目录再跑一次。
- 验收：实例化 Dataset 取单样本，打印图像与掩膜的 shape/dtype，二者空间尺寸必须一致。
- PlantSeg 额外验收：`python scripts/prepare_plantseg.py --check` 通过；`build_datasets()` 返回 7916 / 1247 / 2295。
