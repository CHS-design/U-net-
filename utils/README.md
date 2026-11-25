# utils/ — 工具层（只被 import，不直接运行）

## 作用

放数据管线、损失函数和评估指标。它是"被 models 与入口脚本调用的公共服务"，不定义网络结构。

## 将要有的文件

| 文件 | 导出 | 输入 → 输出 / 要点 |
|---|---|---|
| `data.py` | `build_datasets` / `build_dataloaders` / `load_label_fixes` | 四类数据集、掩膜二值化、同步增强、PlantSeg 标签修正与 EXIF 转正；返回 train/val/test |
| `losses.py` | `build_loss` / BCE+Dice | 输入 logits 与 mask，返回标量 loss；logits 不做 sigmoid（交给 `BCEWithLogitsLoss`） |
| `metrics.py` | `dice` / `iou` / `pixel_acc` / `MetricTracker` | 阈值 0.5 后逐 batch 统计，Tracker 累加后取平均 |

## 规矩

- 模型零件（`nn.Module`）放 `models/`，不放这里；损失写函数即可。
- 所有超参从 `cfg` 取，不在工具函数里写死 batch size、阈值、权重。
- 函数名用 `build_*` 表示"构建"，`MetricTracker` 这种状态对象用类名。
- 验收：玩具张量喂进 loss 与 metric，数值要符合预期（完全一致时 Dice=1）。
