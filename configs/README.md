# configs/ — 超参唯一真源

## 作用

全项目所有超参数集中在这里。其它文件只读 `configs.config`，不在各自文件里写数字。

## 将要有的文件

| 文件 | 职责 | 要点 |
|---|---|---|
| `config.py` | 唯一真源 | 四分区（模型结构 / 数据 / 训练 / 路径）+ 派生量 + assert；阶段 1 由你亲手写 |
| `plantseg_classes.json` | PlantSeg 类名表 | 掩膜值 0~115 → 作物/病名；由 `scripts/prepare_plantseg.py` 生成，勿手改 |
| `plantseg_label_fixes.json` | PlantSeg 错标修正 | 16 张掩膜值的加载时 `from → to` 映射；由同一脚本生成 |

## 规矩

- 其它文件一律 `from configs import config as cfg`，然后 `cfg.XXX`；禁止就地写死数字。
- 每个常量注释里要写"为什么是这个值"，以及调大调小的后果——这份注释是写给三个月后的你。
- 硬约束用 `assert` 在 import 时就炸，别等训练到一半才发现。
- 派生量（如通道序列、CKPT_DIR）由常量算出，禁止手抄第二遍。

## 验收

阶段 1 写完后，在根目录运行：

```powershell
python -c "from configs import config as cfg; print(cfg.CHANNELS, cfg.DOWNS, cfg.CKPT_DIR)"
```

预期能看到通道序列 `[64, 128, 256, 512, 1024]`、下采样次数 `4`、以及由 RUN_NAME 派生的实验目录。
