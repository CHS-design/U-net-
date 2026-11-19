# models/ — 模型零件（一零件一文件）

## 作用

U-Net 的全部结构。每个零件独立成文件，可单独实例化、单独喂玩具张量验证。零件过三关（形状 / dtype / 数值）后才允许总装。

## 将要有的文件

| 文件 | 零件 | 输入 → 输出 | 要点 |
|---|---|---|---|
| `double_conv.py` | DoubleConv | `(B,Cin,H,W)` → `(B,Cout,H,W)` | 两次 Conv3×3 + ReLU；可选 BN；padding 由 `PADDING_MODE` 决定 |
| `down.py` | Down | `(B,C,H,W)` → `(B,2C,H/2,W/2)` | MaxPool2d(2) + DoubleConv |
| `up.py` | Up | `(B,C,H,W)` + skip → `(B,C/2,2H,2W)` | ConvTranspose2d(2,2) 或 bilinear；拼 skip 后再 DoubleConv；valid 模式需中心裁剪 |
| `out_conv.py` | OutConv | `(B,64,H,W)` → `(B,1,H,W)` | 1×1 卷积出 logits，不加 sigmoid |
| `unet.py` | UNet 总装 | `(B,3,H,W)` → `(B,1,H,W)` | 4 次下采样 + 瓶颈 + 4 次上采样 + 跳跃拼接 |

## 规矩

- 不写训练逻辑：loss / optimizer / DataLoader 一律不出现在此目录。
- 每个文件可以在 `__main__` 放本零件的自检代码（喂玩具张量打印形状）。
- 禁用现成分割模型；允许 `nn.Conv2d`、`nn.ConvTranspose2d`、`nn.MaxPool2d`、`nn.ReLU` 等基础算子。
- 网络内部不硬编码通道数，全部从 `cfg.CHANNELS` 等派生量取。
