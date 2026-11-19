# configs/config.py —— 全项目超参唯一真源
# 其它文件一律 `from configs import config as cfg` 后读 cfg.XXX，禁止就地硬编码数字
# 每个常量后注明"为什么是这个值"；这份注释是写给三个月后的你，不用翻聊天记录

import torch  # 仅用于分区三的 DEVICE 自动判断

# ============================================================
# 一、模型结构（决定参数量与张量形状）
# ============================================================
IN_CHANNELS = (
    3  # Pet 转 RGB，toy 也生成 3 通道；长度必须与 NORM_MEAN/STD 一致（见末尾 assert）
)
OUT_CHANNELS = 1  # 二值分割只出 1 个 logit 通道，配 BCEWithLogitsLoss
BASE_CHANNELS = 64  # 论文首层通道数，冒烟不降；通道序列由它派生
NUM_DOWNS = 4  # 4 次下采样 → MIN_SIZE = 16；改它要同步改模型层数
BILINEAR = False  # 原论文用转置卷积；改 True 换成双线性插值，参数更少
PADDING_MODE = "same"  # padding=1 保持等大输出，训练/评估形状直接对齐；"valid" 为论文原版，需中心裁剪
USE_BATCH_NORM = False  # 原论文无 BN；改 True 可对齐 milesial/Pytorch-UNet 的默认行为

# ============================================================
# 二、数据
# ============================================================
DATASET = (
    "oxford_pet"  # 默认 Pet 自动下载；"toy" 无网络冒烟；"folder" 自定义 images/masks
)
DATA_ROOT = "./data"  # 相对项目根，对应 .gitignore 的 data/*
IMAGE_SIZE = (
    256  # 必须是 16 的倍数（4 次下采样）；本机冒烟由 train.py 运行时覆盖为 64/128
)
BATCH_SIZE = 16  # 服务器 5080 的量级；本机 4GB 冒烟靠运行时参数覆盖，不改这里
NUM_WORKERS = 4  # Linux 服务器用 4；Windows 多进程 spawn 需 __main__ 保护，本机覆盖为 0
VAL_RATIO = 0.2  # 从 trainval 切 20% 验证：val 估计更稳，代价是训练样本变少
NORM_MEAN = (
    0.485,
    0.456,
    0.406,
)  # 借用 ImageNet 统计，非 Pet 实测；只作用于图像，掩膜不归一化
NORM_STD = (
    0.229,
    0.224,
    0.225,
)  # 同上；元组长度必须 == IN_CHANNELS，写单值 Normalize 会报错
# TODO 待你确认：8 位灰度掩膜的二值化阈值应是 0~255 量级；0.5 是概率阈值 THRESHOLD 的量级
MASK_THRESHOLD = 127  # folder/toy 掩膜二值化用；Pet 的 trimap 走 ==1 分支，不用这个
TOY_SIZE = 48
# ============================================================
# 三、训练
# ============================================================
EPOCHS = 20  # 演示轮数；5080 可拉到 50+，配合早停不会白跑
LR = 1e-3  # Adam 常用起点；CNN 比 ViT 耐受大 lr（ViT 项目用 3e-4）
WEIGHT_DECAY = 1e-5  # 轻微 L2 正则，抑制过拟合
OPTIMIZER = "adam"  # "adam" | "sgd"；sgd 分支对应原论文动量设置
MOMENTUM = 0.9  # 仅 OPTIMIZER="sgd" 时生效，Adam 下被忽略
LOSS = "bce_dice"  # 逐像素 BCE（收敛稳）+ 集合级 Dice（盯住小前景），互补
BCE_WEIGHT = 1.0  # 两项等权起点；前景极小时可试 DICE_WEIGHT=2.0
DICE_WEIGHT = 1.0  # 同上；这是实验旋钮，不是论文定值
GRAD_CLIP = (
    1.0  # 梯度 L2 范数上限，只缩长度不改方向；clip 必须夹在 backward 与 step 之间
)
THRESHOLD = 0.5  # sigmoid 后判前景；只用于指标与推理，不参与 loss
PATIENCE = 5  # 连续 5 轮 val loss 不创新低就早停
SEED = 42  # random / numpy / torch 三处都要设，切分与初始化才可复现
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 存字符串不存 torch.device，config 保持纯数据可打印

# ============================================================
# 四、路径
# ============================================================
RUN_NAME = "unet_pet_s256_b64"  # 固定名不用时间戳——训练与推理是两个进程，时间戳目录会对不上；s256=尺寸，b64=BASE_CHANNELS
# TODO 待你确认：项目 README 与 .gitignore 约定的是 ./checkpoints/{RUN_NAME}，此处 ./ckpt/ 与之不一致
CKPT_DIR = f"./checkpoints/{RUN_NAME}"  # 由 RUN_NAME 派生，禁止另写一遍路径

# ============================================================
# 派生量（由上面常量算出，勿手改）
# ============================================================
DOWNS = NUM_DOWNS  # 下采样次数，供外部读取
CHANNELS = [BASE_CHANNELS * (2**i) for i in range(NUM_DOWNS + 1)]  # 64→128→256→512→1024
MIN_SIZE = 2**NUM_DOWNS  # 16：输入边长必须整除它

# ============================================================
# 硬约束（import 时就炸，别等训练到一半才发现）
# ============================================================
# 军规：assert 要带消息（如 f"IMAGE_SIZE({IMAGE_SIZE}) 不能被 {MIN_SIZE} 整除"），否则三个月后看不懂
# 另：assert 只校验写死的常量，不用它校验命令行/外部输入（python -O 会整段跳过）
assert IMAGE_SIZE % MIN_SIZE == 0, (
    f"IMAGE_SIZE({IMAGE_SIZE}) 不能被 MIN_SIZE({MIN_SIZE}) 整除，4 次下采样后边长无法对齐"
)
assert PADDING_MODE in ["same", "valid"], (
    f"PADDING_MODE 只能是 same/valid，收到 {PADDING_MODE!r}"
)
assert DATASET in ["oxford_pet", "toy", "folder"], (
    f"DATASET 只能是 oxford_pet/toy/folder，收到 {DATASET!r}"
)
assert len(NORM_MEAN) == IN_CHANNELS, (
    f"NORM_MEAN 长度 {len(NORM_MEAN)} != IN_CHANNELS({IN_CHANNELS})，Normalize 会形状不匹配"
)
assert len(NORM_STD) == IN_CHANNELS, (
    f"NORM_STD 长度 {len(NORM_STD)} != IN_CHANNELS({IN_CHANNELS})，Normalize 会形状不匹配"
)
