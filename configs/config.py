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
NUM_CLASSES = 116  # 0=背景 + 115 类病害（掩膜值 1~115）；89=soybean rust（CSV Index 88），原始 CSV 缺这一类
OUT_CHANNELS = NUM_CLASSES  # 每类一个 logit 通道；切回二值实验时改回 1
BASE_CHANNELS = 64  # 论文首层通道数，冒烟不降；通道序列由它派生
NUM_DOWNS = 4  # 4 次下采样 → MIN_SIZE = 16；改它要同步改模型层数
BILINEAR = True  # 原论文用转置卷积；改 True 换成双线性插值，参数更少
PADDING_MODE = "same"  # padding=1 保持等大输出，训练/评估形状直接对齐；"valid" 为论文原版，需中心裁剪
USE_BATCH_NORM = True  # 原论文无 BN；改 True 可对齐 milesial/Pytorch-UNet 的默认行为

# ============================================================
# 二、数据
# ============================================================
DATASET = (
    "plantseg"  # 当前适配 PlantSeg；可选 "oxford_pet"（Pet）、"toy"（冒烟）、"folder"（自定义）
)
TASK = "multiclass"  # "binary" 走原有 Dice/IoU 二值链路；"multiclass" 走论文协议

DATA_ROOT = "./data"  # 相对项目根，对应 .gitignore 的 data/*
PLANTSEG_ROOT = (
    "./data/plantsegv3"  # 11458 张：images/{train,val,test}/*.jpg + annotations/{train,val,test}/*.png
)
PLANTSEG_MASK_THRESHOLD = 0  # 仅二值分支用：>0 判前景；多类分支保留原始类别索引，不走这个阈值
PLANTSEG_CLASS_MAP = (
    "./configs/plantseg_classes.json"  # 掩膜值 0~115 -> 作物/病名；由 prepare_plantseg.py 生成
)
PLANTSEG_LABEL_FIXES = (
    "./configs/plantseg_label_fixes.json"  # 16 张错标掩膜的加载时修正表；文件缺失视为无修正
)
IMAGE_SIZE = (
    512  # 论文用 512；必须是 16 的倍数（4 次下采样）；小病斑在 256 下细节丢失严重
)
BATCH_SIZE = 8  # 512 输入是 256 的 4 倍像素，batch 减半到 8；实测峰值 8.2GB/16GB
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
EPOCHS = 300  # 全量轮数；配合 PATIENCE=20 早停，给 115 类冷启动足够时间
LR = 1e-3  # 论文 SGD 起点；CNN 比 ViT 耐受大 lr（ViT 项目用 3e-4）
WEIGHT_DECAY = 0.0005  # 论文给出的 weight decay，比原 Adam 基线的 1e-5 大一个量级
OPTIMIZER = "sgd"  # "sgd" 对齐论文；"adam" 保留给旧二值实验
MOMENTUM = 0.9  # 论文值；仅 OPTIMIZER="sgd" 时生效
LOSS = "ce"  # 论文用交叉熵；目标为 0~115 的类别索引 long 张量
USE_CLASS_WEIGHTS = True  # 类别加权 CE：背景占 81%，不加权会塌缩成全预测背景
CLASS_WEIGHT_POWER = 0.5  # weight = (1/freq)^power；0.5 开方缓和极端稀有类
BCE_WEIGHT = 1.0  # 两项等权起点；前景极小时可试 DICE_WEIGHT=2.0
DICE_WEIGHT = 1.0  # 同上；这是实验旋钮，不是论文定值
GRAD_CLIP = (
    1.0  # 梯度 L2 范数上限，只缩长度不改方向；clip 必须夹在 backward 与 step 之间
)
THRESHOLD = 0.5  # sigmoid 后判前景；只用于指标与推理，不参与 loss
PATIENCE = 20  # 连续 20 轮 val mIoU 未创新高才早停；115 类冷启动慢，5 轮太短
SEED = 42  # random / numpy / torch 三处都要设，切分与初始化才可复现
DEVICE = (
    "cuda" if torch.cuda.is_available() else "cpu"
)  # 存字符串不存 torch.device，config 保持纯数据可打印

# ============================================================
# 四、路径
# ============================================================
RUN_NAME = "unet_plantseg_v3_mc_s512_b64_bs8_bn_bilinear_wce_sgd_flip_p20"  # v3=数据版本，mc=多类，s512=论文尺寸，bs8=减半，wce=加权CE，flip=仅翻转增强，p20=早停放宽到20轮
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
assert DATASET in ["oxford_pet", "toy", "folder", "plantseg"], (
    f"DATASET 只能是 oxford_pet/toy/folder/plantseg，收到 {DATASET!r}"
)
assert TASK in ["binary", "multiclass"], (
    f"TASK 只能是 binary/multiclass，收到 {TASK!r}"
)
assert OUT_CHANNELS == (NUM_CLASSES if TASK == "multiclass" else 1), (
    f"OUT_CHANNELS({OUT_CHANNELS}) 与 TASK({TASK}) 不一致"
)
assert len(NORM_MEAN) == IN_CHANNELS, (
    f"NORM_MEAN 长度 {len(NORM_MEAN)} != IN_CHANNELS({IN_CHANNELS})，Normalize 会形状不匹配"
)
assert len(NORM_STD) == IN_CHANNELS, (
    f"NORM_STD 长度 {len(NORM_STD)} != IN_CHANNELS({IN_CHANNELS})，Normalize 会形状不匹配"
)
