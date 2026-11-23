"""utils/data.py — 数据加载与预处理"""

import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet

from configs import config as cfg
from utils.metrics import validate_labels


def build_image_transform(augment=False):
    """PIL 图像 → 归一化后的 (3,H,W) float32 张量

    几何增强（水平/垂直翻转）在 JointTransform 里对图像和掩膜同步做。
    这里不做光度扰动（ColorJitter）——病害判别依赖颜色/纹理，避免扰动丢失信息。
    augment 参数保留仅为接口一致。
    """
    return transforms.Compose(
        [
            transforms.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.NORM_MEAN, cfg.NORM_STD),
        ]
    )


def build_mask_transform():
    """PIL 掩膜 → 保持原始标签值的 (1,H,W) uint8 张量"""
    return transforms.Compose(
        [
            # TODO 4: Resize 到同样尺寸，插值参数用 transforms.InterpolationMode.NEAREST
            transforms.Resize(
                (cfg.IMAGE_SIZE, cfg.IMAGE_SIZE),
                interpolation=transforms.InterpolationMode.NEAREST,
            ),
            # TODO 5: PILToTensor()（不是 ToTensor）
            transforms.PILToTensor(),
        ]
    )


class JointTransform:
    """把图像变换与掩膜变换组合成一个可调用对象

    augment=True 时对图像和掩膜做同步的随机水平/垂直翻转（同一个随机决策），
    保证掩膜与图像严格对齐。ColorJitter 只作用于图像，已包含在 image_tf 中。
    """

    def __init__(self, image_tf, mask_tf, augment=False):
        self.image_tf = image_tf
        self.mask_tf = mask_tf
        self.augment = augment

    def __call__(self, image, mask):
        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        return self.image_tf(image), self.mask_tf(mask)


class OxfordPetSegDataset(Dataset):
    """Oxford Pet 二值分割数据集（trimap == 1 为前景）"""

    def __init__(self, split="trainval", root=cfg.DATA_ROOT, augment=False):
        super().__init__()
        self.base = OxfordIIITPet(
            root=root,
            split=split,
            target_types="segmentation",
            transforms=JointTransform(
                build_image_transform(augment=augment),
                build_mask_transform(),
                augment=augment,
            ),
            download=False,
        )

    def __len__(self):
        # TODO 4: 返回 self.base 的长度
        return len(self.base)

    def __getitem__(self, idx):
        # TODO 5: 从 self.base 取 (image, raw_mask)
        # TODO 6: mask = (raw_mask == 1).float()   # 不是 > cfg.MASK_THRESHOLD
        # TODO 7: 返回 (image, mask)
        image, raw_mask = self.base[idx]
        mask = (raw_mask == 1).float()
        return image, mask


class FolderSegDataset(Dataset):
    """自定义目录：images/ 与 masks/ 按同名主干配对"""

    def __init__(self, root):
        super().__init__()
        root = Path(root)
        # TODO 1: self.images_dir = root / "images"；self.masks_dir = root / "masks"
        # TODO 2: self.image_paths = sorted(self.images_dir.iterdir())
        # TODO 3: 遍历 image_paths，检查 self.masks_dir / (p.stem + ".png") 是否存在；
        #         不存在就 raise FileNotFoundError(f"缺少掩膜: {mask_path.name}")
        # TODO 4: self.joint = JointTransform(build_image_transform(), build_mask_transform())
        self.images_dir = root / "images"
        self.masks_dir = root / "masks"
        self.image_paths = sorted(self.images_dir.iterdir())
        for p in self.image_paths:
            mask_path = self.masks_dir / (p.stem + ".png")
            if not mask_path.exists():
                raise FileNotFoundError(f"缺少掩膜: {mask_path.name}")
        self.joint = JointTransform(build_image_transform(), build_mask_transform())

    def __len__(self):
        # TODO 5: 返回 len(self.image_paths)
        return len(self.image_paths)

    def __getitem__(self, idx):
        # TODO 6: image = Image.open(self.image_paths[idx]).convert("RGB")
        # TODO 7: mask_path = self.masks_dir / (self.image_paths[idx].stem + ".png")
        #         mask = Image.open(mask_path).convert("L")
        # TODO 8: image, raw_mask = self.joint(image, mask)
        # TODO 9: mask = (raw_mask > cfg.MASK_THRESHOLD).float()
        # TODO 10: return image, mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask_path = self.masks_dir / (self.image_paths[idx].stem + ".png")
        mask = Image.open(mask_path).convert("L")
        image, raw_mask = self.joint(image, mask)
        mask = (raw_mask > cfg.MASK_THRESHOLD).float()
        return image, mask


class PlantSegSegDataset(Dataset):
    """PlantSeg 语义分割：images/{split} 与 annotations/{split} 同名配对

    PNG 是类别索引图：0=背景，N（1~115）=第 N 类病害。
    TASK="multiclass" 时保留原索引 (H,W) int64，配 116 类交叉熵；
    TASK="binary" 时仍压成 (1,H,W) float{0,1}，兼容旧二值链路。
    """

    def __init__(self, split="train", root=cfg.PLANTSEG_ROOT, augment=False):
        super().__init__()
        if split not in ("train", "val", "test"):
            raise ValueError(f"PlantSeg split 只能是 train/val/test，收到 {split!r}")
        root = Path(root)
        self.images_dir = root / "images" / split
        self.masks_dir = root / "annotations" / split
        for d in (self.images_dir, self.masks_dir):
            if not d.is_dir():
                raise FileNotFoundError(f"PlantSeg 目录不存在: {d}")

        self.image_paths = sorted(
            p
            for p in self.images_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if not self.image_paths:
            raise FileNotFoundError(f"PlantSeg 目录下没有图像: {self.images_dir}")
        for p in self.image_paths:
            mask_path = self.masks_dir / f"{p.stem}.png"
            if not mask_path.exists():
                raise FileNotFoundError(f"缺少掩膜: {mask_path}")

        self.joint = JointTransform(
            build_image_transform(augment=augment),
            build_mask_transform(),
            augment=augment,
        )
        self.threshold = cfg.PLANTSEG_MASK_THRESHOLD

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.masks_dir / f"{image_path.stem}.png"
        with Image.open(image_path) as im:
            image = im.convert("RGB")
        with Image.open(mask_path) as im:
            mask = im.convert("L")
        image, raw_mask = self.joint(image, mask)
        validate_labels(raw_mask, cfg.NUM_CLASSES, source=f": {mask_path}")
        if cfg.TASK == "multiclass":
            # PILToTensor 给 (1,H,W) uint8；CE 的 target 要 (H,W) int64
            return image, raw_mask.squeeze(0).long()
        # 保留旧二值链路：(1,H,W) float32
        return image, (raw_mask > self.threshold).float()



class ToySegDataset(Dataset):  # pyright: ignore[reportMissingTypeArgument]
    """玩具分割数据集：随机图 + 中心方块掩膜

    输入：idx（样本序号，int，本类暂不使用）
    输出：(image, mask)，形状 (3,H,W) 与 (1,H,W)
    """

    def __init__(self, num_samples, image_size):
        super().__init__()
        # TODO 1: 把 num_samples 存成 self.num_samples，image_size 存成 self.image_size（各一行）
        # TODO 2: 从 cfg 读 MASK_THRESHOLD 存成 self.threshold，禁止写 127
        self.num_samples = num_samples
        self.image_size = image_size
        self.threshold = cfg.MASK_THRESHOLD

    def __len__(self):
        # TODO 3: 返回样本数（一行）
        return self.num_samples

    def __getitem__(self, idx):
        H = self.image_size
        W = self.image_size

        # TODO 4: 用 torch.rand 造 image，形状 (3, H, W)
        image = torch.rand(3, H, W)

        # TODO 5: 用 torch.zeros 造 raw_mask，形状 (1, H, W)
        raw_mask = torch.zeros(1, H, W)

        # TODO 6: 算 H//4 和 3*H//4 作边界，给中心矩形填 255
        #         提示写法：raw_mask[:, a:b, a:b] = 255
        a = H // 4
        b = 3 * H // 4
        raw_mask[:, a:b, a:b] = 255

        # TODO 7: 阈值化并转 float32：mask = (...) .float()
        mask = raw_mask.gt(self.threshold).float()

        # TODO 8: 返回二元组 (image, mask)
        return image, mask


def build_datasets():
    """返回 (train_set, val_set, test_set)"""
    if cfg.DATASET == "toy":
        full = ToySegDataset(cfg.TOY_SIZE, cfg.IMAGE_SIZE)
        # TODO 1: 用 Subset 切成 0:32 / 32:40 / 40:48 三段并返回
        train_set = Subset(full, range(32))
        val_set = Subset(full, range(32, 40))
        test_set = Subset(full, range(40, 48))
        return train_set, val_set, test_set

    elif cfg.DATASET == "oxford_pet":
        # 无增强模式：train / val 均不做增强。
        # 保留两个独立实例与相同 seed 的 perm，保证索引划分与增强实验完全一致，便于对照。
        train_full = OxfordPetSegDataset("trainval", augment=False)
        val_full = OxfordPetSegDataset("trainval", augment=False)
        test_set = OxfordPetSegDataset("test")

        g = torch.Generator().manual_seed(cfg.SEED)
        perm = torch.randperm(len(train_full), generator=g)
        val_size = int(len(train_full) * cfg.VAL_RATIO)
        val_set = Subset(val_full, perm[:val_size])
        train_set = Subset(train_full, perm[val_size:])
        return train_set, val_set, test_set

    elif cfg.DATASET == "plantseg":
        # 用官方划分（train/val/test），不做二次切分；train 开增强（同步翻转 + ColorJitter），val/test 不增强
        return (
            PlantSegSegDataset("train", augment=True),
            PlantSegSegDataset("val", augment=False),
            PlantSegSegDataset("test"),
        )

    else:
        raise NotImplementedError("folder loader 留到 2d")


def build_dataloaders():
    """返回 (train_loader, val_loader, test_loader)"""
    train_set, val_set, test_set = build_datasets()
    # TODO 7: 三个 DataLoader；train shuffle=True，val/test shuffle=False
    #         参数用 cfg.BATCH_SIZE / cfg.NUM_WORKERS
    DataLoader_train = DataLoader(
        train_set, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS
    )
    DataLoader_val = DataLoader(
        val_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS
    )
    DataLoader_test = DataLoader(
        test_set, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS
    )
    return DataLoader_train, DataLoader_val, DataLoader_test
