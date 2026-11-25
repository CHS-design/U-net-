"""prepare_plantseg.py — PlantSeg v3 元数据体检与修复产物生成

审查结论（2026-11）：
    1. 118 张 soybean rust 的掩膜前景值是 89（= 88 号 + 1），这是对的；但发行版
       Metadatav2.csv 把它们写成 bean rust / Index=13，而且 CSV 里根本没有 88 号。
    2. 另有 16 张掩膜的掩膜值与本行 Index+1 不符，文件名与 CSV 一致，属于掩膜侧错标。

本脚本只读 data/，只写三个产物，不改动任何图片或掩膜：
    configs/plantseg_classes.json       掩膜值 0~115 -> 作物/病名
    configs/plantseg_label_fixes.json   16 张错标掩膜的加载时修正表
    data/plantsegv3/metadata_fixed.csv  修正后的元数据副本（原 CSV 不动）

加载时修正由 utils/data.py 的 PlantSegSegDataset 读取 label_fixes 完成，
因此重下数据、换机器都不需要重新改标注文件。

用法：
    python scripts/prepare_plantseg.py            # 体检并生成
    python scripts/prepare_plantseg.py --check    # 只校验，不写文件；不一致退出码 1
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent

SPLIT_TO_DIR = {"Training": "train", "Validation": "val", "Test": "test"}
EXPECTED_ROWS = 11458
EXPECTED_SOYBEAN_RUST = 118
EXPECTED_MASK_FIXES = 16
NUM_CLASSES = 116

COLUMNS = [
    "Name",
    "Index",
    "Plant",
    "Disease",
    "Resolution",
    "Label file",
    "Mask ratio",
    "URL",
    "License",
    "Split",
]

# 这 118 行的掩膜值 89 才是真值：88 号 = soybean rust，掩膜值 = 88 + 1。
SYSTEMATIC_INDEX = 13
SYSTEMATIC_MASK_VALUE = 89
SYSTEMATIC_TRUE_INDEX = 88
SYSTEMATIC_TRUE_DISEASE = "soybean rust"


def load_metadata(path):
    """读 CSV，保留原始字符串（避免 Mask ratio 浮点往返改变格式）

    v7 的 Metadata.csv 带 UTF-8 BOM，v3 的 Metadatav2.csv 不带；
    utf-8-sig 对两者都安全，不会把 BOM 留进第一列列名。
    """
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = [dict(r) for r in reader]
        fields = list(reader.fieldnames or [])
    return fields, rows


def scan_mask_max(root, rows):
    """逐张读取掩膜，返回每行对应掩膜的前景最大值（背景图会是 0）"""
    values = []
    for i, row in enumerate(rows, 1):
        split = SPLIT_TO_DIR.get(row["Split"])
        if split is None:
            raise ValueError(f"未知 Split: {row['Split']!r}（第 {i} 行 {row['Name']}）")
        mask_path = root / "annotations" / split / row["Label file"]
        if not mask_path.is_file():
            raise FileNotFoundError(f"缺少掩膜: {mask_path}")
        with Image.open(mask_path) as im:
            arr = np.asarray(im.convert("L"))
        values.append(int(arr.max()))
        if i % 2000 == 0:
            print(f"  已扫描 {i}/{len(rows)} 张掩膜 ...")
    return values


def repair_rows(rows, mask_max):
    """修正 CSV 侧的 118 条错标，并记录掩膜侧的 16 张错标

    返回 (修正后的行, soybean_rust 行数, 掩膜修正列表, Grapevine 归一化行数)
    """
    fixed_rows, mask_fixes = [], []
    soybean_count = 0
    grapevine_count = 0

    for row, mval in zip(rows, mask_max):
        new_row = dict(row)
        idx = int(new_row["Index"])

        # 规则 1：CSV 侧错标。掩膜值 89 + 作物 Soybean + 原 Index=13，三者同时命中才改。
        if idx == SYSTEMATIC_INDEX and new_row["Plant"] == "Soybean" and mval == SYSTEMATIC_MASK_VALUE:
            new_row["Index"] = str(SYSTEMATIC_TRUE_INDEX)
            new_row["Disease"] = SYSTEMATIC_TRUE_DISEASE
            idx = SYSTEMATIC_TRUE_INDEX
            soybean_count += 1

        # 作物名归一：v3 把葡萄分成 Grape / Grapevine 两种写法，论文是 34 种作物。
        if new_row["Plant"] == "Grapevine":
            new_row["Plant"] = "Grape"
            grapevine_count += 1

        # 规则 2：掩膜侧错标。文件名与 CSV 一致，按 CSV 的 Index+1 修正。
        if mval != idx + 1:
            split = SPLIT_TO_DIR[new_row["Split"]]
            mask_fixes.append(
                {
                    "key": f"{split}/{new_row['Label file']}",
                    "from": mval,
                    "to": idx + 1,
                    "name": new_row["Name"],
                    "disease": new_row["Disease"],
                }
            )

        fixed_rows.append(new_row)

    return fixed_rows, soybean_count, mask_fixes, grapevine_count


def build_class_map(rows):
    """由修正后的行生成 掩膜值 -> (作物, 病名) 表，并校验 0~115 完整且无冲突"""
    classes = {0: {"plant": None, "disease": "background"}}
    conflicts = []
    for row in rows:
        cid = int(row["Index"]) + 1
        entry = {"plant": row["Plant"], "disease": row["Disease"]}
        if cid in classes:
            if classes[cid] != entry:
                conflicts.append((cid, classes[cid], entry, row["Name"]))
        else:
            classes[cid] = entry

    if conflicts:
        raise ValueError(f"同一掩膜值对应多种病名，拒绝生成: {conflicts[:5]}")
    missing = sorted(set(range(NUM_CLASSES)) - set(classes))
    if missing:
        raise ValueError(f"类别表缺 id: {missing}")
    return {str(k): classes[k] for k in range(NUM_CLASSES)}


def read_license_map(path):
    """v3 的 CSV 没有 License 列，按文件名从 v7 借；v7 不存在时返回空表"""
    if not path.is_file():
        return {}
    _, rows = load_metadata(path)
    return {r["Name"]: r.get("License", "") for r in rows if r.get("License")}


def attach_license(rows, license_map):
    """给修正后的行补 License 列；命中 v7 的填原值，其余填 unknown"""
    for row in rows:
        row["License"] = license_map.get(row["Name"], "unknown")
    return rows


def write_artifacts(root, classes, mask_fixes, rows, fields, check, source):
    out_classes = ROOT / "configs" / "plantseg_classes.json"
    out_fixes = ROOT / "configs" / "plantseg_label_fixes.json"
    out_csv = root / "metadata_fixed.csv"

    classes_doc = {
        "num_classes": NUM_CLASSES,
        "background": 0,
        "source": f"{source} (corrected by scripts/prepare_plantseg.py)",
        "classes": classes,
    }
    fixes_doc = {
        "version": 1,
        "note": "掩膜值与本行 Index+1 不符的 16 张；加载时按 from -> to 修正，不改磁盘文件",
        "fixes": {f["key"]: {"from": f["from"], "to": f["to"]} for f in mask_fixes},
    }

    if check:
        errors = []
        if not out_classes.is_file():
            errors.append(f"缺少 {out_classes.relative_to(ROOT)}，先运行不带 --check 的生成")
        elif json.loads(out_classes.read_text(encoding="utf-8")) != classes_doc:
            errors.append(f"{out_classes.relative_to(ROOT)} 与当前数据不一致")
        if not out_fixes.is_file():
            errors.append(f"缺少 {out_fixes.relative_to(ROOT)}，先运行不带 --check 的生成")
        elif json.loads(out_fixes.read_text(encoding="utf-8")) != fixes_doc:
            errors.append(f"{out_fixes.relative_to(ROOT)} 与当前数据不一致")
        if not out_csv.is_file():
            errors.append(f"缺少 {out_csv.relative_to(ROOT)}，先运行不带 --check 的生成")
        else:
            _, existing = load_metadata(out_csv)
            if existing != rows:
                errors.append(f"{out_csv.relative_to(ROOT)} 与当前数据不一致")
        return errors

    out_classes.parent.mkdir(parents=True, exist_ok=True)
    out_classes.write_text(
        json.dumps(classes_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    out_fixes.write_text(
        json.dumps(fixes_doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description="PlantSeg v3 元数据体检与修复产物生成")
    parser.add_argument("--root", default=str(ROOT / "data" / "plantsegv3"), help="PlantSeg v3 根目录")
    parser.add_argument(
        "--metadata",
        default="Metadatav2.csv",
        help="根目录下的元数据文件名（默认 Metadatav2.csv）",
    )
    parser.add_argument(
        "--license-ref",
        default=str(ROOT / "data" / "plantseg_v7" / "plantseg" / "Metadata.csv"),
        help="带 License 列的 v7 元数据，用于按文件名补 License；不存在则跳过",
    )
    parser.add_argument("--check", action="store_true", help="只校验产物是否与当前数据一致")
    args = parser.parse_args(argv)

    root = Path(args.root)
    if not (root / "images").is_dir() or not (root / "annotations").is_dir():
        print(f"[错误] 不是 PlantSeg 根目录: {root}", file=sys.stderr)
        return 1

    meta_path = root / args.metadata
    if not meta_path.is_file():
        print(f"[错误] 找不到元数据: {meta_path}", file=sys.stderr)
        return 1

    print(f"[1/4] 读取元数据 {meta_path}")
    _, rows = load_metadata(meta_path)
    if len(rows) != EXPECTED_ROWS:
        print(f"[错误] 行数 {len(rows)} != 预期 {EXPECTED_ROWS}", file=sys.stderr)
        return 1

    print(f"[2/4] 扫描 {len(rows)} 张掩膜（只读，不改文件）")
    mask_max = scan_mask_max(root, rows)

    print("[3/4] 校验与修正")
    fixed_rows, soybean_count, mask_fixes, grapevine_count = repair_rows(rows, mask_max)

    if soybean_count != EXPECTED_SOYBEAN_RUST:
        print(
            f"[错误] soybean rust 修正命中 {soybean_count} 行，预期 {EXPECTED_SOYBEAN_RUST}；"
            "数据可能已被改动，拒绝生成产物",
            file=sys.stderr,
        )
        return 1
    if len(mask_fixes) != EXPECTED_MASK_FIXES:
        print(
            f"[错误] 掩膜侧错标命中 {len(mask_fixes)} 张，预期 {EXPECTED_MASK_FIXES}；"
            "先人工确认再更新脚本预期值",
            file=sys.stderr,
        )
        for f in mask_fixes:
            print(f"  {f['key']}: {f['from']} -> {f['to']}", file=sys.stderr)
        return 1

    classes = build_class_map(fixed_rows)
    license_map = read_license_map(Path(args.license_ref))
    if not license_map:
        print(f"  [提示] 未找到 {args.license_ref}，License 列统一填 unknown")
    fixed_rows = attach_license(fixed_rows, license_map)

    print("[4/4] 写出产物")
    errors = write_artifacts(
        root, classes, mask_fixes, fixed_rows, COLUMNS, args.check, out_relative(meta_path)
    )
    if errors:
        for e in errors:
            print(f"  [不一致] {e}", file=sys.stderr)
        return 1

    if args.check:
        print("  [OK] 类别表、标签修正表、metadata_fixed.csv 均与当前数据一致")
        return 0

    licensed = sum(1 for r in fixed_rows if r["License"] != "unknown")
    cc0 = sum(1 for r in fixed_rows if r["License"] == "CC0")
    print(f"  类别表: {NUM_CLASSES} 个 id（0 背景 + 1~115 病害）")
    print(f"  soybean rust 修正: {soybean_count} 行（Index 13 -> 88，病名 bean rust -> soybean rust）")
    print(f"  掩膜值修正表: {len(mask_fixes)} 张（加载时生效，不改磁盘）")
    print(f"  作物名归一: {grapevine_count} 行（Grapevine -> Grape）")
    print(f"  License: {licensed}/{len(fixed_rows)} 有值，其中 CC0 {cc0} 张")
    print(f"  产物: configs/plantseg_classes.json")
    print(f"        configs/plantseg_label_fixes.json")
    print(f"        {out_relative(root / 'metadata_fixed.csv')}")
    return 0


def out_relative(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
