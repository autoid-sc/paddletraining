#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRPD -> PaddleOCR det 格式.
支持 CRPD_single / CRPD_double / CRPD_multi 三个子集合并.
CRPD 标注每行: x1 y1 x2 y2 x3 y3 x4 y4 type content  (空格分隔)
输出: det_train_list.txt, det_val_list.txt  (PaddleOCR SimpleDataSet 格式)
用法:
    python3 prepare_crpd_det_data.py --crpd_dir /data/CRPD_all --output_dir /data/crpd_train
"""
import os
import json
import random
import argparse
from tqdm import tqdm

# CRPD 子集目录名 (相对于 crpd_dir)
SUBSETS = ["CRPD_single", "CRPD_double", "CRPD_multi"]


def process_subset(crpd_dir, subset, split, max_samples=None):
    """处理一个子集的一个 split(train/val/test).
    返回 list of (img_abs_path, label_json_str)
    """
    img_dir = os.path.join(crpd_dir, subset, split, "images")
    lbl_dir = os.path.join(crpd_dir, subset, split, "labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        print(f"  [SKIP] {subset}/{split} 目录不存在")
        return []

    results = []
    label_files = [f for f in os.listdir(lbl_dir) if f.endswith(".txt")]
    if max_samples:
        random.shuffle(label_files)
        label_files = label_files[:max_samples]

    for ln in tqdm(label_files, desc=f"  {subset}/{split}"):
        # 对应图片名
        img_name = ln.replace(".txt", ".jpg")
        img_path = os.path.join(img_dir, img_name)
        if not os.path.exists(img_path):
            img_name_png = ln.replace(".txt", ".png")
            img_path = os.path.join(img_dir, img_name_png)
            if not os.path.exists(img_path):
                continue

        # 读取标注
        polys = []
        with open(os.path.join(lbl_dir, ln), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                try:
                    pts = [[float(parts[i]), float(parts[i + 1])] for i in range(0, 8, 2)]
                    content = " ".join(parts[9:]).strip()
                except Exception:
                    continue
                polys.append({
                    "transcription": content,
                    "points": pts,
                    "difficult": False,
                })

        if not polys:
            continue

        results.append((img_path, json.dumps(polys, ensure_ascii=False)))

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crpd_dir", default="/data/CRPD_all",
                   help="CRPD 根目录，包含 CRPD_single/CRPD_double/CRPD_multi")
    p.add_argument("--output_dir", default="/data/crpd_train",
                   help="输出目录，det_train_list.txt / det_val_list.txt 生成在这里")
    p.add_argument("--max_train_per_subset", type=int, default=0,
                   help="每个子集最多取多少张训练图，0=不限")
    p.add_argument("--max_val_per_subset", type=int, default=0,
                   help="每个子集最多取多少张验证图，0=不限")
    p.add_argument("--use_test_as_val", action="store_true",
                   help="用 test 集作为验证集（默认用 val 集）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    val_split = "test" if args.use_test_as_val else "val"
    max_train = args.max_train_per_subset if args.max_train_per_subset > 0 else None
    max_val = args.max_val_per_subset if args.max_val_per_subset > 0 else None

    # ---- 收集训练集 ----
    print("=== 收集训练集 ===")
    train_all = []
    for sub in SUBSETS:
        items = process_subset(args.crpd_dir, sub, "train", max_train)
        print(f"  {sub}/train: {len(items)} 张")
        train_all.extend(items)

    random.shuffle(train_all)

    # ---- 收集验证集 ----
    print(f"\n=== 收集验证集 ({val_split}) ===")
    val_all = []
    for sub in SUBSETS:
        items = process_subset(args.crpd_dir, sub, val_split, max_val)
        print(f"  {sub}/{val_split}: {len(items)} 张")
        val_all.extend(items)

    random.shuffle(val_all)

    # ---- 写入标注文件 ----
    train_path = os.path.join(args.output_dir, "det_train_list.txt")
    val_path = os.path.join(args.output_dir, "det_val_list.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        for img_path, label_str in train_all:
            f.write(f"{img_path}\t{label_str}\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for img_path, label_str in val_all:
            f.write(f"{img_path}\t{label_str}\n")

    print(f"\n=== 完成 ===")
    print(f"  训练集: {len(train_all)} 张  -> {train_path}")
    print(f"  验证集: {len(val_all)} 张  -> {val_path}")


if __name__ == "__main__":
    main()
