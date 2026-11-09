#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CRPD_all -> PaddleOCR rec 格式.
支持 CRPD_single / CRPD_double / CRPD_multi 三个子集合并.
支持双行黄牌展开为单行.
CRPD 标注每行: x1 y1 x2 y2 x3 y3 x4 y4 type content  (空格分隔)
type: 0=蓝单行 1=黄单行 2=黄双行 3=白
用法:
    python3 prepare_crpd_rec_data.py --crpd_dir /data/CRPD_all --output_dir /data/crpd_train/rec
"""
import os
import cv2
import random
import argparse
import numpy as np
from tqdm import tqdm

provinces = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
alphabets = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
plate_chars = set(provinces + alphabets + list("0123456789") + ["挂","学","警","领","港","澳","使","超"])
TYPE_NAME = {0: "蓝单行", 1: "黄单行", 2: "黄双行", 3: "白"}

SUBSETS = ["CRPD_single", "CRPD_double", "CRPD_multi"]


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)  # tl,tr,br,bl


def content_ok(text):
    return len(text) >= 7 and all(c in plate_chars for c in text)


def warp_plate(img, pts):
    src = order_points(pts)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if w < 10 or h < 10:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))


def unfold_double(crop):
    """双行黄牌: 上(省+字母)约40%高, 下(数字)约60%高; 上下展开拼接成单行."""
    h, w = crop.shape[:2]
    split = int(h * 0.42)
    top = crop[0:split, :]
    bot = crop[split:h, :]
    th = max(1, split)
    bh = max(1, h - split)
    target_h = 48
    top_r = cv2.resize(top, (int(w * target_h / th), target_h))
    bot_r = cv2.resize(bot, (int(w * target_h / bh), target_h))
    return np.hstack([top_r, bot_r])


def degrade(crop, prob=0.7):
    if random.random() > prob:
        return crop
    h, w = crop.shape[:2]
    if random.random() < 0.5:
        s = random.uniform(0.3, 0.6)
        crop = cv2.resize(cv2.resize(crop, (max(2, int(w * s)), max(2, int(h * s)))), (w, h))
    if random.random() < 0.5:
        crop = cv2.GaussianBlur(crop, (random.choice([3, 5]),) * 2, 0)
    if random.random() < 0.5:
        _, e = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, random.randint(40, 80)])
        crop = cv2.imdecode(e, cv2.IMREAD_COLOR)
    return crop


def process_split(crpd_dir, subset, split, output_img_dir, start_idx, max_rec, types, degrade_prob):
    """处理一个子集的一个 split，返回 (rec_list, end_idx, type_count, heights)"""
    img_dir = os.path.join(crpd_dir, subset, split, "images")
    lbl_dir = os.path.join(crpd_dir, subset, split, "labels")
    if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
        print(f"  [SKIP] {subset}/{split} 目录不存在")
        return [], start_idx, {k: 0 for k in TYPE_NAME}, []

    rec_list = []
    type_count = {k: 0 for k in TYPE_NAME}
    heights = []
    idx = start_idx

    names = [n for n in os.listdir(lbl_dir) if n.endswith(".txt")]
    random.shuffle(names)

    for ln in tqdm(names, desc=f"  {subset}/{split}"):
        if max_rec > 0 and len(rec_list) >= max_rec:
            break
        img_path = os.path.join(img_dir, ln.replace(".txt", ".jpg"))
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg", ".png")
            if not os.path.exists(img_path):
                continue
        img = cv2.imread(img_path)
        if img is None:
            continue
        with open(os.path.join(lbl_dir, ln), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 10:
                    continue
                try:
                    pts = [[float(parts[i]), float(parts[i + 1])] for i in range(0, 8, 2)]
                    ptype = int(parts[8])
                    content = " ".join(parts[9:]).strip()
                except Exception:
                    continue
                if ptype not in types or not content_ok(content):
                    continue
                crop = warp_plate(img, pts)
                if crop is None:
                    continue
                if ptype == 2:  # 双行展开
                    crop = unfold_double(crop)
                if crop is None or crop.size == 0:
                    continue
                heights.append(crop.shape[0])
                type_count[ptype] += 1
                if degrade_prob > 0 and random.random() < degrade_prob:
                    crop = degrade(crop, 1.0)  # 已经概率判断过了
                fn = f"crop_{idx:06d}.jpg"
                cv2.imwrite(os.path.join(output_img_dir, fn), crop)
                rec_list.append(f"crop_images/{fn}\t{content}")
                idx += 1

    return rec_list, idx, type_count, heights


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--crpd_dir", default="/data/CRPD_all",
                   help="CRPD 根目录，包含 CRPD_single/CRPD_double/CRPD_multi")
    p.add_argument("--output_dir", default="/data/crpd_train/rec",
                   help="输出目录，crop_images/ 和 rec_*_list.txt 生成在这里")
    p.add_argument("--max_train", type=int, default=0,
                   help="训练集最大样本数（三个子集合并后总数上限），0=不限")
    p.add_argument("--max_val", type=int, default=0,
                   help="验证集最大样本数，0=不限")
    p.add_argument("--types", default="0,1,2,3",
                   help="保留牌型: 0蓝 1黄单 2黄双 3白")
    p.add_argument("--degrade_prob", type=float, default=0.5,
                   help="训练集降质增强概率 (0~1)")
    p.add_argument("--use_test_as_val", action="store_true",
                   help="用 test 集作为验证集（默认用 val 集）")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    random.seed(args.seed)
    types = [int(t) for t in args.types.split(",")]
    os.makedirs(os.path.join(args.output_dir, "crop_images"), exist_ok=True)

    val_split = "test" if args.use_test_as_val else "val"

    # ---- 训练集 ----
    print("=== 处理训练集 ===")
    train_all = []
    idx = 0
    train_type_count = {k: 0 for k in TYPE_NAME}
    train_heights = []
    for sub in SUBSETS:
        remaining = args.max_train - len(train_all) if args.max_train > 0 else 0
        if args.max_train > 0 and remaining <= 0:
            break
        items, idx, tc, hs = process_split(
            args.crpd_dir, sub, "train",
            os.path.join(args.output_dir, "crop_images"),
            idx, remaining, types, args.degrade_prob
        )
        for k in TYPE_NAME:
            train_type_count[k] += tc[k]
        train_heights.extend(hs)
        train_all.extend(items)
        print(f"  {sub}/train: {len(items)} 张")

    random.shuffle(train_all)

    # ---- 验证集 ----
    print(f"\n=== 处理验证集 ({val_split}) ===")
    val_all = []
    val_type_count = {k: 0 for k in TYPE_NAME}
    val_heights = []
    for sub in SUBSETS:
        remaining = args.max_val - len(val_all) if args.max_val > 0 else 0
        if args.max_val > 0 and remaining <= 0:
            break
        items, idx, tc, hs = process_split(
            args.crpd_dir, sub, val_split,
            os.path.join(args.output_dir, "crop_images"),
            idx, remaining, types, 0.0  # 验证集不降质
        )
        for k in TYPE_NAME:
            val_type_count[k] += tc[k]
        val_heights.extend(hs)
        val_all.extend(items)
        print(f"  {sub}/{val_split}: {len(items)} 张")

    random.shuffle(val_all)

    # ---- 写入 ----
    train_path = os.path.join(args.output_dir, "rec_train_list.txt")
    val_path = os.path.join(args.output_dir, "rec_val_list.txt")

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_all) + "\n")
    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_all) + "\n")

    # ---- 统计 ----
    print("\n=== 训练集统计 ===")
    print(f"  总数: {len(train_all)}")
    for k, v in train_type_count.items():
        print(f"  {TYPE_NAME[k]}: {v}")
    if train_heights:
        hs = np.array(train_heights)
        print(f"  裁剪高度: min={hs.min()} p25={int(np.percentile(hs,25))} "
              f"median={int(np.percentile(hs,50))} p75={int(np.percentile(hs,75))} max={hs.max()}")

    print(f"\n=== 验证集统计 ===")
    print(f"  总数: {len(val_all)}")
    for k, v in val_type_count.items():
        print(f"  {TYPE_NAME[k]}: {v}")

    print(f"\n=== 完成 ===")
    print(f"  训练集: {len(train_all)} 张  -> {train_path}")
    print(f"  验证集: {len(val_all)} 张  -> {val_path}")
    print(f"  裁剪图: {len(train_all) + len(val_all)} 张  -> {args.output_dir}/crop_images/")


if __name__ == "__main__":
    main()
