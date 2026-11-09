#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCPD Dataset Preprocessing for PaddleOCR PP-OCRv6 fine-tuning. v4
Balanced sampling, perspective warp, color images, 73-char dict (no dot).
Generates: det_train/val_list.txt, rec_train/val_list.txt, plate_dict.txt (73 chars).
"""
import os
import cv2
import json
import random
import glob
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

# 31 provinces
provinces = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
# 24 alphabets (no I, no O)
alphabets = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
# ads: alphabets + digits + O(placeholder)
ads = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9","O"]

def decode_plate_number(plate_indices_str):
    """Decode CCPD plate indices to license plate string with dot separator."""
    try:
        indices = [int(x) for x in plate_indices_str.split('_')]
        if len(indices) < 7:
            return None
        prov = provinces[indices[0]]
        alpha = alphabets[indices[1]]
        rest = ""
        for idx in indices[2:]:
            char = ads[idx]
            if char != "O":
                rest += char
        return f"{prov}{alpha}{rest}"
    except Exception:
        return None

def parse_ccpd_filename(filepath):
    """Parse coordinates and plate number from CCPD filename."""
    basename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.split('-')
    if len(parts) < 5:
        return None
    try:
        bbox_str = parts[2]
        bbox = [list(map(int, pt.split('&'))) for pt in bbox_str.split('_')]
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]

        vertices_str = parts[3]
        pts = [list(map(int, pt.split('&'))) for pt in vertices_str.split('_')]
        # Reorder to clockwise from top-left
        points = [pts[2], pts[3], pts[0], pts[1]]

        plate_str = decode_plate_number(parts[4])
        if not plate_str:
            return None

        # 省份索引用于平衡采样
        prov_idx = int(parts[4].split('_')[0])

        return {
            "filepath": filepath,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "points": points, "plate": plate_str,
            "prov_idx": prov_idx
        }
    except Exception:
        return None

def degrade(crop, prob=0.7):
    """模拟真实摄像头采集的退化: 下采样+模糊+噪声+JPEG压缩+亮度.
    这是提升真实场景准确率的关键——训练数据必须包含测试时的退化."""
    if random.random() > prob:
        return crop
    h, w = crop.shape[:2]
    # 1. 下采样再上采样 (模拟低分辨率)
    if random.random() < 0.5:
        scale = random.uniform(0.3, 0.6)
        small = cv2.resize(crop, (max(2, int(w*scale)), max(2, int(h*scale))))
        crop = cv2.resize(small, (w, h))
    # 2. 高斯模糊 (模拟失焦/运动模糊)
    if random.random() < 0.5:
        k = random.choice([3, 5])
        crop = cv2.GaussianBlur(crop, (k, k), 0)
    # 3. 高斯噪声 (模拟传感器噪声)
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(5, 20), crop.shape)
        crop = np.clip(crop.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # 4. JPEG 压缩伪影 (模拟摩尔纹/压缩)
    if random.random() < 0.5:
        q = random.randint(40, 80)
        _, enc = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, q])
        crop = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    # 5. 亮度/对比度 (模拟曝光差异)
    if random.random() < 0.4:
        alpha = random.uniform(0.7, 1.3); beta = random.uniform(-30, 30)
        crop = np.clip(crop.astype(np.float32)*alpha + beta, 0, 255).astype(np.uint8)
    return crop

def process_ccpd(ccpd_dir, output_dir, max_det=30000, max_rec=300000,
                 max_per_prov=5000, grayscale=False):
    print("=" * 60)
    print(f"CCPD v3 Preprocessor")
    print(f"  Dataset root: {ccpd_dir}")
    print(f"  Output:       {output_dir}")
    print(f"  max_per_prov: {max_per_prov} (balanced sampling)")
    print(f"  grayscale:    {grayscale}")
    print(f"  max_det:      {max_det}")
    print(f"  max_rec:      {max_rec}")
    print("=" * 60)

    # 1. 扫描所有图片，按省份分桶
    prov_buckets = defaultdict(list)
    subdirs = ["ccpd_base","ccpd_tilt","ccpd_db","ccpd_weather","ccpd_challenge","ccpd_blur","ccpd_fn"]
    for s in subdirs:
        found = glob.glob(os.path.join(ccpd_dir, s, "*.jpg"))
        print(f"  {s}: {len(found)} images")
        for f in found:
            info = parse_ccpd_filename(f)
            if info:
                prov_buckets[info["prov_idx"]].append(info)

    print(f"\n省份分布 (平衡前):")
    for idx in sorted(prov_buckets.keys()):
        print(f"  [{idx:2d}] {provinces[idx]}: {len(prov_buckets[idx])}")

    # 2. 平衡采样：每省最多 max_per_prov 张
    random.seed(42)
    balanced = []
    for idx in sorted(prov_buckets.keys()):
        bucket = prov_buckets[idx]
        random.shuffle(bucket)
        selected = bucket[:max_per_prov]
        balanced.extend(selected)
        if len(bucket) > max_per_prov:
            print(f"  [{idx:2d}] {provinces[idx]}: {len(bucket)} -> {max_per_prov} (trimmed)")

    random.shuffle(balanced)
    print(f"\n平衡后总量: {len(balanced)}")

    # 统计平衡后的分布
    bal_dist = defaultdict(int)
    for info in balanced:
        bal_dist[info["prov_idx"]] += 1
    print("省份分布 (平衡后):")
    for idx in sorted(bal_dist.keys()):
        print(f"  [{idx:2d}] {provinces[idx]}: {bal_dist[idx]}")

    # 3. 创建输出目录
    det_out_dir = os.path.join(output_dir, "det")
    rec_out_dir = os.path.join(output_dir, "rec")
    rec_img_dir = os.path.join(rec_out_dir, "crop_images")
    os.makedirs(det_out_dir, exist_ok=True)
    os.makedirs(rec_img_dir, exist_ok=True)

    det_list = []
    rec_list = []

    # 4. 处理每张图片
    print(f"\nProcessing {len(balanced)} images...")
    for info in tqdm(balanced):
        # Detection
        if len(det_list) < max_det:
            line = f"{info['filepath']}\t" + json.dumps(
                [{"points": info["points"], "transcription": info["plate"]}],
                ensure_ascii=False)
            det_list.append(line)

        # Recognition: perspective-corrected crop
        if len(rec_list) < max_rec:
            img = cv2.imread(info["filepath"])
            if img is not None:
                src_pts = np.array(info["points"], dtype=np.float32)
                w_top = np.linalg.norm(src_pts[1] - src_pts[0])
                w_bot = np.linalg.norm(src_pts[2] - src_pts[3])
                h_left = np.linalg.norm(src_pts[3] - src_pts[0])
                h_right = np.linalg.norm(src_pts[2] - src_pts[1])
                dst_w = int(max(w_top, w_bot))
                dst_h = int(max(h_left, h_right))
                if dst_w > 10 and dst_h > 10:
                    dst_pts = np.array([
                        [0, 0], [dst_w - 1, 0],
                        [dst_w - 1, dst_h - 1], [0, dst_h - 1]
                    ], dtype=np.float32)
                    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                    crop = cv2.warpPerspective(img, M, (dst_w, dst_h))
                else:
                    h, w, _ = img.shape
                    ymin = max(0, info["ymin"] - 2)
                    ymax = min(h, info["ymax"] + 2)
                    xmin = max(0, info["xmin"] - 2)
                    xmax = min(w, info["xmax"] + 2)
                    crop = img[ymin:ymax, xmin:xmax]

                if crop.size > 0:
                    # v4: 真实退化增强 (下采样+模糊+噪声+JPEG+亮度)
                    crop = degrade(crop)
                    if grayscale:
                        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                        crop = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                    crop_filename = f"crop_{len(rec_list):06d}.jpg"
                    crop_filepath = os.path.join(rec_img_dir, crop_filename)
                    cv2.imwrite(crop_filepath, crop)
                    rel_path = f"crop_images/{crop_filename}"
                    rec_list.append(f"{rel_path}\t{info['plate']}")

    print(f"\nDet: {len(det_list)}, Rec: {len(rec_list)}")

    # 5. 划分 train/val (9:1)
    random.shuffle(det_list)
    det_split = int(len(det_list) * 0.9)
    with open(os.path.join(output_dir, "det_train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(det_list[:det_split]) + "\n")
    with open(os.path.join(output_dir, "det_val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(det_list[det_split:]) + "\n")
    print(f"det train: {det_split}, val: {len(det_list)-det_split}")

    random.shuffle(rec_list)
    rec_split = int(len(rec_list) * 0.9)
    with open(os.path.join(output_dir, "rec_train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rec_list[:rec_split]) + "\n")
    with open(os.path.join(output_dir, "rec_val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rec_list[rec_split:]) + "\n")
    print(f"rec train: {rec_split}, val: {len(rec_list)-rec_split}")

    # 6. 生成字典 (73 字符，不含分隔符)
    plate_chars = provinces + alphabets + ["0","1","2","3","4","5","6","7","8","9",
                     "挂","学","警","领","港","澳","使","超"]
    dict_file = os.path.join(output_dir, "plate_dict.txt")
    with open(dict_file, "w", encoding="utf-8") as f:
        for c in plate_chars:
            f.write(c + "\n")
        f.write("\n")  # trailing empty line (REQUIRED by PaddleOCR)
    print(f"Dict: {len(plate_chars)} chars -> {dict_file}")

    # 7. 抽样验证
    print("\n--- 抽样验证 ---")
    for i in range(min(5, len(rec_list))):
        print(f"  {rec_list[i]}")
    print("DONE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccpd_dir", default="/data/CCPD2019")
    parser.add_argument("--output_dir", default="/data/plate_train_data")
    parser.add_argument("--max_det", type=int, default=30000)
    parser.add_argument("--max_rec", type=int, default=300000)
    parser.add_argument("--max_per_prov", type=int, default=5000,
                        help="Max images per province for balanced sampling (5000*31=~155k)")
    parser.add_argument("--grayscale", action="store_true",
                        help="Enable grayscale conversion (default: color images)")
    args = parser.parse_args()
    process_ccpd(args.ccpd_dir, args.output_dir, args.max_det, args.max_rec,
                 args.max_per_prov, grayscale=args.grayscale)
