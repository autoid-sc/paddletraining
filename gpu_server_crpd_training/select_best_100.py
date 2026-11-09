#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从 CRPD_double 中筛选 100 张最清晰、省份+类型分布均匀的车牌图片。
策略:
  1. 对每张图, 取其中最高清的车牌作为代表 (省/类型)
  2. 按 (省, 类型) 分组, 每组内按裁剪高度降序
  3. 分配配额: 每 (省,类型) 组合最多取 1 张, 剩余按总缺口再均衡
  4. 总配额: 白牌25, 黄单20, 黄双15, 蓝牌40  (稀有类型优先)
"""
import os
import cv2
import shutil
import numpy as np
from collections import defaultdict
from tqdm import tqdm

TYPE_NAME = {0: "蓝", 1: "黄单", 2: "黄双", 3: "白"}
TYPE_QUOTA = {3: 25, 1: 20, 2: 15, 0: 40}  # 总计 100
TARGET_TOTAL = 100

BASE_DIR = "E:/xuhrWorkProject/floating_ocr_ai_paddle/docs/CRPD_all/CRPD_double"
OUT_DIR = "E:/xuhrWorkProject/floating_ocr_ai_paddle/docs/CRPD_all/CRPD_double_best100"


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)


def crop_height(img, pts):
    src = order_points(pts)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if w < 10 or h < 10:
        return 0
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
    # 清晰度 = 裁剪高度 * 拉普拉斯方差 (越高越清晰)
    if crop.size == 0:
        return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    # 综合分: 高度为主, 清晰度为辅
    return h * 100 + min(sharpness, 5000)


def parse_label(line):
    parts = line.strip().split()
    if len(parts) < 10:
        return None
    try:
        pts = [[float(parts[i]), float(parts[i+1])] for i in range(0, 8, 2)]
        ptype = int(parts[8])
        text = " ".join(parts[9:]).strip()
    except Exception:
        return None
    if len(text) < 2:
        return None
    return pts, ptype, text[0], text


def main():
    os.makedirs(os.path.join(OUT_DIR, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "labels"), exist_ok=True)

    # 分组: (省份, 类型) -> list of (score, img_path, label_file)
    groups = defaultdict(list)

    print("=== 扫描数据 ===")
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(BASE_DIR, split, "images")
        lbl_dir = os.path.join(BASE_DIR, split, "labels")
        if not os.path.isdir(img_dir) or not os.path.isdir(lbl_dir):
            continue

        for lf in tqdm(os.listdir(lbl_dir), desc=split):
            img_path = os.path.join(img_dir, lf.replace(".txt", ".jpg"))
            if not os.path.exists(img_path):
                continue
            img = cv2.imread(img_path)
            if img is None:
                continue

            best_in_img = None  # (score, ptype, province, text)
            with open(os.path.join(lbl_dir, lf), encoding="utf-8") as f:
                for line in f:
                    parsed = parse_label(line)
                    if parsed is None:
                        continue
                    pts, ptype, province, text = parsed
                    score = crop_height(img, pts)
                    if score < 2000:  # 太模糊的跳过 (高度<20px 或极糊)
                        continue
                    if best_in_img is None or score > best_in_img[0]:
                        best_in_img = (score, ptype, province, text)

            if best_in_img is None:
                continue
            score, ptype, province, text = best_in_img
            groups[(province, ptype)].append((score, img_path, lf))

    # ====== 选择阶段 1: 每 (省, 类型) 取最清晰的 1 张 ======
    selected = []
    used_imgs = set()

    for (province, ptype), items in groups.items():
        items_sorted = sorted(items, key=lambda x: -x[0])
        for score, img_path, lf in items_sorted:
            if img_path not in used_imgs:
                selected.append((score, img_path, lf, ptype, province))
                used_imgs.add(img_path)
                break

    print(f"阶段1: 每 (省,类型) 取1张, 已选 {len(selected)} 张")

    # 检查各类型已选数量
    for ptype in [3, 1, 2, 0]:
        cnt = sum(1 for s in selected if s[3] == ptype)
        print(f"  {TYPE_NAME[ptype]}: {cnt} 张 (目标 {TYPE_QUOTA[ptype]})")

    # ====== 阶段 2: 按类型配额补齐 ======
    for ptype in [3, 1, 2, 0]:
        target = TYPE_QUOTA[ptype]
        current = [s for s in selected if s[3] == ptype]
        need = target - len(current)
        if need <= 0:
            continue
        # 从同类型未选中的候选里, 按 score 降序补 (优先覆盖不同省份)
        candidates = []
        for (prov, pt), items in groups.items():
            if pt != ptype:
                continue
            items_sorted = sorted(items, key=lambda x: -x[0])
            for score, img_path, lf in items_sorted:
                if img_path in used_imgs:
                    continue
                candidates.append((score, img_path, lf, ptype, prov))
        candidates.sort(key=lambda x: -x[0])
        added = 0
        for c in candidates:
            if added >= need:
                break
            if c[1] in used_imgs:
                continue
            used_imgs.add(c[1])
            selected.append(c)
            added += 1
        print(f"  {TYPE_NAME[ptype]}: 补充 {added} 张, 现 {len(current) + added} 张")

    # ====== 阶段 3: 总配额补齐 (若不足 100) ======
    if len(selected) < TARGET_TOTAL:
        need = TARGET_TOTAL - len(selected)
        print(f"\n还差 {need} 张, 从所有候选中按清晰度补...")
        candidates = []
        for (prov, pt), items in groups.items():
            items_sorted = sorted(items, key=lambda x: -x[0])
            for score, img_path, lf in items_sorted:
                if img_path in used_imgs:
                    continue
                candidates.append((score, img_path, lf, pt, prov))
        candidates.sort(key=lambda x: -x[0])
        for c in candidates:
            if len(selected) >= TARGET_TOTAL:
                break
            if c[1] in used_imgs:
                continue
            used_imgs.add(c[1])
            selected.append(c)

    selected = selected[:TARGET_TOTAL]
    print(f"\n=== 最终选择 {len(selected)} 张 ===")

    # 复制文件
    for _, img_path, lf, _, _ in tqdm(selected, desc="复制文件"):
        shutil.copy(img_path, os.path.join(OUT_DIR, "images", os.path.basename(img_path)))
        src_lbl_dir = os.path.dirname(img_path).replace("images", "labels")
        shutil.copy(os.path.join(src_lbl_dir, lf), os.path.join(OUT_DIR, "labels", lf))

    # 统计并写入 stat.txt
    type_count = defaultdict(int)
    prov_count = defaultdict(int)
    examples = defaultdict(list)
    for _, _, lf, ptype, prov in selected:
        type_count[ptype] += 1
        prov_count[prov] += 1
        if len(examples[ptype]) < 4:
            with open(os.path.join(OUT_DIR, "labels", lf), encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 10 and int(parts[8]) == ptype:
                        examples[ptype].append(" ".join(parts[9:]))
                        break

    with open(os.path.join(OUT_DIR, "stat.txt"), "w", encoding="utf-8") as f:
        f.write("=== 类型分布 ===\n")
        for t in sorted(type_count):
            f.write(f"  {TYPE_NAME.get(t, str(t))}: {type_count[t]} 张\n")
            for ex in examples.get(t, []):
                f.write(f"    例: {ex}\n")
        f.write(f"\n=== 省份分布 ({len(prov_count)} 个省份) ===\n")
        for p in sorted(prov_count.keys(), key=lambda x: -prov_count[x]):
            f.write(f"  {p}: {prov_count[p]}\n")
        scores = [s[0] for s in selected]
        f.write(f"\n=== 清晰度分数 ===\n")
        f.write(f"  最小: {min(scores):.0f}\n")
        f.write(f"  25%:  {np.percentile(scores, 25):.0f}\n")
        f.write(f"  中位: {np.percentile(scores, 50):.0f}\n")
        f.write(f"  75%:  {np.percentile(scores, 75):.0f}\n")
        f.write(f"  最大: {max(scores):.0f}\n")


if __name__ == "__main__":
    main()
