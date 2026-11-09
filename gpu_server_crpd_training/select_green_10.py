#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从所有 CRPD_all 中挑选 10 张最清晰的新能源绿牌, 保证省份覆盖。
绿牌识别规则: 第2位是D/F (非F省城市代码) 或 第3位是D/F, 排除特殊牌。
"""
import os
import cv2
import shutil
import numpy as np
from collections import defaultdict

BASE = "E:/xuhrWorkProject/floating_ocr_ai_paddle/docs/CRPD_all"
OUT = "E:/xuhrWorkProject/floating_ocr_ai_paddle/docs/CRPD_all/CRPD_green_best10"
F_PROV = {"闽": "F"}  # F省城市代码


def is_green(text):
    if len(text) != 7: return False
    if text[-1] in "警使学领港澳挂": return False
    if text[0] == "字": return False
    if len(text) >= 2 and text[1] in "DF":
        if F_PROV.get(text[0]) == text[1]: return False
        return True
    if len(text) >= 3 and text[2] in "DF":
        return True
    return False


def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)


def crop_score(img, pts):
    src = order_points(pts)
    w = int(max(np.linalg.norm(src[1] - src[0]), np.linalg.norm(src[2] - src[3])))
    h = int(max(np.linalg.norm(src[3] - src[0]), np.linalg.norm(src[2] - src[1])))
    if w < 10 or h < 10: return 0
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    crop = cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))
    if crop.size == 0: return 0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    return h * 100 + min(sharpness, 5000)


def main():
    os.makedirs(os.path.join(OUT, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "labels"), exist_ok=True)

    candidates = []  # (score, text, ptype, sub, img_path, label_file)
    for sub in ["CRPD_single", "CRPD_double", "CRPD_multi"]:
        for split in ["train", "val", "test"]:
            lbl_dir = os.path.join(BASE, sub, split, "labels")
            if not os.path.isdir(lbl_dir): continue
            for fn in os.listdir(lbl_dir):
                img_path = os.path.join(BASE, sub, split, "images", fn.replace(".txt", ".jpg"))
                if not os.path.exists(img_path): continue
                img = cv2.imread(img_path)
                if img is None: continue
                with open(os.path.join(lbl_dir, fn), encoding="utf-8") as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) < 10: continue
                        try:
                            pts = [[float(parts[i]), float(parts[i+1])] for i in range(0, 8, 2)]
                            ptype = int(parts[8])
                            text = " ".join(parts[9:])
                        except: continue
                        if not is_green(text): continue
                        score = crop_score(img, pts)
                        if score < 5000: continue
                        candidates.append((score, text, ptype, sub, img_path, fn))

    print(f"共 {len(candidates)} 个绿牌候选 (清晰度 >= 5000)")

    # 按省份分组, 每省取最清晰 1 张, 凑够 10 个不同省份
    by_province = defaultdict(list)
    for c in candidates:
        by_province[c[1][0]].append(c)

    # 每个省份按清晰度降序
    for prov in by_province:
        by_province[prov].sort(key=lambda x: -x[0])

    # 按省份中位数清晰度排序, 选 top 10 省份
    prov_rank = sorted(by_province.keys(), key=lambda p: -by_province[p][0][0])

    selected = []
    for prov in prov_rank[:10]:
        selected.append(by_province[prov][0])

    print(f"\n=== 选定 {len(selected)} 张 ===")
    type_name = {0: "蓝", 1: "黄单", 2: "黄双", 3: "白"}

    # 复制文件
    for score, text, ptype, sub, img_path, lf in selected:
        shutil.copy(img_path, os.path.join(OUT, "images", os.path.basename(img_path)))
        for sp in ["train", "val", "test"]:
            p = os.path.join(BASE, sub, sp, "labels", lf)
            if os.path.exists(p):
                shutil.copy(p, os.path.join(OUT, "labels", lf))
                break

    # 写统计
    with open(os.path.join(OUT, "stat.txt"), "w", encoding="utf-8") as f:
        f.write(f"=== {len(selected)} 张最清晰绿牌 (每省1张, 省份按最高清晰度排序) ===\n\n")
        for score, text, ptype, sub, img_path, lf in selected:
            f.write(f"  省份={text[0]}  类型={type_name.get(ptype, str(ptype))}  "
                    f"车牌={text}  清晰度={score:.0f}  子集={sub}  文件={lf}\n")

    print(f"已保存到 {OUT}")


if __name__ == "__main__":
    main()
