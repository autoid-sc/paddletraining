#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CRPD -> PaddleOCR rec 格式. 支持多牌型(蓝/黄单行/黄双行/白) + 双行展开 + 清晰度统计.
CRPD 标注每行: x1,y1,x2,y2,x3,y3,x4,y4,type,content
type: 0=蓝单行 1=黄单行 2=黄双行 3=白
用法: python3 prepare_crpd_data.py --crpd_dir /data/CRPD --output_dir /data/crpd_rec
"""
import os, cv2, random, argparse
import numpy as np
from tqdm import tqdm

provinces = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
alphabets = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
plate_chars = set(provinces + alphabets + list("0123456789") + ["挂","学","警","领","港","澳","使","超"])
TYPE_NAME = {0:"蓝单行",1:"黄单行",2:"黄双行",3:"白"}

def order_points(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmax(d)],
                     pts[np.argmax(s)], pts[np.argmin(d)]], dtype=np.float32)  # tl,tr,br,bl

def content_ok(text):
    return len(text) >= 7 and all(c in plate_chars for c in text)

def warp_plate(img, pts):
    src = order_points(pts)
    w = int(max(np.linalg.norm(src[1]-src[0]), np.linalg.norm(src[2]-src[3])))
    h = int(max(np.linalg.norm(src[3]-src[0]), np.linalg.norm(src[2]-src[1])))
    if w < 10 or h < 10: return None
    dst = np.array([[0,0],[w-1,0],[w-1,h-1],[0,h-1]], dtype=np.float32)
    return cv2.warpPerspective(img, cv2.getPerspectiveTransform(src, dst), (w, h))

def unfold_double(crop):
    """双行黄牌: 上(省+字母)约40%高, 下(数字)约60%高; 上下展开拼接成单行."""
    h, w = crop.shape[:2]
    split = int(h * 0.42)
    top = crop[0:split, :]
    bot = crop[split:h, :]
    th = max(1, split); bh = max(1, h - split)
    # 统一高度后水平拼接
    target_h = 48
    top_r = cv2.resize(top, (int(w * target_h / th), target_h))
    bot_r = cv2.resize(bot, (int(w * target_h / bh), target_h))
    return np.hstack([top_r, bot_r])

def degrade(crop, prob=0.7):
    if random.random() > prob: return crop
    h, w = crop.shape[:2]
    if random.random() < 0.5:
        s = random.uniform(0.3, 0.6)
        crop = cv2.resize(cv2.resize(crop, (max(2,int(w*s)), max(2,int(h*s)))), (w, h))
    if random.random() < 0.5:
        crop = cv2.GaussianBlur(crop, (random.choice([3,5]),)*2, 0)
    if random.random() < 0.5:
        _, e = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, random.randint(40,80)])
        crop = cv2.imdecode(e, cv2.IMREAD_COLOR)
    return crop

def process(crpd_dir, output_dir, max_rec=200000, degrade_prob=0.7, types=(0,1,2,3)):
    os.makedirs(os.path.join(output_dir, "crop_images"), exist_ok=True)
    img_dir, lbl_dir = os.path.join(crpd_dir,"images"), os.path.join(crpd_dir,"labels")
    if not os.path.isdir(img_dir):
        for sub in ["CRPD-single","CRPD-double","CRPD-multi",""]:
            if sub and os.path.isdir(os.path.join(crpd_dir, sub, "images")):
                img_dir = os.path.join(crpd_dir, sub, "images"); lbl_dir = os.path.join(crpd_dir, sub, "labels"); break

    rec_list = []; type_count = {k:0 for k in TYPE_NAME}; heights = []
    names = [n for n in os.listdir(lbl_dir) if n.endswith(".txt")]
    random.seed(42); random.shuffle(names)

    for ln in tqdm(names):
        if len(rec_list) >= max_rec: break
        img_path = os.path.join(img_dir, ln.replace(".txt",".jpg"))
        if not os.path.exists(img_path):
            img_path = img_path.replace(".jpg",".png")
            if not os.path.exists(img_path): continue
        img = cv2.imread(img_path)
        if img is None: continue
        with open(os.path.join(lbl_dir, ln), encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 10: continue
                try:
                    pts = [[float(parts[i]), float(parts[i+1])] for i in range(0,16,2)]
                    ptype = int(parts[8]); content = "".join(parts[9:]).strip()
                except Exception:
                    continue
                if ptype not in types or not content_ok(content): continue
                crop = warp_plate(img, pts)
                if crop is None: continue
                if ptype == 2:  # 双行展开
                    crop = unfold_double(crop)
                if crop is None or crop.size == 0: continue
                heights.append(crop.shape[0])
                type_count[ptype] += 1
                if random.random() < degrade_prob:
                    crop = degrade(crop)
                fn = f"crop_{len(rec_list):06d}.jpg"
                cv2.imwrite(os.path.join(output_dir,"crop_images",fn), crop)
                rec_list.append(f"crop_images/{fn}\t{content}")

    # 清晰度/牌型统计
    print("\n=== 牌型分布 ===")
    for k,v in type_count.items(): print(f"  {TYPE_NAME[k]}: {v}")
    if heights:
        hs = np.array(heights)
        print(f"=== 清晰度(裁剪高度) ===  min={hs.min()} p25={int(np.percentile(hs,25))} "
              f"median={int(np.percentile(hs,50))} p75={int(np.percentile(hs,75))} max={hs.max()}")
        low = (hs < 20).mean()*100
        print(f"  高度<20px 的低清样本占比: {low:.1f}%  (过高则考虑过滤)")

    random.shuffle(rec_list)
    split = int(len(rec_list)*0.9)
    with open(os.path.join(output_dir,"rec_train_list.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(rec_list[:split])+"\n")
    with open(os.path.join(output_dir,"rec_val_list.txt"),"w",encoding="utf-8") as f:
        f.write("\n".join(rec_list[split:])+"\n")
    print(f"rec train: {split}, val: {len(rec_list)-split}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--crpd_dir", default="/data/CRPD")
    p.add_argument("--output_dir", default="/data/crpd_rec")
    p.add_argument("--max_rec", type=int, default=200000)
    p.add_argument("--types", default="0,1,2,3", help="保留牌型: 0蓝 1黄单 2黄双 3白")
    a = p.parse_args()
    process(a.crpd_dir, a.output_dir, a.max_rec, types=[int(t) for t in a.types.split(",")])
