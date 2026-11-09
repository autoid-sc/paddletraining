#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CCPD Dataset Preprocessing Script for PaddleOCR.
Decodes filenames, crops plate regions for recognition, creates detection ground truth,
and splits the data 9:1.
"""
import os
import cv2
import json
import random
import glob
import argparse
from tqdm import tqdm

# Character mappings for CCPD
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "O"]

def decode_plate_number(plate_indices_str):
    """Decode CCPD plate indices (e.g. 0_0_22_27_27_33_16) to license plate string."""
    try:
        indices = [int(x) for x in plate_indices_str.split('_')]
        if len(indices) < 7:
            return None
        
        prov = provinces[indices[0]]
        alpha = alphabets[indices[1]]
        rest = ""
        for idx in indices[2:]:
            char = ads[idx]
            if char != "O": # 'O' is placeholder for empty
                rest += char
        return f"{prov}{alpha}{rest}"
    except Exception:
        return None

def parse_ccpd_filename(filepath):
    """
    Parse coordinates and plate number from CCPD filename.
    Format: area-tilt-boundingbox-fourvertices-plate-brightness-blur.jpg
    """
    basename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.split('-')
    if len(parts) < 5:
        return None
    
    try:
        # Bounding box coordinates
        bbox_str = parts[2]
        bbox = [list(map(int, pt.split('&'))) for pt in bbox_str.split('_')]
        # [[xmin, ymin], [xmax, ymax]]
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]
        
        # Four vertices (ccpd: bottom-right, bottom-left, top-left, top-right)
        vertices_str = parts[3]
        pts = [list(map(int, pt.split('&'))) for pt in vertices_str.split('_')]
        
        # Reorder to clockwise starting from top-left: top-left, top-right, bottom-right, bottom-left
        # ccpd indices: pts[2] = top-left, pts[3] = top-right, pts[0] = bottom-right, pts[1] = bottom-left
        points = [pts[2], pts[3], pts[0], pts[1]]
        
        # Decode plate string
        plate_str = decode_plate_number(parts[4])
        if not plate_str:
            return None
            
        return {
            "filepath": filepath,
            "xmin": xmin,
            "ymin": ymin,
            "xmax": xmax,
            "ymax": ymax,
            "points": points,
            "plate": plate_str
        }
    except Exception:
        return None

def process_ccpd(ccpd_dir, output_dir, max_det=15000, max_rec=30000):
    print("=" * 60)
    print(f"CCPD Parser & Preprocessor")
    print(f"Dataset root: {ccpd_dir}")
    print(f"Output folder: {output_dir}")
    print("=" * 60)
    
    # 1. Find all images in CCPD
    image_paths = []
    # Search in common directories: ccpd_base, ccpd_tilt, ccpd_db, ccpd_weather, ccpd_challenge, ccpd_blur, ccpd_fn
    subdirs = ["ccpd_base", "ccpd_tilt", "ccpd_db", "ccpd_weather", "ccpd_challenge", "ccpd_blur", "ccpd_fn"]
    for s in subdirs:
        search_path = os.path.join(ccpd_dir, s, "*.jpg")
        found = glob.glob(search_path)
        print(f"Found {len(found)} images in {s}")
        image_paths.extend(found)
        
    if not image_paths:
        # Try finding all jpgs recursively if subdirectory names don't match exactly
        print("[WARNING] Searching recursively in all directories...")
        image_paths = glob.glob(os.path.join(ccpd_dir, "**", "*.jpg"), recursive=True)
        
    print(f"Total images found: {len(image_paths)}")
    if not image_paths:
        print("[ERROR] No images found. Please check dataset path.")
        return
        
    # Shuffle for even distribution
    random.seed(42)
    random.shuffle(image_paths)
    
    # Create output subdirectories
    det_out_dir = os.path.join(output_dir, "det")
    rec_out_dir = os.path.join(output_dir, "rec")
    rec_img_dir = os.path.join(rec_out_dir, "crop_images")
    
    os.makedirs(det_out_dir, exist_ok=True)
    os.makedirs(rec_out_dir, exist_ok=True)
    os.makedirs(rec_img_dir, exist_ok=True)
    
    det_list = []
    rec_list = []
    
    print("\nParsing filenames and generating datasets...")
    
    for filepath in tqdm(image_paths):
        info = parse_ccpd_filename(filepath)
        if not info:
            continue
            
        # Add to detection dataset
        if len(det_list) < max_det:
            # We reference the absolute path of the original image to avoid copying files!
            line = f"{info['filepath']}\t" + json.dumps([{"points": info["points"], "transcription": info["plate"]}], ensure_ascii=False)
            det_list.append(line)
            
        # Add to recognition dataset (requires cropping)
        if len(rec_list) < max_rec:
            # Load image and crop plate
            img = cv2.imread(info["filepath"])
            if img is not None:
                # Add margin to crop
                h, w, _ = img.shape
                ymin = max(0, info["ymin"] - 2)
                ymax = min(h, info["ymax"] + 2)
                xmin = max(0, info["xmin"] - 2)
                xmax = min(w, info["xmax"] + 2)
                
                crop = img[ymin:ymax, xmin:xmax]
                if crop.size > 0:
                    crop_filename = f"crop_{len(rec_list):06d}.jpg"
                    crop_filepath = os.path.join(rec_img_dir, crop_filename)
                    cv2.imwrite(crop_filepath, crop)
                    
                    # Store path relative to rec/ folder: crop_images/crop_xxxx.jpg
                    rel_crop_path = f"crop_images/{crop_filename}"
                    rec_line = f"{rel_crop_path}\t{info['plate']}"
                    rec_list.append(rec_line)
                    
        # Stop early if we collected enough of both
        if len(det_list) >= max_det and len(rec_list) >= max_rec:
            break
            
    print(f"\nCollected {len(det_list)} detection items and {len(rec_list)} recognition items.")
    
    # 2. Write detection files
    random.shuffle(det_list)
    det_split = int(len(det_list) * 0.9)
    det_train = det_list[:det_split]
    det_val = det_list[det_split:]
    
    with open(os.path.join(output_dir, "det_train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(det_train) + "\n")
    with open(os.path.join(output_dir, "det_val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(det_val) + "\n")
        
    print(f"[OK] Saved detection lists. Train: {len(det_train)}, Val: {len(det_val)}")
    
    # 3. Write recognition files
    random.shuffle(rec_list)
    rec_split = int(len(rec_list) * 0.9)
    rec_train = rec_list[:rec_split]
    rec_val = rec_list[rec_split:]
    
    with open(os.path.join(output_dir, "rec_train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rec_train) + "\n")
    with open(os.path.join(output_dir, "rec_val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rec_val) + "\n")
        
    print(f"[OK] Saved recognition lists. Train: {len(rec_train)}, Val: {len(rec_val)}")
    
    # 4. Generate Plate Dictionary
    dict_file = os.path.join(output_dir, "plate_dict.txt")
    # Custom plate characters dict
    plate_chars = provinces + alphabets + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "挂", "学", "警", "领", "港", "澳", "使", "超"]
    with open(dict_file, "w", encoding="utf-8") as f:
        for char in plate_chars:
            f.write(char + "\n")
        f.write("\n")
    print(f"[OK] Generated custom plate dictionary at: {dict_file}")
    
    print("\n[SUCCESS] CCPD dataset preparation completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CCPD Dataset Parser")
    parser.add_argument("--ccpd_dir", type=str, default="/data/CCPD2019", help="Root directory of CCPD2019 dataset")
    parser.add_argument("--output_dir", type=str, default="/data/plate_train_data", help="Output directory for processed data")
    parser.add_argument("--max_det", type=int, default=15000, help="Maximum detection images to process")
    parser.add_argument("--max_rec", type=int, default=30000, help="Maximum recognition images to crop and process")
    
    args = parser.parse_args()
    process_ccpd(args.ccpd_dir, args.output_dir, args.max_det, args.max_rec)
