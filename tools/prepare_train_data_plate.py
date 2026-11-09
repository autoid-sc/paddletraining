#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare training data for license plate detection and recognition models.
- Detection: parses shapes from .json files, shuffles and splits 9:1 into det_train_list.txt and det_val_list.txt under plate folder
- Recognition: aggregates annotation.txt from subfolders or main folder, splits 9:1 into rec_train_list.txt and rec_val_list.txt
"""
import os
import json
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DET_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "plate", "det")
REC_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "plate", "rec")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "plate")

def prepare_detection_data():
    print("Processing detection dataset...")
    if not os.path.exists(DET_DIR):
        print(f"[WARNING] Detection directory not found: {DET_DIR}")
        print("Please create it and place your plate images and JSON label files there.")
        return
        
    json_files = [f for f in os.listdir(DET_DIR) if f.endswith(".json")]
    if not json_files:
        print(f"[WARNING] No JSON annotation files found in {DET_DIR}")
        return
        
    print(f"Found {len(json_files)} JSON files in {DET_DIR}")
    
    det_lines = []
    for json_file in json_files:
        json_path = os.path.join(DET_DIR, json_file)
        # Check corresponding image file
        image_name = json_file.replace(".json", ".jpg")
        image_path = os.path.join(DET_DIR, image_name)
        if not os.path.exists(image_path):
            image_name = json_file.replace(".json", ".png")
            image_path = os.path.join(DET_DIR, image_name)
            if not os.path.exists(image_path):
                image_name = json_file.replace(".json", ".jpeg")
                image_path = os.path.join(DET_DIR, image_name)
                if not os.path.exists(image_path):
                    continue
                
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            
            shapes = label_data.get("shapes", [])
            points_list = []
            for shape in shapes:
                points = shape.get("points", [])
                label = shape.get("label", "plate") # default label is plate
                points_list.append({
                    "transcription": label,
                    "points": points
                })
            
            # Standard PaddleOCR detection label format
            line = f"{image_name}\t{json.dumps(points_list, ensure_ascii=False)}"
            det_lines.append(line)
        except Exception as e:
            print(f"[WARNING] Failed to parse {json_file}: {e}")
            
    if not det_lines:
        print("[WARNING] No valid detection lines parsed.")
        return

    # Shuffle and split 9:1
    random.seed(42)
    random.shuffle(det_lines)
    split_idx = int(len(det_lines) * 0.9)
    train_lines = det_lines[:split_idx]
    val_lines = det_lines[split_idx:]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_file = os.path.join(OUTPUT_DIR, "det_train_list.txt")
    val_file = os.path.join(OUTPUT_DIR, "det_val_list.txt")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")
            
    print(f"[OK] Detection split complete. Train: {len(train_lines)}, Val: {len(val_lines)}")
    print(f"Saved: {train_file} and {val_file}")

def prepare_recognition_data():
    print("\nProcessing recognition dataset...")
    if not os.path.exists(REC_DIR):
        print(f"[WARNING] Recognition directory not found: {REC_DIR}")
        print("Please create it and place your license plate cropped images and annotation files there.")
        return
        
    # We will search for annotation files: 'annotation.txt' or 'anotation.txt'
    # in the REC_DIR directly, and also in its subdirectories.
    annotation_files = []
    
    # 1. Check main directory
    for f in ["annotation.txt", "anotation.txt"]:
        p = os.path.join(REC_DIR, f)
        if os.path.exists(p):
            annotation_files.append((REC_DIR, p))
            
    # 2. Check subdirectories
    subdirs = [d for d in os.listdir(REC_DIR) if os.path.isdir(os.path.join(REC_DIR, d))]
    for subdir in subdirs:
        subdir_path = os.path.join(REC_DIR, subdir)
        for f in ["annotation.txt", "anotation.txt"]:
            p = os.path.join(subdir_path, f)
            if os.path.exists(p):
                annotation_files.append((subdir_path, p))
                
    if not annotation_files:
        print(f"[WARNING] No annotation.txt or anotation.txt found in {REC_DIR} or subdirectories.")
        return
        
    print(f"Found {len(annotation_files)} annotation files to process.")
    
    rec_lines = []
    
    for dir_path, ann_path in annotation_files:
        try:
            with open(ann_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            # Get directory relative to REC_DIR
            rel_dir = os.path.relpath(dir_path, REC_DIR)
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                    
                img_file, text_label = parts
                text_label = text_label.strip()
                if not text_label:
                    continue
                
                # Relativize path
                if rel_dir == ".":
                    rel_img_path = img_file
                else:
                    rel_img_path = os.path.join(rel_dir, img_file).replace("\\", "/")
                    
                # Verify image exists
                full_img_path = os.path.join(dir_path, img_file)
                if not os.path.exists(full_img_path):
                    continue
                    
                formatted_line = f"{rel_img_path}\t{text_label}"
                rec_lines.append(formatted_line)
        except Exception as e:
            print(f"[WARNING] Failed to parse annotation {ann_path}: {e}")
            
    if not rec_lines:
        print("[WARNING] No valid recognition lines parsed.")
        return
        
    # Shuffle and split 9:1
    random.seed(42)
    random.shuffle(rec_lines)
    split_idx = int(len(rec_lines) * 0.9)
    train_lines = rec_lines[:split_idx]
    val_lines = rec_lines[split_idx:]
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_file = os.path.join(OUTPUT_DIR, "rec_train_list.txt")
    val_file = os.path.join(OUTPUT_DIR, "rec_val_list.txt")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")
            
    print(f"[OK] Recognition split complete. Train: {len(train_lines)}, Val: {len(val_lines)}")
    print(f"Saved: {train_file} and {val_file}")

def main():
    os.makedirs(DET_DIR, exist_ok=True)
    os.makedirs(REC_DIR, exist_ok=True)
    prepare_detection_data()
    prepare_recognition_data()
    print("\nData preparation check completed!")

if __name__ == "__main__":
    main()
