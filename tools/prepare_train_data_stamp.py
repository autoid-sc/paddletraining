#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare training data for detection and recognition models.
- Detection: parses shapes from .json files, shuffles and splits 9:1 into det_train_list.txt and det_val_list.txt
- Recognition: aggregates anotation.txt from subfolders, splits 9:1 into rec_train_list.txt and rec_val_list.txt, and extracts unique character dictionary
"""
import os
import json
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DET_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "det")
REC_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "rec")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec")

def prepare_detection_data():
    print("Processing detection dataset...")
    if not os.path.exists(DET_DIR):
        print(f"[ERROR] Detection directory not found: {DET_DIR}")
        return
        
    json_files = [f for f in os.listdir(DET_DIR) if f.endswith(".json")]
    print(f"Found {len(json_files)} JSON files in {DET_DIR}")
    
    det_lines = []
    for json_file in json_files:
        json_path = os.path.join(DET_DIR, json_file)
        # Corresponding image file
        image_name = json_file.replace(".json", ".jpg")
        image_path = os.path.join(DET_DIR, image_name)
        if not os.path.exists(image_path):
            # Try png or jpeg
            image_name = json_file.replace(".json", ".png")
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
                label = shape.get("label", "1")
                points_list.append({
                    "transcription": label,
                    "points": points
                })
            
            # Standard PaddleOCR detection label format: relative_path \t JSON_string
            # Since we will set data_dir to DET_DIR, we just need the image filename
            line = f"{image_name}\t{json.dumps(points_list, ensure_ascii=False)}"
            det_lines.append(line)
        except Exception as e:
            print(f"[WARNING] Failed to parse {json_file}: {e}")
            
    # Shuffle and split 9:1
    random.seed(42)
    random.shuffle(det_lines)
    split_idx = int(len(det_lines) * 0.9)
    train_lines = det_lines[:split_idx]
    val_lines = det_lines[split_idx:]
    
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
        print(f"[ERROR] Recognition directory not found: {REC_DIR}")
        return
        
    subdirs = [d for d in os.listdir(REC_DIR) if os.path.isdir(os.path.join(REC_DIR, d))]
    print(f"Found {len(subdirs)} subdirectories in {REC_DIR}")
    
    rec_lines = []
    char_set = set()
    
    for subdir in subdirs:
        subdir_path = os.path.join(REC_DIR, subdir)
        annotation_file = os.path.join(subdir_path, "anotation.txt")
        if not os.path.exists(annotation_file):
            # Check other spelling just in case
            annotation_file = os.path.join(subdir_path, "annotation.txt")
            if not os.path.exists(annotation_file):
                print(f"[WARNING] Annotation file not found in {subdir_path}")
                continue
                
        try:
            with open(annotation_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Split by first whitespace (can be spaces or tabs)
                parts = line.split(None, 1)
                if len(parts) < 2:
                    # Skip empty labels (e.g. "28_2.jpg ")
                    continue
                    
                img_file, text_label = parts
                text_label = text_label.strip()
                if not text_label:
                    continue
                
                # Relativize image path to the rec directory
                rel_img_path = f"{subdir}/{img_file}"
                
                # Verify that the cropped image file actually exists
                full_img_path = os.path.join(REC_DIR, subdir, img_file)
                if not os.path.exists(full_img_path):
                    continue
                    
                # Format: relative_path \t text_label
                formatted_line = f"{rel_img_path}\t{text_label}"
                rec_lines.append(formatted_line)
                
                # Extract characters for the dictionary
                for char in text_label:
                    # Add all characters except space/tabs (if any)
                    if char not in (' ', '\t', '\n', '\r'):
                        char_set.add(char)
                        
        except Exception as e:
            print(f"[WARNING] Failed to parse annotation in {subdir_path}: {e}")
            
    if not rec_lines:
        print("[ERROR] No valid recognition lines parsed!")
        return
        
    # Shuffle and split 9:1
    random.seed(42)
    random.shuffle(rec_lines)
    split_idx = int(len(rec_lines) * 0.9)
    train_lines = rec_lines[:split_idx]
    val_lines = rec_lines[split_idx:]
    
    train_file = os.path.join(OUTPUT_DIR, "rec_train_list.txt")
    val_file = os.path.join(OUTPUT_DIR, "rec_val_list.txt")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for line in train_lines:
            f.write(line + "\n")
            
    with open(val_file, "w", encoding="utf-8") as f:
        for line in val_lines:
            f.write(line + "\n")
            
    # Save dictionary
    dict_file = os.path.join(OUTPUT_DIR, "stamp_dict.txt")
    # Sort for deterministic order
    sorted_chars = sorted(list(char_set))
    with open(dict_file, "w", encoding="utf-8") as f:
        for char in sorted_chars:
            f.write(char + "\n")
        # CRITICAL: Always append an extra empty line at the end so PaddleOCR doesn't miss the last character!
        f.write("\n")
        
    print(f"[OK] Recognition split complete. Train: {len(train_lines)}, Val: {len(val_lines)}")
    print(f"Saved: {train_file} and {val_file}")
    print(f"[OK] Dictionary created with {len(sorted_chars)} characters: {dict_file}")
    print(f"Characters: {''.join(sorted_chars)}")

def main():
    prepare_detection_data()
    prepare_recognition_data()
    print("\nData preparation completed successfully!")

if __name__ == "__main__":
    main()
