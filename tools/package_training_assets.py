#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package all training assets into a single folder 'gpu_server_training'
and generate helper bash scripts with LF line endings.
"""
import os
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "gpu_server_training")

def setup_directories():
    dirs = [
        os.path.join(PACKAGE_DIR, "configs"),
        os.path.join(PACKAGE_DIR, "setup"),
        os.path.join(PACKAGE_DIR, "models", "pretrained"),
        os.path.join(PACKAGE_DIR, "train_det_rec")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Created directory structure under gpu_server_training/")

def copy_assets():
    # 1. Copy pre-trained weights
    pretrained_src = os.path.join(PROJECT_ROOT, "models", "pretrained")
    pretrained_dest = os.path.join(PACKAGE_DIR, "models", "pretrained")
    for f in os.listdir(pretrained_src):
        if f.endswith(".pdparams"):
            shutil.copy2(os.path.join(pretrained_src, f), os.path.join(pretrained_dest, f))
    print("[OK] Copied pretrained weights.")

    # 2. Copy dataset text files
    dataset_src = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec")
    dataset_dest = os.path.join(PACKAGE_DIR, "train_det_rec")
    text_files = [
        "det_train_list.txt",
        "det_val_list.txt",
        "rec_train_list.txt",
        "rec_val_list.txt",
        "stamp_dict.txt"
    ]
    for f in text_files:
        src_file = os.path.join(dataset_src, f)
        if os.path.exists(src_file):
            shutil.copy2(src_file, os.path.join(dataset_dest, f))
    print("[OK] Copied dataset train/val lists and dict.")

    # 3. Copy datasets directories
    det_img_src = os.path.join(dataset_src, "det")
    det_img_dest = os.path.join(dataset_dest, "det")
    rec_img_src = os.path.join(dataset_src, "rec")
    rec_img_dest = os.path.join(dataset_dest, "rec")

    print("Copying det images (this might take a few seconds)...")
    if os.path.exists(det_img_dest):
        shutil.rmtree(det_img_dest)
    shutil.copytree(det_img_src, det_img_dest)
    
    print("Copying rec images (this might take a few seconds)...")
    if os.path.exists(rec_img_dest):
        shutil.rmtree(rec_img_dest)
    shutil.copytree(rec_img_src, rec_img_dest)
    
    print("[OK] Copied image folders.")

def generate_bash_scripts():
    # 1. Generate install_requirements.sh
    install_script = """#!/bin/bash
# Exit on error
set -e

echo "=========================================================="
echo "          Setting up PaddleOCR Server Environment"
echo "=========================================================="

# 1. Clone PaddleOCR repository
if [ ! -d "PaddleOCR" ]; then
    echo "Cloning official PaddleOCR repository..."
    git clone https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR
else
    echo "PaddleOCR directory already exists. Skipping clone."
fi

# 2. Install Python dependencies
echo "Installing pip requirements..."
pip install -r PaddleOCR/requirements.txt
pip install albumentations

# 3. Deploy configurations and dataset links
echo "Linking configurations, dataset, and pretrained weights..."

# Copy configurations
mkdir -p PaddleOCR/configs/det/PP-OCRv6/
mkdir -p PaddleOCR/configs/rec/PP-OCRv6/
cp configs/stamp_det_tiny_gpu.yml PaddleOCR/configs/det/PP-OCRv6/
cp configs/stamp_rec_small_gpu.yml PaddleOCR/configs/rec/PP-OCRv6/

# Move/link datasets (use symlinks to avoid duplication)
if [ ! -d "PaddleOCR/train_det_rec" ]; then
    ln -s ../train_det_rec PaddleOCR/train_det_rec
    echo "Linked train_det_rec to PaddleOCR."
fi

# Move/link pretrained weights
mkdir -p PaddleOCR/models
if [ ! -d "PaddleOCR/models/pretrained" ]; then
    ln -s ../../models/pretrained PaddleOCR/models/pretrained
    echo "Linked models/pretrained to PaddleOCR."
fi

echo "=========================================================="
echo "[SUCCESS] Environment setup complete!"
echo "You can now run: ./run_gpu_training.sh to start training."
echo "=========================================================="
"""

    # 2. Generate run_gpu_training.sh
    run_script = """#!/bin/bash
set -e

# Change directory to PaddleOCR
cd PaddleOCR

echo "=========================================================="
echo "          Starting GPU Training on Server"
echo "=========================================================="

# Detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT NVIDIA GPU(s) on the system."
else
    GPU_COUNT=1
    echo "nvidia-smi not found. Defaulting to 1 GPU."
fi

# Create a comma-separated list of GPUS based on GPU_COUNT
GPUS_LIST=$(seq -s, 0 $((GPU_COUNT-1)))

echo "Using GPUs: $GPUS_LIST"

echo "----------------------------------------------------------"
echo "Select training option:"
echo "  [1] Train DBNet Detection (PP-OCRv6 tiny, 40 epochs)"
echo "  [2] Train SVTR_LCNet Recognition (PP-OCRv6 small, 100 epochs)"
echo "  [3] Train BOTH sequentially"
echo "----------------------------------------------------------"
read -p "Enter choice [1-3]: " CHOICE

case $CHOICE in
    1)
        echo "Starting Detection model training..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/det/PP-OCRv6/stamp_det_tiny_gpu.yml
        ;;
    2)
        echo "Starting Recognition model training..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/rec/PP-OCRv6/stamp_rec_small_gpu.yml
        ;;
    3)
        echo "Starting Detection model training..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/det/PP-OCRv6/stamp_det_tiny_gpu.yml
        echo "Detection training complete. Starting Recognition model training..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/rec/PP-OCRv6/stamp_rec_small_gpu.yml
        ;;
    *)
        echo "Invalid option. Exiting."
        exit 1
        ;;
esac
"""

    # Write scripts with Linux LF line endings (\n) in binary mode
    install_script_path = os.path.join(PACKAGE_DIR, "setup", "install_requirements.sh")
    with open(install_script_path, "wb") as f:
        f.write(install_script.encode("utf-8").replace(b"\r\n", b"\n"))
    print(f"[OK] Generated {install_script_path} with LF endings.")

    run_script_path = os.path.join(PACKAGE_DIR, "run_gpu_training.sh")
    with open(run_script_path, "wb") as f:
        f.write(run_script.encode("utf-8").replace(b"\r\n", b"\n"))
    print(f"[OK] Generated {run_script_path} with LF endings.")

def main():
    print("=" * 60)
    print("Packaging GPU Server training package...")
    print("=" * 60)
    setup_directories()
    copy_assets()
    generate_bash_scripts()
    print("=" * 60)
    print("[SUCCESS] All assets packaged inside gpu_server_training/ directory!")
    print("=" * 60)

if __name__ == "__main__":
    main()
