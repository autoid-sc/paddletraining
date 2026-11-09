#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Package all plate training assets into a single folder 'gpu_server_plate_training'
and compress it into a .zip file for uploading to the GPU server.
"""
import os
import shutil
import zipfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "gpu_server_plate_training")

def setup_directories():
    dirs = [
        os.path.join(PACKAGE_DIR, "configs"),
        os.path.join(PACKAGE_DIR, "setup")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    print("[OK] Created directory structure under gpu_server_plate_training/")

def copy_assets():
    # 1. Copy prepare_ccpd_data.py
    shutil.copy2(
        os.path.join(PROJECT_ROOT, "tools", "prepare_ccpd_data.py"),
        os.path.join(PACKAGE_DIR, "prepare_ccpd_data.py")
    )
    print("[OK] Copied prepare_ccpd_data.py script.")

    # 2. Copy config files
    shutil.copy2(
        os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "configs", "det", "PP-OCRv6", "plate_det_tiny_gpu.yml"),
        os.path.join(PACKAGE_DIR, "configs", "plate_det_tiny_gpu.yml")
    )
    shutil.copy2(
        os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "configs", "rec", "PP-OCRv6", "plate_rec_small_gpu.yml"),
        os.path.join(PACKAGE_DIR, "configs", "plate_rec_small_gpu.yml")
    )
    print("[OK] Copied GPU-optimized YAML configuration files.")

def generate_bash_scripts():
    # 1. Generate install_plate_requirements.sh
    install_script = """#!/bin/bash
# Exit on error
set -e

echo "=========================================================="
echo "      Setting up PaddleOCR License Plate Server Env"
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
pip install albumentations tqdm opencv-python

# 3. Deploy configurations
echo "Deploying custom plate configuration files..."
mkdir -p PaddleOCR/configs/det/PP-OCRv6/
mkdir -p PaddleOCR/configs/rec/PP-OCRv6/
cp configs/plate_det_tiny_gpu.yml PaddleOCR/configs/det/PP-OCRv6/
cp configs/plate_rec_small_gpu.yml PaddleOCR/configs/rec/PP-OCRv6/

# 4. Download pre-trained models on server
echo "Creating pretrained models directory at /data/plate_train_data/pretrained/"
mkdir -p /data/plate_train_data/pretrained/

echo "Downloading tiny detection pre-trained weights..."
if [ ! -f "/data/plate_train_data/pretrained/detect_tiny_pretrained.pdparams" ]; then
    wget -O /data/plate_train_data/pretrained/detect_tiny_pretrained.pdparams https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams
else
    echo "Tiny detection weights already exist."
fi

echo "Downloading small recognition pre-trained weights..."
if [ ! -f "/data/plate_train_data/pretrained/PP-OCRv6_small_rec_pretrained.pdparams" ]; then
    wget -O /data/plate_train_data/pretrained/PP-OCRv6_small_rec_pretrained.pdparams https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams
else
    echo "Small recognition weights already exist."
fi

echo "=========================================================="
echo "[SUCCESS] Environment setup complete!"
echo "Run: ./run_gpu_plate_training.sh to start dataset processing & training."
echo "=========================================================="
"""

    # 2. Generate run_gpu_plate_training.sh
    run_script = """#!/bin/bash
set -e

echo "=========================================================="
echo "        CCPD Dataset Preprocessing & GPU Training"
echo "=========================================================="

# 1. Ask for dataset directories & max limits
read -p "Enter path to CCPD2019 dataset root [/data/CCPD2019]: " CCPD_DIR
CCPD_DIR=${CCPD_DIR:-/data/CCPD2019}

read -p "Enter path to output processed train data [/data/plate_train_data]: " OUT_DIR
OUT_DIR=${OUT_DIR:-/data/plate_train_data}

read -p "Enter max detection training images to process [15000]: " MAX_DET
MAX_DET=${MAX_DET:-15000}

read -p "Enter max recognition training images to process [30000]: " MAX_REC
MAX_REC=${MAX_REC:-30000}

# 2. Run Python preprocessing script
echo "Running prepare_ccpd_data.py..."
python3 prepare_ccpd_data.py --ccpd_dir "$CCPD_DIR" --output_dir "$OUT_DIR" --max_det "$MAX_DET" --max_rec "$MAX_REC"

# 3. Detect available GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "Detected $GPU_COUNT NVIDIA GPU(s) on the system."
else
    GPU_COUNT=1
    echo "nvidia-smi not found. Defaulting to 1 GPU."
fi

GPUS_LIST=$(seq -s, 0 $((GPU_COUNT-1)))
echo "Using GPUs: $GPUS_LIST"

echo "----------------------------------------------------------"
echo "Select training option:"
echo "  [1] Train DBNet Detection (PP-OCRv6 tiny, 100 epochs)"
echo "  [2] Train SVTR_LCNet Recognition (PP-OCRv6 small, 100 epochs)"
echo "  [3] Train BOTH (Sequential)"
echo "  [4] Exit"
echo "----------------------------------------------------------"
read -p "Enter choice [1-4]: " CHOICE

cd PaddleOCR

case $CHOICE in
    1)
        echo "Starting Detection model training on GPUs: $GPUS_LIST..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml
        ;;
    2)
        echo "Starting Recognition model training on GPUs: $GPUS_LIST..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml
        ;;
    3)
        echo "Starting Detection model training..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml
        echo "Detection training complete. Starting Recognition model training..."
        python3 -m paddle.distributed.launch --gpus "$GPUS_LIST" tools/train.py -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml
        ;;
    *)
        echo "Exiting."
        exit 0
        ;;
esac
"""

    # Write scripts with Linux LF line endings (\n) in binary mode
    install_script_path = os.path.join(PACKAGE_DIR, "setup", "install_plate_requirements.sh")
    with open(install_script_path, "wb") as f:
        f.write(install_script.encode("utf-8").replace(b"\r\n", b"\n"))
    print(f"[OK] Generated {install_script_path} with LF endings.")

    run_script_path = os.path.join(PACKAGE_DIR, "run_gpu_plate_training.sh")
    with open(run_script_path, "wb") as f:
        f.write(run_script.encode("utf-8").replace(b"\r\n", b"\n"))
    print(f"[OK] Generated {run_script_path} with LF endings.")

def zip_package():
    zip_path = os.path.join(PROJECT_ROOT, "gpu_server_plate_training.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print("Zipping package to gpu_server_plate_training.zip...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(PACKAGE_DIR):
            for file in files:
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(file_path, PACKAGE_DIR)
                zipf.write(file_path, rel_path)
                
    print(f"[SUCCESS] Zip archive created at: {zip_path}")

def main():
    print("=" * 60)
    print("Packaging GPU Server Plate Training Package...")
    print("=" * 60)
    setup_directories()
    copy_assets()
    generate_bash_scripts()
    zip_package()
    print("=" * 60)
    print("[SUCCESS] Package is fully ready for upload!")
    print("=" * 60)

if __name__ == "__main__":
    main()
