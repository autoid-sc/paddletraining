#!/bin/bash
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
