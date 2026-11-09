#!/bin/bash
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
