#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download PP-OCRv6 small pre-trained weights for fine-tuning.
"""
import os
import urllib.request
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "pretrained")

WEIGHTS = {
    "PP-OCRv6_small_det_pretrained.pdparams": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_det_pretrained.pdparams",
    "PP-OCRv6_small_rec_pretrained.pdparams": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams"
}

def download_file(url, filepath):
    filename = os.path.basename(filepath)
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 1 * 1024 * 1024:
            print(f"[OK] Pretrained weight already exists: {filename}")
            return True
            
    print(f"\nDownloading {filename} from {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            total_size = int(response.info().get('Content-Length', 0))
            downloaded = 0
            chunk_size = 512 * 1024
            
            with open(filepath, 'wb') as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    if total_size > 0:
                        percent = downloaded * 100.0 / total_size
                        sys.stdout.write(f"\rProgress: {percent:.1f}% ({downloaded:,}/{total_size:,} bytes)")
                        sys.stdout.flush()
                        
            if downloaded == total_size or total_size == 0:
                print(f"\n[OK] {filename} downloaded successfully.")
                return True
            else:
                print(f"\n[WARNING] Byte count mismatch: {downloaded} vs {total_size}. Retrying...")
    except Exception as e:
        print(f"\n[ERROR] Failed to download {filename}: {e}")
        
    return False

def main():
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    print("=" * 60)
    print("PP-OCRv6 Pre-trained Weights Downloader")
    print("=" * 60)
    
    success = True
    for filename, url in WEIGHTS.items():
        filepath = os.path.join(PRETRAINED_DIR, filename)
        if not download_file(url, filepath):
            success = False
            
    print("=" * 60)
    if success:
        print(f"[SUCCESS] All pre-trained weights are ready in {PRETRAINED_DIR}!")
    else:
        print("[ERROR] Some pre-trained weights failed to download.")
        sys.exit(1)

if __name__ == "__main__":
    main()
