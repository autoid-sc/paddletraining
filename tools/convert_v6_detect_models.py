#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download PP-OCRv6 (tiny, small, medium) detection models,
export them to fluid format, convert to nb format, and deploy to assets.
"""
import os
import sys
import shutil
import urllib.request
import time
import subprocess

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv_paddle2", "Scripts", "python.exe")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR")

# paddle_lite_opt script path
PADDLE_LITE_OPT = os.path.join(PROJECT_ROOT, ".venv_paddle2", "Scripts", "paddle_lite_opt")

WEIGHTS = {
    "detect_tiny": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams",
    "detect_small": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_det_pretrained.pdparams",
    "detect_medium": "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_medium_det_pretrained.pdparams"
}

CONFIGS = {
    "detect_tiny": "configs/det/PP-OCRv6/PP-OCRv6_tiny_det.yml",
    "detect_small": "configs/det/PP-OCRv6/PP-OCRv6_small_det.yml",
    "detect_medium": "configs/det/PP-OCRv6/PP-OCRv6_medium_det.yml"
}

PRETRAINED_DIR = os.path.join(PROJECT_ROOT, "models", "pretrained")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "infer_v6_detect_exported")
ASSETS_DET_DIR = os.path.join(PROJECT_ROOT, "app", "src", "main", "assets", "models", "detect")

os.makedirs(PRETRAINED_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DET_DIR, exist_ok=True)

def download_file(url, filepath, max_retries=3):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    filename = os.path.basename(filepath)
    
    if os.path.exists(filepath):
        if os.path.getsize(filepath) > 1 * 1024 * 1024:
            print(f"[OK] File already exists: {filename}")
            return True
            
    for attempt in range(1, max_retries + 1):
        print(f"\nDownloading {filename} (attempt {attempt}/{max_retries})...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as response:
                total_size = int(response.info().get('Content-Length', 0))
                downloaded = 0
                chunk_size = 256 * 1024
                
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
                    print(f"\n[OK] {filename} download complete")
                    return True
                else:
                    print(f"\n[WARNING] Byte count mismatch, redownloading")
        except Exception as e:
            print(f"\n[WARNING] Download failed: {e}")
            
        time.sleep(3)
        
    return False

def export_model(name, pdparams_path):
    config_rel_path = CONFIGS[name]
    pretrained_arg = pdparams_path.replace(".pdparams", "")
    save_inference_dir = os.path.join(OUTPUT_DIR, name)
    
    if os.path.exists(save_inference_dir):
        shutil.rmtree(save_inference_dir)
        
    print(f"\nExporting {name} model to fluid format...")
    
    cmd = [
        PYTHON_EXE, "tools/export_model.py",
        "-c", config_rel_path,
        "-o",
        f"Global.pretrained_model={pretrained_arg}",
        f"Global.save_inference_dir={save_inference_dir}",
        "Global.export_with_pir=False"
    ]
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, cwd=TOOLS_DIR, env=env, check=True, capture_output=True, text=True, shell=True)
        print(f"[OK] Exported successfully to: {save_inference_dir}")
        return save_inference_dir
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to export {name}. Code: {e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return None

def optimize_model(name, exported_dir):
    pdmodel = os.path.join(exported_dir, "inference.pdmodel")
    pdiparams = os.path.join(exported_dir, "inference.pdiparams")
    
    if not os.path.exists(pdmodel) or not os.path.exists(pdiparams):
        print(f"[ERROR] Exported files not found in {exported_dir}")
        return None
        
    nb_prefix = os.path.join(OUTPUT_DIR, f"opt_{name}")
    
    print(f"Optimizing {name} to PaddleLite nb format...")
    cmd = [
        PYTHON_EXE, PADDLE_LITE_OPT,
        "--model_file", pdmodel,
        "--param_file", pdiparams,
        "--optimize_out", nb_prefix,
        "--valid_targets", "arm",
        "--optimize_out_type", "naive_buffer"
    ]
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, shell=True)
        print(result.stdout)
        
        generated_nb = nb_prefix + ".nb"
        if os.path.exists(generated_nb):
            print(f"[OK] Nb converted successfully: {generated_nb}")
            return generated_nb
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert nb. Code: {e.returncode}\nSTDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        
    return None

def main():
    print("=" * 60)
    print("PP-OCRv6 Detection Models Export & Conversion Script")
    print("=" * 60)
    
    nb_files = {}
    
    for name, url in WEIGHTS.items():
        pdparams_path = os.path.join(PRETRAINED_DIR, f"{name}_pretrained.pdparams")
        
        # 1. Download pdparams
        if not download_file(url, pdparams_path):
            print(f"\n[ERROR] Failed to download {name} weights!")
            sys.exit(1)
            
        # 2. Export fluid model
        exported_dir = export_model(name, pdparams_path)
        if not exported_dir:
            print(f"\n[ERROR] Failed to export {name}!")
            sys.exit(1)
            
        # 3. Optimize to nb
        nb_path = optimize_model(name, exported_dir)
        if not nb_path:
            print(f"\n[ERROR] Failed to optimize {name}!")
            sys.exit(1)
            
        nb_files[name] = nb_path
        
    print("\n" + "=" * 60)
    print("Deploying optimized models to assets...")
    print("=" * 60)
    
    for key, name in [("detect_tiny", "detect_tiny"), ("detect_small", "detect_small"), ("detect_medium", "detect_medium")]:
        src = nb_files[key]
        dest = os.path.join(ASSETS_DET_DIR, f"{name}.nb")
        shutil.copy2(src, dest)
        print(f"[OK] Deployed: {src} -> {dest}")
        
    print("\n[SUCCESS] All PP-OCRv6 detection models optimized and deployed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
