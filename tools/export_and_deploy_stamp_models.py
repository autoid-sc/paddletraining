#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click script to export trained stamp detection and recognition models,
convert them to PaddleLite .nb format, and deploy to Android main assets.
"""
import os
import sys
import shutil
import glob
import subprocess

# Define root paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv_paddle2", "Scripts", "python.exe")
PADDLE_LITE_OPT = os.path.join(PROJECT_ROOT, ".venv_paddle2", "Scripts", "paddle_lite_opt")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR")

# Configs and outputs
DET_CONFIG = "configs/det/PP-OCRv6/stamp_det_tiny.yml"
REC_CONFIG = "configs/rec/PP-OCRv6/stamp_rec_small.yml"

DET_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "stamp_det_tiny")
REC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "stamp_rec_small")

# Temp inference dirs
DET_TEMP_INF = os.path.join(PROJECT_ROOT, "output", "stamp_det_tiny_inference")
REC_TEMP_INF = os.path.join(PROJECT_ROOT, "output", "stamp_rec_small_inference")

# Asset deployment targets
ASSETS_DET_DIR = os.path.join(PROJECT_ROOT, "app", "src", "main", "assets", "models", "detect")
ASSETS_REC_DIR = os.path.join(PROJECT_ROOT, "app", "src", "main", "assets", "models", "recognize")

DICT_SRC = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "stamp_dict.txt")
DICT_DEST = os.path.join(ASSETS_REC_DIR, "ch_dict.txt")

def find_best_checkpoint(output_dir):
    """
    Search for the best checkpoint file prefix in output directory.
    Checks:
    1. best_accuracy.pdparams
    2. latest.pdparams
    3. Any other *.pdparams (sorted by file modify time, return latest)
    """
    best_path = os.path.join(output_dir, "best_accuracy")
    if os.path.exists(best_path + ".pdparams"):
        return best_path
        
    latest_path = os.path.join(output_dir, "latest")
    if os.path.exists(latest_path + ".pdparams"):
        return latest_path
        
    # Search for any epoch weights
    pattern = os.path.join(output_dir, "*_epoch.pdparams")
    epoch_files = glob.glob(pattern)
    if epoch_files:
        # Sort by modification time to get the newest trained epoch
        epoch_files.sort(key=os.path.getmtime, reverse=True)
        newest_file = epoch_files[0]
        # Remove suffix
        return newest_file.replace(".pdparams", "")
        
    pattern_any = os.path.join(output_dir, "*.pdparams")
    any_files = glob.glob(pattern_any)
    if any_files:
        any_files.sort(key=os.path.getmtime, reverse=True)
        return any_files[0].replace(".pdparams", "")
        
    return None

def export_model(name, config_path, checkpoint_prefix, save_inference_dir):
    """
    Run tools/export_model.py to export to fluid format
    """
    if os.path.exists(save_inference_dir):
        shutil.rmtree(save_inference_dir)
        
    print(f"\n[INFO] Exporting {name} model from checkpoint '{checkpoint_prefix}'...")
    
    cmd = [
        PYTHON_EXE, "tools/export_model.py",
        "-c", config_path,
        "-o",
        f"Global.checkpoints={checkpoint_prefix}",
        f"Global.save_inference_dir={save_inference_dir}",
        "Global.export_with_pir=False"
    ]
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        # Run inside tools/PaddleOCR
        result = subprocess.run(cmd, cwd=TOOLS_DIR, env=env, check=True, capture_output=True, text=True, shell=True)
        print(f"[OK] Exported successfully to: {save_inference_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to export {name}. Exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        return False

def optimize_model(name, exported_dir, output_prefix):
    """
    Compile/optimize the exported model to PaddleLite .nb format
    """
    pdmodel = os.path.join(exported_dir, "inference.pdmodel")
    pdiparams = os.path.join(exported_dir, "inference.pdiparams")
    
    if not os.path.exists(pdmodel) or not os.path.exists(pdiparams):
        print(f"[ERROR] Exported inference files not found in {exported_dir}")
        return None
        
    print(f"[INFO] Optimizing {name} to PaddleLite .nb format...")
    
    cmd = [
        PYTHON_EXE, PADDLE_LITE_OPT,
        "--model_file", pdmodel,
        "--param_file", pdiparams,
        "--optimize_out", output_prefix,
        "--valid_targets", "arm",
        "--optimize_out_type", "naive_buffer"
    ]
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, env=env, check=True, capture_output=True, text=True, shell=True)
        # Log standard output from paddle_lite_opt
        print(result.stdout)
        
        generated_nb = output_prefix + ".nb"
        if os.path.exists(generated_nb):
            print(f"[OK] Successfully generated .nb file: {generated_nb}")
            return generated_nb
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert nb. Exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        
    return None

def main():
    print("==================================================================")
    print("          PP-OCRv6 Stamp Model Local Export & Deploy")
    print("==================================================================")
    
    # 0. Ensure environment paths exist
    if not os.path.exists(PYTHON_EXE):
        print(f"[ERROR] Python interpreter not found at: {PYTHON_EXE}")
        sys.exit(1)
    if not os.path.exists(PADDLE_LITE_OPT):
        print(f"[ERROR] paddle_lite_opt not found at: {PADDLE_LITE_OPT}")
        sys.exit(1)
    if not os.path.exists(TOOLS_DIR):
        print(f"[ERROR] PaddleOCR path does not exist: {TOOLS_DIR}")
        sys.exit(1)
        
    os.makedirs(ASSETS_DET_DIR, exist_ok=True)
    os.makedirs(ASSETS_REC_DIR, exist_ok=True)

    # 1. Process Detection Model
    print("\n------------------ [1] DBNet Detection Model ------------------")
    det_ckpt = find_best_checkpoint(DET_OUTPUT_DIR)
    det_success = False
    if not det_ckpt:
        print(f"[WARNING] No detection checkpoint found in {DET_OUTPUT_DIR}. Skipping detection export.")
    else:
        print(f"[OK] Found detection checkpoint: {det_ckpt}.pdparams")
        if export_model("Detection", DET_CONFIG, det_ckpt, DET_TEMP_INF):
            opt_det_prefix = os.path.join(DET_TEMP_INF, "detect_small")
            nb_det_path = optimize_model("Detection", DET_TEMP_INF, opt_det_prefix)
            if nb_det_path:
                dest = os.path.join(ASSETS_DET_DIR, "detect_small.nb")
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.copy2(nb_det_path, dest)
                print(f"[SUCCESS] Deployed: {nb_det_path} -> {dest}")
                det_success = True
                
    # 2. Process Recognition Model
    print("\n----------------- [2] SVTR_LCNet Recognition Model -----------------")
    rec_ckpt = find_best_checkpoint(REC_OUTPUT_DIR)
    rec_success = False
    if not rec_ckpt:
        print(f"[WARNING] No recognition checkpoint found in {REC_OUTPUT_DIR}. Skipping recognition export.")
    else:
        print(f"[OK] Found recognition checkpoint: {rec_ckpt}.pdparams")
        if export_model("Recognition", REC_CONFIG, rec_ckpt, REC_TEMP_INF):
            opt_rec_prefix = os.path.join(REC_TEMP_INF, "rec_small")
            nb_rec_path = optimize_model("Recognition", REC_TEMP_INF, opt_rec_prefix)
            if nb_rec_path:
                dest = os.path.join(ASSETS_REC_DIR, "rec_small.nb")
                if os.path.exists(dest):
                    os.remove(dest)
                shutil.copy2(nb_rec_path, dest)
                print(f"[SUCCESS] Deployed: {nb_rec_path} -> {dest}")
                rec_success = True

    # 3. Deploy Dictionary
    print("\n--------------------- [3] Deploy Dictionary ---------------------")
    if os.path.exists(DICT_SRC):
        if os.path.exists(DICT_DEST):
            os.remove(DICT_DEST)
        shutil.copy2(DICT_SRC, DICT_DEST)
        print(f"[SUCCESS] Deployed dict: {DICT_SRC} -> {DICT_DEST}")
        dict_success = True
    else:
        print(f"[WARNING] Dict source not found at: {DICT_SRC}")
        dict_success = False

    # 4. Clean up temp directories
    print("\n----------------------- [4] Cleaning Up -----------------------")
    for temp_dir in [DET_TEMP_INF, REC_TEMP_INF]:
        if os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                print(f"[OK] Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"[WARNING] Failed to clean up {temp_dir}: {e}")

    print("\n==================================================================")
    print("                         SUMMARY OF DEPLOYMENT")
    print("==================================================================")
    print(f"Detection Model:   {'[SUCCESS]' if det_success else '[SKIPPED / FAILED]'}")
    print(f"Recognition Model: {'[SUCCESS]' if rec_success else '[SKIPPED / FAILED]'}")
    print(f"Dictionary File:   {'[SUCCESS]' if dict_success else '[SKIPPED / FAILED]'}")
    print("==================================================================")
    
if __name__ == "__main__":
    main()
