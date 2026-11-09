#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One-click script to export trained license plate detection and recognition models,
convert them to PaddleLite .nb format, and deploy directly to Android assets for the "Plate" model tier.
"""
import os
import sys
import shutil
import glob
import subprocess

try:
    import paddle.base.proto.framework_pb2 as framework_pb2
except ImportError:
    try:
        import paddle.fluid.proto.framework_pb2 as framework_pb2
    except ImportError:
        import paddle.framework.proto.framework_pb2 as framework_pb2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
PYTHON_EXE = os.path.join(PROJECT_ROOT, ".venv_paddle2", "Scripts", "python.exe")
# Use venv_paddle37's paddle_lite_opt (v2.12) for .nb conversion.
# venv_paddle2's paddle_lite_opt (v2.13-rc) has a segfault bug when optimizing
# PP-OCRv6 rec models (crashes during optimization pass).
PADDLE_LITE_OPT = os.path.join(PROJECT_ROOT, ".venv_paddle37", "Scripts", "paddle_lite_opt")
PADDLE_LITE_OPT_PYTHON = os.path.join(PROJECT_ROOT, ".venv_paddle37", "Scripts", "python.exe")
TOOLS_DIR = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR")

# Configs and outputs
DET_CONFIG = "configs/det/PP-OCRv6/plate_det_tiny.yml"
REC_CONFIG = "configs/rec/PP-OCRv6/plate_rec_small.yml"

DET_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plate_det_tiny")
REC_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "plate_rec_small")

# Temp inference dirs
DET_TEMP_INF = os.path.join(PROJECT_ROOT, "output", "plate_det_tiny_inference")
REC_TEMP_INF = os.path.join(PROJECT_ROOT, "output", "plate_rec_small_inference")

# Asset deployment targets
ASSETS_DET_DIR = os.path.join(PROJECT_ROOT, "app", "src", "main", "assets", "models", "detect")
ASSETS_REC_DIR = os.path.join(PROJECT_ROOT, "app", "src", "main", "assets", "models", "recognize")

# Custom plate dict location
DICT_SRC_OPTIONS = [
    os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "plate", "plate_dict.txt"),
    os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "plate", "plate_dict.txt"),
    os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "train_det_rec", "plate_dict.txt")
]
DICT_DEST = os.path.join(ASSETS_REC_DIR, "plate_dict.txt")

def find_best_checkpoint(output_dir):
    best_path = os.path.join(output_dir, "best_accuracy")
    if os.path.exists(best_path + ".pdparams"):
        return best_path
        
    latest_path = os.path.join(output_dir, "latest")
    if os.path.exists(latest_path + ".pdparams"):
        return latest_path
        
    pattern = os.path.join(output_dir, "*.pdparams")
    any_files = glob.glob(pattern)
    if any_files:
        any_files.sort(key=os.path.getmtime, reverse=True)
        return any_files[0].replace(".pdparams", "")
        
    return None

def fix_pdmodel_attrs(pdmodel_path):
    """Patch missing default attributes in pdmodel for PaddleLite compatibility.

    Fixes:
    - swish op: adds beta=1.0 if missing
    - hard_sigmoid op: adds slope=0.2, offset=0.5 if missing
    """
    if not os.path.exists(pdmodel_path):
        print(f"[ERROR] pdmodel not found: {pdmodel_path}")
        return False

    program = framework_pb2.ProgramDesc()
    with open(pdmodel_path, "rb") as f:
        program.ParseFromString(f.read())

    swish_count = 0
    hs_count = 0

    for block in program.blocks:
        for op in block.ops:
            if op.type == "swish":
                has_beta = any(attr.name == "beta" for attr in op.attrs)
                if not has_beta:
                    attr = op.attrs.add()
                    attr.name = "beta"
                    attr.type = 1  # FLOAT
                    attr.f = 1.0
                    swish_count += 1
            elif op.type in ["hard_sigmoid", "hardsigmoid"]:
                has_slope = any(attr.name == "slope" for attr in op.attrs)
                if not has_slope:
                    attr = op.attrs.add()
                    attr.name = "slope"
                    attr.type = 1
                    attr.f = 0.2
                    hs_count += 1
                has_offset = any(attr.name == "offset" for attr in op.attrs)
                if not has_offset:
                    attr = op.attrs.add()
                    attr.name = "offset"
                    attr.type = 1
                    attr.f = 0.5
                    hs_count += 1

    with open(pdmodel_path, "wb") as f:
        f.write(program.SerializeToString())

    print(f"[OK] Patched {swish_count} swish op(s), {hs_count} hard_sigmoid attr(s) in {os.path.basename(pdmodel_path)}")
    return True

def verify_rec_model(inference_dir, dict_path):
    """Run Python verification on exported rec model using CCPD samples."""
    print(f"\n[INFO] Verifying rec model on CCPD samples...")
    ccpd_dir = os.path.join(PROJECT_ROOT, "docs", "CCPD2019", "ccpd_base")
    verify_script = os.path.join(SCRIPT_DIR, "verify_plate_inference.py")

    cmd = [
        PYTHON_EXE, verify_script,
        inference_dir, dict_path, ccpd_dir
    ]

    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, shell=True)
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode == 0:
            print("[OK] Rec model verification passed.")
            return True
        else:
            print(f"[ERROR] Rec model verification failed (exit code {result.returncode}).")
            return False
    except Exception as e:
        print(f"[ERROR] Verification script execution failed: {e}")
        return False

def export_model(name, config_path, checkpoint_prefix, save_inference_dir):
    if os.path.exists(save_inference_dir):
        shutil.rmtree(save_inference_dir)
        
    print(f"\n[INFO] Exporting {name} model from checkpoint '{checkpoint_prefix}'...")
    
    cmd = [
        PYTHON_EXE, "tools/export_model.py",
        "-c", config_path,
        "-o",
        f"Global.checkpoints={checkpoint_prefix}",
        f"Global.save_inference_dir={save_inference_dir}",
        "Global.export_with_pir=False",
        "Global.infer_mode=True"
    ]
    
    try:
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(cmd, cwd=TOOLS_DIR, env=env, check=True, capture_output=True, text=True, shell=True)
        print(f"[OK] Exported successfully to: {save_inference_dir}")

        # Patch pdmodel for PaddleLite compatibility
        pdmodel_path = os.path.join(save_inference_dir, "inference.pdmodel")
        fix_pdmodel_attrs(pdmodel_path)

        return True
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to export {name}. Exit code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}\nSTDERR:\n{e.stderr}")
        return False

def optimize_model(name, exported_dir, output_prefix):
    pdmodel = os.path.join(exported_dir, "inference.pdmodel")
    pdiparams = os.path.join(exported_dir, "inference.pdiparams")
    
    if not os.path.exists(pdmodel) or not os.path.exists(pdiparams):
        print(f"[ERROR] Exported inference files not found in {exported_dir}")
        return None
        
    print(f"[INFO] Optimizing {name} to PaddleLite .nb format...")
    
    cmd = [
        PADDLE_LITE_OPT_PYTHON, PADDLE_LITE_OPT,
        "--model_file", pdmodel,
        "--param_file", pdiparams,
        "--optimize_out", output_prefix,
        "--valid_targets", "arm",
        "--optimize_out_type", "naive_buffer"
    ]
    
    try:
        # Call paddle_lite_opt through bash to match the direct command-line
        # invocation that works. When called via subprocess.run directly
        # (CreateProcess), venv_paddle37's Python crashes with
        # STATUS_STACK_BUFFER_OVERRUN on the rec model. Running through bash
        # uses the shebang and environment setup that works.
        import shutil as _shutil
        bash_path = _shutil.which("bash") or "bash"
        cmd_str = (
            f'"{PADDLE_LITE_OPT}" '
            f'--model_file "{pdmodel}" '
            f'--param_file "{pdiparams}" '
            f'--optimize_out "{output_prefix}" '
            f'--valid_targets arm '
            f'--optimize_out_type naive_buffer'
        )
        result = subprocess.run([bash_path, "-c", cmd_str], check=True)

        generated_nb = output_prefix + ".nb"
        if os.path.exists(generated_nb):
            print(f"[OK] Successfully generated .nb file: {generated_nb}")
            return generated_nb
        else:
            print(f"[ERROR] .nb file not found after optimization: {generated_nb}")
            return None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Failed to convert nb. Exit code: {e.returncode}")
        return None
    except Exception as e:
        print(f"[ERROR] Failed to convert nb: {e}")
        
    return None

def main():
    print("==================================================================")
    print("      PP-OCRv6 License Plate Model Local Export & Deploy")
    print("==================================================================")
    
    os.makedirs(ASSETS_DET_DIR, exist_ok=True)
    os.makedirs(ASSETS_REC_DIR, exist_ok=True)

    # 1. Detection Model
    print("\n------------------ [1] Plate Detection Model ------------------")
    det_ckpt = find_best_checkpoint(DET_OUTPUT_DIR)
    det_success = False
    if not det_ckpt:
        print(f"[WARNING] No detection checkpoint found in {DET_OUTPUT_DIR}. Skipping detection export.")
    else:
        print(f"[OK] Found detection checkpoint: {det_ckpt}.pdparams")
        if export_model("Detection", DET_CONFIG, det_ckpt, DET_TEMP_INF):
            opt_det_prefix = os.path.join(DET_TEMP_INF, "plate_det_small")
            nb_det_path = optimize_model("Detection", DET_TEMP_INF, opt_det_prefix)
            if nb_det_path:
                dest = os.path.join(ASSETS_DET_DIR, "plate_det_small.nb")
                shutil.copy2(nb_det_path, dest)
                print(f"[SUCCESS] Deployed: {nb_det_path} -> {dest}")
                det_success = True

    # 2. Recognition Model
    print("\n----------------- [2] Plate Recognition Model -----------------")
    rec_ckpt = find_best_checkpoint(REC_OUTPUT_DIR)
    rec_success = False
    if not rec_ckpt:
        print(f"[WARNING] No recognition checkpoint found in {REC_OUTPUT_DIR}. Skipping recognition export.")
    else:
        print(f"[OK] Found recognition checkpoint: {rec_ckpt}.pdparams")
        if export_model("Recognition", REC_CONFIG, rec_ckpt, REC_TEMP_INF):
            # Verify rec model before converting to .nb
            dict_path = None
            for src in DICT_SRC_OPTIONS:
                if os.path.exists(src):
                    dict_path = src
                    break
            if dict_path:
                if not verify_rec_model(REC_TEMP_INF, dict_path):
                    print("[WARNING] Rec verification failed, but continuing with .nb conversion...")
            else:
                print("[WARNING] Dict not found, skipping rec verification.")

            opt_rec_prefix = os.path.join(REC_TEMP_INF, "plate_rec_small")
            nb_rec_path = optimize_model("Recognition", REC_TEMP_INF, opt_rec_prefix)
            if nb_rec_path:
                dest = os.path.join(ASSETS_REC_DIR, "plate_rec_small.nb")
                shutil.copy2(nb_rec_path, dest)
                print(f"[SUCCESS] Deployed: {nb_rec_path} -> {dest}")
                rec_success = True

    # 3. Deploy Dictionary
    print("\n--------------------- [3] Deploy Dictionary ---------------------")
    dict_success = False
    for src in DICT_SRC_OPTIONS:
        if os.path.exists(src):
            shutil.copy2(src, DICT_DEST)
            print(f"[SUCCESS] Deployed plate dict: {src} -> {DICT_DEST}")
            dict_success = True
            break
            
    if not dict_success:
        print(f"[WARNING] Plate dict source not found in any of: {DICT_SRC_OPTIONS}")

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
    print(f"Plate Detection Model (`plate_det_small.nb`):   {'[SUCCESS]' if det_success else '[SKIPPED / FAILED]'}")
    print(f"Plate Recognition Model (`plate_rec_small.nb`): {'[SUCCESS]' if rec_success else '[SKIPPED / FAILED]'}")
    print(f"Plate Dictionary (`plate_dict.txt`):            {'[SUCCESS]' if dict_success else '[SKIPPED / FAILED]'}")
    print("==================================================================")

if __name__ == "__main__":
    main()
