import os
import urllib.request
import tarfile
import subprocess
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMP_DIR = os.path.join(PROJECT_ROOT, "tools", "temp_rec_models")
PYTHON_INTERPRETER = os.path.join(PROJECT_ROOT, ".venv_paddle2", "Scripts", "python.exe")
PADDLE_LITE_OPT = os.path.join(PROJECT_ROOT, ".venv_paddle2", "Scripts", "paddle_lite_opt")
ASSETS_REC_DIR = os.path.join(PROJECT_ROOT, "app", "src", "main", "assets", "models", "recognize")
SRC_DICT_PATH = os.path.join(PROJECT_ROOT, "tools", "PaddleOCR", "ppocr", "utils", "ppocr_keys_v1.txt")
DEST_DICT_PATH = os.path.join(ASSETS_REC_DIR, "ch_dict.txt")

MODELS = {
    "tiny": {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv2/chinese/ch_PP-OCRv2_rec_infer.tar",
        "folder": "ch_PP-OCRv2_rec_infer",
        "output_nb": "rec_tiny.nb"
    },
    "small": {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_infer.tar",
        "folder": "ch_PP-OCRv3_rec_infer",
        "output_nb": "rec_small.nb"
    },
    "medium": {
        "url": "https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar",
        "folder": "ch_PP-OCRv4_rec_infer",
        "output_nb": "rec_medium.nb"
    }
}

def download_and_compile():
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)
    if not os.path.exists(ASSETS_REC_DIR):
        os.makedirs(ASSETS_REC_DIR)

    for tier, info in MODELS.items():
        print(f"\n================ Processing {tier.upper()} Recognition Model ================")
        tar_filename = f"{tier}_rec_infer.tar"
        tar_path = os.path.join(TEMP_DIR, tar_filename)
        
        # 1. Download
        print(f"Downloading from {info['url']}...")
        urllib.request.urlretrieve(info['url'], tar_path)
        print(f"Download complete: {tar_path}")
        
        # 2. Extract
        print(f"Extracting {tar_filename}...")
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=TEMP_DIR)
        print("Extraction complete.")
        
        # 3. Locate model files
        extracted_dir = os.path.join(TEMP_DIR, info["folder"])
        # Some tars might have inference.pdmodel, others ch_PP-OCRvX_rec_infer/inference.pdmodel
        model_file = os.path.join(extracted_dir, "inference.pdmodel")
        param_file = os.path.join(extracted_dir, "inference.pdiparams")
        
        if not os.path.exists(model_file) or not os.path.exists(param_file):
            # Try recursive find just in case
            found = False
            for root, dirs, files in os.walk(extracted_dir):
                if "inference.pdmodel" in files and "inference.pdiparams" in files:
                    model_file = os.path.join(root, "inference.pdmodel")
                    param_file = os.path.join(root, "inference.pdiparams")
                    found = True
                    break
            if not found:
                print(f"Error: inference.pdmodel or inference.pdiparams not found in {extracted_dir}")
                continue
        
        # 4. Compile using paddle_lite_opt via Python interpreter
        opt_out_base = os.path.join(TEMP_DIR, f"opt_{tier}")
        print(f"Compiling model to Paddle Lite nb format...")
        cmd = [
            PYTHON_INTERPRETER,
            PADDLE_LITE_OPT,
            f"--model_file={model_file}",
            f"--param_file={param_file}",
            f"--optimize_out={opt_out_base}",
            "--valid_targets=arm"
        ]
        print(f"Running command: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True)
        print(res.stdout)
        if res.stderr:
            print(res.stderr)
            
        # 5. Move compiled file to assets
        compiled_nb_path = opt_out_base + ".nb"
        if os.path.exists(compiled_nb_path):
            dest_path = os.path.join(ASSETS_REC_DIR, info["output_nb"])
            print(f"Moving compiled model to assets: {dest_path}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            shutil.move(compiled_nb_path, dest_path)
            print(f"Successfully deployed {info['output_nb']}")
        else:
            print(f"Error: Compilation failed, output .nb file not found for {tier}")

    # 6. Copy dictionary
    if os.path.exists(SRC_DICT_PATH):
        print(f"\nCopying dictionary to assets: {DEST_DICT_PATH}")
        if os.path.exists(DEST_DICT_PATH):
            os.remove(DEST_DICT_PATH)
        shutil.copy2(SRC_DICT_PATH, DEST_DICT_PATH)
        print("Dictionary deployed successfully.")
    else:
        print(f"Error: Dictionary source not found at {SRC_DICT_PATH}")

    # 7. Clean up temp folder
    try:
        shutil.rmtree(TEMP_DIR)
        print("\nCleaned up temporary directories.")
    except Exception as e:
        print(f"Warning: Failed to clean up temp dir: {e}")

if __name__ == "__main__":
    download_and_compile()
