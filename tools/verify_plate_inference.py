#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Verify exported plate recognition inference model on CCPD sample images.

Usage:
    python verify_plate_inference.py <inference_dir> <dict_path> [ccpd_dir]

Exit code: 0 = pass, 1 = fail
"""
import os
import sys
import random
import glob

import cv2
import numpy as np
import paddle.inference


def load_dict(dict_path):
    """Load plate dict, return list of characters (without blank)."""
    chars = []
    with open(dict_path, "rb") as f:
        for line in f.readlines():
            chars.append(line.decode("utf-8").strip("\n").strip("\r\n"))
    return chars


def parse_ccpd_filename(filename):
    """Parse CCPD filename to get plate char indices and bbox.

    CCPD format: area-tilt-bbox-corners-plate_indices-brightness-blur.jpg
    Example: 025-95_113-154&383_386&383-386&348_154&348_154&383_386&383-0_0_15_24_27_21_16_28-128-84.jpg

    Returns (char_indices, bbox) where bbox = (x1, y1, x2, y2).
    """
    name = os.path.splitext(os.path.basename(filename))[0]
    parts = name.split("-")
    if len(parts) < 5:
        return None, None
    # parts[2] = bbox: x1&y1_x2&y2
    bbox_coords = parts[2].split("_")
    if len(bbox_coords) != 2:
        return None, None
    x1, y1 = bbox_coords[0].split("&")
    x2, y2 = bbox_coords[1].split("&")
    bbox = (int(x1), int(y1), int(x2), int(y2))
    # parts[4] = plate char indices: idx0_idx1_...
    char_indices = [int(x) for x in parts[4].split("_")]
    return char_indices, bbox


def decode_ctc(output, char_set):
    """CTC decode: blank at index 0, chars from index 1.

    output: numpy array shape (1, T, C)
    """
    result = []
    pred_idx = output.argmax(axis=2)[0]  # (T,)
    prev = -1
    for idx in pred_idx:
        idx = int(idx)
        if idx != 0 and idx != prev:
            char_idx = idx - 1
            if 0 <= char_idx < len(char_set):
                ch = char_set[char_idx]
                if ch:  # skip empty string entries
                    result.append(ch)
        prev = idx
    return "".join(result)


def resize_norm_img(img, img_h=48, img_w=160):
    """Replicate PaddleOCR resize_norm_img: keep aspect ratio, pad right.

    Returns NCHW float32 array shape (1, 3, img_h, img_w).
    Normalization: (pixel/255 - 0.5) / 0.5
    Channel order: BGR (img is BGR from cv2.imread).
    """
    h, w = img.shape[:2]
    ratio = w / float(h)
    resized_w = int(np.ceil(img_h * ratio))
    if resized_w > img_w:
        resized_w = img_w
    resized = cv2.resize(img, (resized_w, img_h))
    resized = resized.astype("float32")
    # HWC -> CHW, cv2 gives BGR which matches training img_mode: BGR
    resized = resized.transpose((2, 0, 1)) / 255.0
    resized -= 0.5
    resized /= 0.5
    padding = np.zeros((3, img_h, img_w), dtype=np.float32)
    padding[:, :, 0:resized_w] = resized
    return padding[np.newaxis, :]  # NCHW


def main():
    if len(sys.argv) < 3:
        print("Usage: python verify_plate_inference.py <inference_dir> <dict_path> [ccpd_dir]")
        sys.exit(1)

    inference_dir = os.path.abspath(sys.argv[1])
    dict_path = os.path.abspath(sys.argv[2])
    if len(sys.argv) > 3:
        ccpd_dir = os.path.abspath(sys.argv[3])
    else:
        ccpd_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "docs", "CCPD2019", "ccpd_base"
        ))

    model_path = os.path.join(inference_dir, "inference.pdmodel")
    params_path = os.path.join(inference_dir, "inference.pdiparams")
    if not os.path.exists(model_path) or not os.path.exists(params_path):
        print(f"[ERROR] Model files not found in {inference_dir}")
        sys.exit(1)

    char_set = load_dict(dict_path)
    print(f"[INFO] Dict loaded: {len(char_set)} chars")

    # Load model
    print(f"[INFO] Loading model from {inference_dir}...")
    config = paddle.inference.Config(model_path, params_path)
    config.disable_gpu()
    config.disable_glog_info()
    predictor = paddle.inference.create_predictor(config)

    input_names = predictor.get_input_names()
    output_names = predictor.get_output_names()
    print(f"[INFO] Inputs: {input_names}, Outputs: {output_names}")

    # Get sample images
    images = glob.glob(os.path.join(ccpd_dir, "*.jpg"))
    if not images:
        print(f"[ERROR] No images found in {ccpd_dir}")
        sys.exit(1)

    random.seed(42)
    samples = random.sample(images, min(3, len(images)))

    all_pass = True
    for img_path in samples:
        char_indices, bbox = parse_ccpd_filename(img_path)
        if char_indices is None:
            print(f"[SKIP] Cannot parse: {os.path.basename(img_path)}")
            continue

        # Ground truth: CCPD index maps directly to plate dict index
        gt = "".join(
            char_set[i] for i in char_indices
            if 0 <= i < len(char_set) and char_set[i]
        )

        # Load and crop plate region
        img = cv2.imread(img_path)  # BGR
        if img is None:
            print(f"[SKIP] Cannot read: {os.path.basename(img_path)}")
            continue
        x1, y1, x2, y2 = bbox
        x1, x2 = sorted([max(0, x1), max(0, x2)])
        y1, y2 = sorted([max(0, y1), max(0, y2)])
        plate = img[y1:y2, x1:x2]
        if plate.size == 0:
            print(f"[SKIP] Empty crop: {os.path.basename(img_path)}")
            continue

        # Preprocess
        input_data = resize_norm_img(plate, 48, 160)

        # Infer
        input_handle = predictor.get_input_handle(input_names[0])
        input_handle.reshape([1, 3, 48, 160])
        input_handle.copy_from_cpu(input_data)
        predictor.run()
        output_handle = predictor.get_output_handle(output_names[0])
        output = output_handle.copy_to_cpu()

        # Decode
        pred = decode_ctc(output, char_set)

        print(f"[IMG] {os.path.basename(img_path)}")
        print(f"  GT:   {gt}")
        print(f"  Pred: {pred}")

        if not pred.strip():
            all_pass = False

    if all_pass:
        print("\n[PASS] All samples produced non-empty results.")
        sys.exit(0)
    else:
        print("\n[FAIL] Some samples produced empty results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
