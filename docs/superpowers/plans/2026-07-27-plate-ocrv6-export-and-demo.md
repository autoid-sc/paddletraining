# PP-OCRv6 车牌模型导出与 Demo 对齐实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 完善车牌模型导出主管线（集成 pdmodel 补丁 + Python 预检验），生成 det/rec 两个 .nb 并部署到 assets；对齐 PaddleOCRService.kt 的 PLATE 分支预处理使其与训练配置一致。

**Architecture:** 在 `export_and_deploy_plate_models.py` 中内联 pdmodel 属性补丁，在 paddle_lite_opt 前自动修复 swish/hard_sigmoid 算子；新增 `verify_plate_inference.py` 用 CCPD 样本验证导出的识别模型；修改 `PaddleOCRService.kt` 的 PLATE 分支预处理（det 动态 resize + ImageNet 归一化，rec 48×160 保持宽高比 + padding + BGR 通道顺序）。

**Tech Stack:** Python 3 / PaddlePaddle 2.6.2 / PaddleLite / Kotlin / OpenCV4Android

**Spec:** `docs/superpowers/specs/2026-07-27-plate-ocrv6-export-and-demo-design.md`

**注意：** 项目非 git 仓库，跳过 commit 步骤，用「运行验证」替代。

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `tools/verify_plate_inference.py` | 新建 | 用 CCPD 样本验证导出的 rec 推理模型 |
| `tools/export_and_deploy_plate_models.py` | 修改 | 集成 pdmodel 补丁 + 调用预检验 |
| `app/src/main/java/com/seuic/floating/network/PaddleOCRService.kt` | 修改 | 对齐 PLATE 分支 det/rec 预处理 |

---

## Task 1: 创建 verify_plate_inference.py

**Files:**
- Create: `tools/verify_plate_inference.py`

- [ ] **Step 1: 创建 verify_plate_inference.py 完整文件**

写入以下完整内容到 `tools/verify_plate_inference.py`：

```python
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
```

- [ ] **Step 2: 验证脚本能被 Python 正确加载（语法检查）**

Run:
```bash
"E:/xuhrWorkProject/floating_ocr_ai_paddle/.venv_paddle2/Scripts/python.exe" -c "import ast; ast.parse(open('E:/xuhrWorkProject/floating_ocr_ai_paddle/tools/verify_plate_inference.py', encoding='utf-8').read()); print('Syntax OK')"
```
Expected output: `Syntax OK`

- [ ] **Step 3: 用已导出的 test_rec_inf 模型做冒烟测试**

Run:
```bash
cd "E:/xuhrWorkProject/floating_ocr_ai_paddle" && "E:/xuhrWorkProject/floating_ocr_ai_paddle/.venv_paddle2/Scripts/python.exe" tools/verify_plate_inference.py output/test_rec_inf tools/PaddleOCR/plate/plate_dict.txt docs/CCPD2019/ccpd_base
```
Expected: 打印 3 张图的 `GT` vs `Pred`，退出码 0（若 test_rec_inf 模型已损坏则可能为 1，此时记下输出供后续对比）

---

## Task 2: 修改 export_and_deploy_plate_models.py — 集成补丁与预检验

**Files:**
- Modify: `tools/export_and_deploy_plate_models.py`

- [ ] **Step 1: 在文件顶部 import 区新增 paddle protobuf 导入**

在 `import subprocess` 之后、`SCRIPT_DIR = ...` 之前，插入：

```python
try:
    import paddle.base.proto.framework_pb2 as framework_pb2
except ImportError:
    try:
        import paddle.fluid.proto.framework_pb2 as framework_pb2
    except ImportError:
        import paddle.framework.proto.framework_pb2 as framework_pb2
```

- [ ] **Step 2: 新增 fix_pdmodel_attrs 函数**

在 `find_best_checkpoint` 函数之后，新增：

```python
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
```

- [ ] **Step 3: 新增 verify_rec_model 函数**

在 `fix_pdmodel_attrs` 函数之后，新增：

```python
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
```

- [ ] **Step 4: 修改 export_model 函数，导出后自动补丁 pdmodel**

在 `export_model` 函数中，找到 `print(f"[OK] Exported successfully to: {save_inference_dir}")` 这一行，在其后、`return True` 之前，插入补丁调用：

```python
        print(f"[OK] Exported successfully to: {save_inference_dir}")

        # Patch pdmodel for PaddleLite compatibility
        pdmodel_path = os.path.join(save_inference_dir, "inference.pdmodel")
        fix_pdmodel_attrs(pdmodel_path)

        return True
```

- [ ] **Step 5: 修改 main 函数中 rec 部分，导出后、optimize 前插入预检验**

在 `main()` 函数中，找到 rec 模型导出区块：

```python
        if export_model("Recognition", REC_CONFIG, rec_ckpt, REC_TEMP_INF):
            opt_rec_prefix = os.path.join(REC_TEMP_INF, "plate_rec_small")
```

在 `if export_model(...)` 和 `opt_rec_prefix = ...` 之间，插入预检验：

```python
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
```

- [ ] **Step 6: 语法检查修改后的文件**

Run:
```bash
"E:/xuhrWorkProject/floating_ocr_ai_paddle/.venv_paddle2/Scripts/python.exe" -c "import ast; ast.parse(open('E:/xuhrWorkProject/floating_ocr_ai_paddle/tools/export_and_deploy_plate_models.py', encoding='utf-8').read()); print('Syntax OK')"
```
Expected: `Syntax OK`

---

## Task 3: 运行完整导出管线，验证 .nb 生成

**Files:**
- 无文件修改，运行验证

- [ ] **Step 1: 运行一键导出脚本**

Run:
```bash
cd "E:/xuhrWorkProject/floating_ocr_ai_paddle" && "E:/xuhrWorkProject/floating_ocr_ai_paddle/.venv_paddle2/Scripts/python.exe" tools/export_and_deploy_plate_models.py
```
Expected:
- 检测模型导出 + 补丁 + 转 .nb 成功
- 识别模型导出 + 补丁 + 预检验（打印 3 张 CCPD 图的 GT vs Pred）+ 转 .nb 成功
- 最终 SUMMARY 显示三项 `[SUCCESS]`

- [ ] **Step 2: 验证 .nb 文件已部署到 assets**

Run:
```bash
ls -la "E:/xuhrWorkProject/floating_ocr_ai_paddle/app/src/main/assets/models/detect/plate_det_small.nb" "E:/xuhrWorkProject/floating_ocr_ai_paddle/app/src/main/assets/models/recognize/plate_rec_small.nb" "E:/xuhrWorkProject/floating_ocr_ai_paddle/app/src/main/assets/models/recognize/plate_dict.txt"
```
Expected: 三个文件都存在，`plate_rec_small.nb` 大小 > 1MB

---

## Task 4: 修改 PaddleOCRService.kt — 检测 PLATE 分支预处理

**Files:**
- Modify: `app/src/main/java/com/seuic/floating/network/PaddleOCRService.kt`

- [ ] **Step 1: 修改 detectText 函数，为 PLATE 分支添加动态 resize + ImageNet 归一化，保留非 PLATE 的 pad-to-480 逻辑**

找到 `private fun detectText(mat: Mat): List<TextBox> {` 函数体，将整个函数替换为：

```kotlin
    private fun detectText(mat: Mat): List<TextBox> {
        val boxes = mutableListOf<TextBox>()

        try {
            val predictor = detPredictor ?: return boxes

            val isPlate = OcrConfig.ocrModelTier.value == OcrConfig.OcrModelTier.PLATE

            // Compute resize parameters and produce normalized Mat
            val ratioH: Float
            val ratioW: Float
            val inputH: Int
            val inputW: Int
            val normalized: Mat

            if (isPlate) {
                // Dynamic resize: min side = 640, align to 32 multiples (matches DetResizeForTest)
                val limitSideLen = 640
                val srcH = mat.height()
                val srcW = mat.width()
                val ratio = if (min(srcH, srcW) < limitSideLen) {
                    limitSideLen.toFloat() / min(srcH, srcW)
                } else {
                    1.0f
                }
                val rh = maxOf(kotlin.math.round((srcH * ratio) / 32.0).toInt() * 32, 32)
                val rw = maxOf(kotlin.math.round((srcW * ratio) / 32.0).toInt() * 32, 32)
                val resizedMat = Mat()
                Imgproc.resize(mat, resizedMat, Size(rw.toDouble(), rh.toDouble()))
                normalized = Mat()
                resizedMat.convertTo(normalized, CvType.CV_32FC3, 1.0 / 255.0)
                resizedMat.release()
                inputH = rh
                inputW = rw
                ratioH = rh.toFloat() / srcH
                ratioW = rw.toFloat() / srcW
            } else {
                // Original: resize + pad to 480x480 (preserves existing CHINESE/ENGLISH behavior)
                val inputSize = 480
                val scale = minOf(inputSize.toFloat() / mat.width(), inputSize.toFloat() / mat.height())
                val newWidth = (mat.width() * scale).toInt()
                val newHeight = (mat.height() * scale).toInt()
                val resizedMat = Mat()
                Imgproc.resize(mat, resizedMat, Size(newWidth.toDouble(), newHeight.toDouble()))
                val normTemp = Mat()
                resizedMat.convertTo(normTemp, CvType.CV_32FC3, 1.0 / 255.0)
                resizedMat.release()
                val padded = Mat.zeros(inputSize, inputSize, CvType.CV_32FC3)
                val roi = padded.submat(0, newHeight, 0, newWidth)
                normTemp.copyTo(roi)
                normTemp.release()
                normalized = padded
                inputH = inputSize
                inputW = inputSize
                ratioH = scale
                ratioW = scale
            }

            // ImageNet normalization for PLATE, identity for others
            val meanVals = if (isPlate) doubleArrayOf(0.485, 0.456, 0.406) else doubleArrayOf(0.0, 0.0, 0.0)
            val stdVals = if (isPlate) doubleArrayOf(0.229, 0.224, 0.225) else doubleArrayOf(1.0, 1.0, 1.0)

            // NCHW with BGR channel order (2-c converts RGB->BGR, matching training img_mode: BGR)
            val inputData = FloatArray(3 * inputH * inputW)
            val tempBuffer = FloatArray(inputH * inputW * 3)
            normalized.get(0, 0, tempBuffer)

            for (c in 0 until 3) {
                val offset = c * inputH * inputW
                val colorChannelIdx = 2 - c
                for (h in 0 until inputH) {
                    val rowOffset = h * inputW
                    for (w in 0 until inputW) {
                        val idx = offset + rowOffset + w
                        val srcIdx = (rowOffset + w) * 3
                        val pixel = tempBuffer[srcIdx + colorChannelIdx]
                        inputData[idx] = ((pixel - meanVals[c]) / stdVals[c]).toFloat()
                    }
                }
            }

            // Prepare input tensor
            val inputTensor = predictor.getInput(0)
            inputTensor.resize(longArrayOf(1, 3, inputH.toLong(), inputW.toLong()))
            inputTensor.setData(inputData)

            // Run inference
            predictor.run()

            // Parse DBNet output
            val detectedBoxes = parseDBNetOutput(predictor, mat.width(), mat.height(), ratioW, ratioH)
            boxes.addAll(detectedBoxes)

            // Cleanup
            normalized.release()

        } catch (e: Exception) {
            Log.e(TAG, "文本检测失败", e)
        }

        return boxes
    }
```

- [ ] **Step 2: 修改 parseDBNetOutput 函数签名，用 ratioW/ratioH 替代单个 scale**

找到 `private fun parseDBNetOutput(` 函数签名行，将整个函数替换为：

```kotlin
    private fun parseDBNetOutput(
        predictor: PaddlePredictor,
        originalWidth: Int,
        originalHeight: Int,
        ratioW: Float,
        ratioH: Float
    ): List<TextBox> {
        val boxes = mutableListOf<TextBox>()

        try {
            val outputTensor = predictor.getOutput(0)
            val outputData = outputTensor.getFloatData()
            val outputShape = outputTensor.shape()

            Log.d(TAG, "检测输出shape: ${outputShape.joinToString("x")}")

            if (outputShape.size < 3) {
                Log.e(TAG, "检测输出形状不正确")
                return boxes
            }

            val batchSize = outputShape[0].toInt()
            val channels = if (outputShape.size > 1) outputShape[1].toInt() else 1
            val outputHeight = outputShape[outputShape.size - 2].toInt()
            val outputWidth = outputShape.last().toInt()

            Log.d(TAG, "检测输出尺寸: $batchSize x $channels x $outputHeight x $outputWidth")

            val probMat = Mat.zeros(outputHeight, outputWidth, CvType.CV_32F)
            val probData = FloatArray(outputHeight * outputWidth)
            for (h in 0 until outputHeight) {
                val rowOffset = h * outputWidth
                for (w in 0 until outputWidth) {
                    val idx = rowOffset + w
                    probData[idx] = outputData[idx]
                }
            }
            probMat.put(0, 0, probData)

            val threshold = 0.3
            val binaryMat = Mat()
            Imgproc.threshold(probMat, binaryMat, threshold, 255.0, Imgproc.THRESH_BINARY)

            val binaryMat8U = Mat()
            binaryMat.convertTo(binaryMat8U, CvType.CV_8U)

            val contours = java.util.ArrayList<MatOfPoint>()
            val hierarchy = Mat()
            Imgproc.findContours(
                binaryMat8U, contours, hierarchy,
                Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE
            )

            Log.d(TAG, "找到 ${contours.size} 个轮廓")

            val minBoxSize = 3.0
            val minArea = minBoxSize * minBoxSize

            for (i in 0 until contours.size) {
                val contour = contours[i]
                val rect = Imgproc.boundingRect(contour)
                val width = rect.width.toDouble()
                val height = rect.height.toDouble()

                if (width < minBoxSize || height < minBoxSize) {
                    continue
                }

                val area = width * height
                if (area < minArea) {
                    continue
                }

                // Scale back to original image using separate ratios
                val scaledLeft = rect.x.toDouble() / ratioW
                val scaledTop = rect.y.toDouble() / ratioH
                val scaledRight = (rect.x + rect.width).toDouble() / ratioW
                val scaledBottom = (rect.y + rect.height).toDouble() / ratioH

                val clippedLeft = scaledLeft.coerceIn(0.0, originalWidth.toDouble())
                val clippedTop = scaledTop.coerceIn(0.0, originalHeight.toDouble())
                val clippedRight = scaledRight.coerceIn(0.0, originalWidth.toDouble())
                val clippedBottom = scaledBottom.coerceIn(0.0, originalHeight.toDouble())

                boxes.add(TextBox(clippedLeft.toFloat(), clippedTop.toFloat(), clippedRight.toFloat(), clippedBottom.toFloat()))
            }

            probMat.release()
            binaryMat.release()
            binaryMat8U.release()
            hierarchy.release()
            for (contour in contours) {
                contour.release()
            }

            Log.d(TAG, "过滤后得到 ${boxes.size} 个文本框")

        } catch (e: Exception) {
            Log.e(TAG, "解析 DBNet 输出失败", e)
        }

        return boxes
    }
```

- [ ] **Step 3: 语法检查（Kotlin 编译在 Task 7 统一做）**

确认替换后文件中 `detectText` 和 `parseDBNetOutput` 函数签名匹配（`ratioW: Float, ratioH: Float`）。

---

## Task 5: 修改 PaddleOCRService.kt — 识别 PLATE 分支预处理

**Files:**
- Modify: `app/src/main/java/com/seuic/floating/network/PaddleOCRService.kt`

- [ ] **Step 1: 修改 recognizeText 函数，为 PLATE 分支添加 48×160 保持宽高比 + padding + BGR 通道顺序**

找到 `private fun recognizeText(mat: Mat, box: TextBox): String? {` 函数体，将整个函数替换为：

```kotlin
    private fun recognizeText(mat: Mat, box: TextBox): String? {
        try {
            val predictor = recPredictor ?: return null
            val isPlate = OcrConfig.ocrModelTier.value == OcrConfig.OcrModelTier.PLATE

            // Extract ROI
            val rectWidth = (box.right - box.left).toInt()
            val rectHeight = (box.bottom - box.top).toInt()
            if (rectWidth <= 0 || rectHeight <= 0) {
                Log.w(TAG, "跳过无效的文本框 (width=$rectWidth, height=$rectHeight)")
                return null
            }
            val roiRect = org.opencv.core.Rect(
                box.left.toInt(),
                box.top.toInt(),
                rectWidth,
                rectHeight
            )
            val roiMat = Mat(mat, roiRect)

            // Resize parameters
            val inputHeight = 48
            val inputWidth: Int
            val resizedW: Int
            val resizedMat = Mat()

            if (isPlate) {
                // Keep aspect ratio: resize to height=48, width<=160, then pad right to 160
                inputWidth = 160
                val ratio = rectWidth.toFloat() / rectHeight
                resizedW = minOf(kotlin.math.ceil(inputHeight * ratio).toInt(), inputWidth)
                val tempResized = Mat()
                Imgproc.resize(roiMat, tempResized,
                    Size(resizedW.toDouble(), inputHeight.toDouble()))
                // Pad to inputWidth (right-side zero padding, matches resize_norm_img)
                resizedMat.create(inputHeight, inputWidth, tempResized.type())
                resizedMat.setTo(org.opencv.core.Scalar(0.0, 0.0, 0.0))
                val roi = resizedMat.submat(0, inputHeight, 0, resizedW)
                tempResized.copyTo(roi)
                tempResized.release()
            } else {
                // Original: force resize to 48x320
                inputWidth = 320
                resizedW = inputWidth
                Imgproc.resize(roiMat, resizedMat,
                    Size(inputWidth.toDouble(), inputHeight.toDouble()))
            }

            // Normalize to float32, scale 1/255
            val normalized = Mat()
            resizedMat.convertTo(normalized, CvType.CV_32FC3, 1.0 / 255.0)

            // NCHW conversion
            val inputData = FloatArray(3 * inputHeight * inputWidth)
            val tempBuffer = FloatArray(inputHeight * inputWidth * 3)
            normalized.get(0, 0, tempBuffer)

            if (isPlate) {
                // PLATE: (x-0.5)/0.5 normalization + BGR channel order (2-c)
                // Matches training: resize_norm_img with img_mode BGR
                for (c in 0 until 3) {
                    val offset = c * inputHeight * inputWidth
                    val colorChannelIdx = 2 - c
                    for (h in 0 until inputHeight) {
                        val rowOffset = h * inputWidth
                        for (w in 0 until inputWidth) {
                            val srcIdx = (rowOffset + w) * 3
                            val value = tempBuffer[srcIdx + colorChannelIdx]
                            inputData[offset + rowOffset + w] = (value - 0.5f) / 0.5f
                        }
                    }
                }
            } else {
                // Original: (x-0.5)/0.5 with RGB channel order (c directly)
                for (c in 0 until 3) {
                    val offset = c * inputHeight * inputWidth
                    for (h in 0 until inputHeight) {
                        val rowOffset = h * inputWidth
                        for (w in 0 until inputWidth) {
                            val srcIdx = (rowOffset + w) * 3
                            val value = tempBuffer[srcIdx + c]
                            inputData[offset + rowOffset + w] = (value - 0.5f) / 0.5f
                        }
                    }
                }
            }

            // Set input
            val inputTensor = predictor.getInput(0)
            inputTensor.resize(longArrayOf(1, 3, inputHeight.toLong(), inputWidth.toLong()))
            inputTensor.setData(inputData)

            // Run inference
            predictor.run()

            // Get output and decode
            val outputTensor = predictor.getOutput(0)
            val outputData = outputTensor.getFloatData()
            val outputShape = outputTensor.shape()
            Log.d(TAG, "识别输出shape: ${outputShape.joinToString("x")}, dataSize=${outputData.size}, dictSize=${charSet.size}")

            val text = decodeCTC(outputData, outputShape)

            // Cleanup
            roiMat.release()
            resizedMat.release()
            normalized.release()

            return text

        } catch (e: Exception) {
            Log.e(TAG, "文本识别失败", e)
        }

        return null
    }
```

- [ ] **Step 2: 确认 recognizeText 中调用 parseDBNetOutput 的一致性**

`recognizeText` 不调用 `parseDBNetOutput`，无需额外修改。确认 `detectText` 中调用 `parseDBNetOutput(predictor, mat.width(), mat.height(), ratioW, ratioH)` 参数顺序与 Task 4 Step 2 的签名 `ratioW: Float, ratioH: Float` 一致。

---

## Task 6: 修改 PaddleOCRService.kt — PLATE 分支字符集不加空格

**Files:**
- Modify: `app/src/main/java/com/seuic/floating/network/PaddleOCRService.kt`

- [ ] **Step 1: 修改 loadCharSet 函数，PLATE 分支不追加空格字符**

找到 `private fun loadCharSet(): Boolean {` 函数体，将其中追加空格的部分：

```kotlin
            // 追加空格字符以匹配 use_space_char=True 的训练行为
            list.add(" ")
```

替换为：

```kotlin
            // 追加空格字符以匹配 use_space_char=True 的训练行为
            // PLATE 模型 use_space_char=false，不追加空格
            val isPlate = OcrConfig.ocrModelTier.value == OcrConfig.OcrModelTier.PLATE
            if (!isPlate) {
                list.add(" ")
            }
```

- [ ] **Step 2: 确认 decodeCTC 函数无需修改**

`decodeCTC` 函数使用 `charSet.size` 做边界检查，`blankIndex = 0`，`charIndex = maxIndex - 1`。PLATE 分支 charSet 有 74 个条目（73 字符 + 1 空行），模型输出 75 类（blank + 74），解码逻辑正确，无需修改。

---

## Task 7: 编译验证 Android 应用

**Files:**
- 无文件修改，编译验证

- [ ] **Step 1: 编译 Android 应用（debug 变体）**

Run:
```bash
cd "E:/xuhrWorkProject/floating_ocr_ai_paddle" && ./gradlew :app:compileDebugKotlin 2>&1 | tail -30
```
Expected: `BUILD SUCCESSFUL`。若有 Kotlin 编译错误，根据报错修正（常见：未导入 `kotlin.math` 函数、变量名拼写）。

- [ ] **Step 2: 完整 assembleDebug（可选，验证资源打包含 .nb）**

Run:
```bash
cd "E:/xuhrWorkProject/floating_ocr_ai_paddle" && ./gradlew :app:assembleDebug 2>&1 | tail -20
```
Expected: `BUILD SUCCESSFUL`，生成的 APK 中 `assets/models/detect/plate_det_small.nb` 和 `assets/models/recognize/plate_rec_small.nb` 均存在。

---

## 验证清单

- [ ] `tools/verify_plate_inference.py` 语法正确，冒烟测试能跑 CCPD 图
- [ ] `tools/export_and_deploy_plate_models.py` 语法正确
- [ ] 运行导出管线后，`plate_det_small.nb` + `plate_rec_small.nb` + `plate_dict.txt` 均在 assets 中
- [ ] PaddleOCRService.kt 编译通过
- [ ] PLATE 分支 det 预处理：动态 resize + ImageNet 归一化
- [ ] PLATE 分支 rec 预处理：48×160 保持宽高比 + padding + BGR
- [ ] PLATE 分支 charSet 不含多余空格
