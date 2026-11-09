# PP-OCRv6 针对CCPD车牌完整微调训练方案

> 日期: 2026-07-27
> 目标: 在单卡 10GB GPU 服务器上，基于 PP-OCRv6 对 CCPD2019 车牌数据集进行检测+识别完整微调，并导出部署到 Android。
> 范围: 环境清理 -> 数据准备 -> 检测训练 -> 识别训练 -> 评估 -> 导出 -> 转.nb -> Android 部署

---

## 0. 设计决策与问题修复

### 0.1 本方案修复的 5 个关键问题

| # | 问题 | 旧值(错误) | 新值(修正) |
|---|------|-----------|-----------|
| 1 | 字典字符数 | README 说 75，脚本生成 73，部署用 74，三处不一致 | **统一 73 字符**，训练/导出/Android 用同一文件 |
| 2 | RecAug 参数 | 自定义 7 个 prob 参数，与官方 PP-OCRv6 实现不匹配 | 改回官方写法 `RecAug:`（无参数，用默认增强） |
| 3 | det 预训练模型名 | `detect_tiny_pretrained`（不存在的文件名） | `PP-OCRv6_tiny_det_pretrained`（官方已验证 URL） |
| 4 | eval_batch_step | `[0, 200]`（数据量小时评估过于频繁，拖慢训练） | `[0, 500]` |
| 5 | batch_size | rec 64（10GB 显存会 OOM） | rec **32**, det **4** |

### 0.2 字典字符数论证（73 字符）

中国车牌字符集构成：
- 省份简称: 31 个（京津沪渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新）
- 字母: 24 个（A-Z，去除 I 和 O，因为易与 1、0 混淆，车牌中不使用）
- 数字: 10 个（0-9）
- 特殊字符: 8 个（挂学警领港澳使超）

合计: 31 + 24 + 10 + 8 = **73 字符**

CCPD2019 数据集的标注中，`alphabets` 数组明确不含 I 和 O，`ads` 数组中的 O 仅作为占位符（decode 时跳过）。因此 73 字符字典与数据完全匹配。

> **关键原则**: 字典文件必须在训练、导出、Android 部署三处使用同一份文件，字符顺序、数量、末尾空行必须完全一致。任何一处不一致都会导致推理乱码。

### 0.3 预训练模型 URL（均已验证 HTTP 200）

| 模型 | URL |
|------|-----|
| PP-OCRv6 small rec | `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams` |
| PP-OCRv6 tiny det | `https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams` |

---

## 1. 环境要求

| 项目 | 要求 |
|------|------|
| OS | Linux（Ubuntu 18.04+ 推荐） |
| GPU | 单卡，显存 ≥ 10GB |
| Python | 3.8+ |
| PaddlePaddle | GPU 版 ≥ 2.6 |
| PaddleOCR | 最新 develop 分支（含 PP-OCRv6 配置） |
| 磁盘 | ≥ 50GB（CCPD 数据 12GB + 处理后数据 + 输出） |
| PaddleLite | v2.12（**不要用 v2.13-rc，有 rec 模型转换崩溃 bug**） |

---

## 2. 目录结构（GPU 服务器）

```
/data/
├── CCPD2019/                      # 原始数据集（已有）
│   ├── ccpd_base/                 # ~33 万张
│   ├── ccpd_db/
│   ├── ccpd_tilt/
│   └── ...
├── PaddleOCR/                     # PaddleOCR 仓库
└── plate_train_data/              # 全新创建（清理旧的）
    ├── plate_dict.txt             # 73 字符统一字典
    ├── prepare_ccpd_data.py       # 数据处理脚本
    ├── pretrained/
    │   ├── PP-OCRv6_small_rec_pretrained.pdparams
    │   └── PP-OCRv6_tiny_det_pretrained.pdparams
    ├── det_train_list.txt         # 检测训练标注
    ├── det_val_list.txt
    ├── rec_train_list.txt         # 识别训练标注
    ├── rec_val_list.txt
    ├── rec/                       # 识别图片
    │   └── crop_images/
    │       ├── crop_000000.jpg
    │       └── ...
    └── output/
        ├── plate_det_tiny/        # 检测训练输出
        │   ├── best_accuracy.pdparams
        │   └── latest.pdparams
        ├── plate_rec_small/       # 识别训练输出
        │   ├── best_accuracy.pdparams
        │   └── latest.pdparams
        ├── plate_det_tiny_inference/   # 检测推理模型
        └── plate_rec_small_inference/  # 识别推理模型
```

---

## 3. 阶段 0: 环境清理（消除旧训练残留）

> **目的**: 之前训练产生过乱码模型，必须清理旧 output、旧数据、校验 PaddleOCR 仓库代码未被改动。

### 3.1 清理旧训练输出和旧数据

```bash
# 删除旧的处理后数据（如果存在）
rm -rf /data/plate_train_data

# 删除旧的训练输出（如果存在，可能在 PaddleOCR/output 下）
rm -rf /data/PaddleOCR/output

# 确认清理干净
ls /data/plate_train_data 2>/dev/null && echo "清理失败，目录仍存在" || echo "已清理干净"
```

### 3.2 校验 PaddleOCR 仓库代码完整性

```bash
cd /data/PaddleOCR

# 检查 git 状态，确认没有未提交的本地修改
git status

# 如果有本地修改，应该丢弃（确保用官方代码）
# ⚠️ 警告: 这会丢弃所有本地修改，请确认没有需要保留的改动
git checkout .

# 拉取最新代码（确保 PP-OCRv6 配置齐全）
git pull origin develop

# 确认 PP-OCRv6 配置目录存在
ls configs/rec/PP-OCRv6/
ls configs/det/PP-OCRv6/
```

### 3.3 校验 PaddlePaddle 环境

```bash
# 确认 PaddlePaddle GPU 版本
python3 -c "import paddle; print('PaddlePaddle:', paddle.__version__); print('GPU available:', paddle.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count())"
```

预期输出类似:
```
PaddlePaddle: 2.6.1
GPU available: True
GPU count: 1
```

如果 `GPU available: False`，说明安装的是 CPU 版，需要重装 GPU 版:
```bash
# 卸载 CPU 版
pip uninstall paddlepaddle -y

# 安装 GPU 版（CUDA 11.8 示例，根据服务器 CUDA 版本调整）
python3 -m pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

### 3.4 安装 PaddleOCR 依赖

```bash
cd /data/PaddleOCR
pip install -r requirements.txt
```

### 3.5 安装 PaddleLite v2.12（用于后续转 .nb）

```bash
# 下载 PaddleLite v2.12 的 paddle_lite_opt 工具
wget https://github.com/PaddleLite/Paddle-Lite/releases/download/v2.12/paddle_lite-linux-x64-2.12.tar.gz
tar -xzf paddle_lite-linux-x64-2.12.tar.gz
# 将 paddle_lite_opt 加入 PATH（或后续用绝对路径调用）
export PATH=$PATH:$(pwd)/paddle_lite-linux-x64-2.12/bin
paddle_lite_opt --version
```

---

## 4. 阶段 1: 创建目录结构 & 下载预训练权重

```bash
# 创建目录
mkdir -p /data/plate_train_data/pretrained
mkdir -p /data/plate_train_data/rec/crop_images
mkdir -p /data/plate_train_data/output

# 下载预训练权重（已验证 URL 有效）
cd /data/plate_train_data/pretrained

wget -O PP-OCRv6_small_rec_pretrained.pdparams \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams

wget -O PP-OCRv6_tiny_det_pretrained.pdparams \
  https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams

# 验证下载完整性
ls -lh /data/plate_train_data/pretrained/
# 预期:
# PP-OCRv6_small_rec_pretrained.pdparams  ~119MB
# PP-OCRv6_tiny_det_pretrained.pdparams   ~2MB
```

---

## 5. 阶段 2: 数据准备（CCPD 处理）

### 5.1 数据处理脚本 `prepare_ccpd_data.py`

将以下脚本保存到 `/data/plate_train_data/prepare_ccpd_data.py`:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CCPD Dataset Preprocessing for PaddleOCR PP-OCRv6 fine-tuning.
Generates: det_train/val_list.txt, rec_train/val_list.txt, plate_dict.txt (73 chars).
"""
import os
import cv2
import json
import random
import glob
import argparse
from tqdm import tqdm

# 31 provinces
provinces = ["皖","沪","津","渝","冀","晋","蒙","辽","吉","黑","苏","浙","京","闽","赣","鲁","豫","鄂","湘","粤","桂","琼","川","贵","云","藏","陕","甘","青","宁","新"]
# 24 alphabets (no I, no O)
alphabets = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"]
# ads: alphabets + digits + O(placeholder)
ads = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9","O"]

def decode_plate_number(plate_indices_str):
    try:
        indices = [int(x) for x in plate_indices_str.split('_')]
        if len(indices) < 7:
            return None
        prov = provinces[indices[0]]
        alpha = alphabets[indices[1]]
        rest = ""
        for idx in indices[2:]:
            char = ads[idx]
            if char != "O":
                rest += char
        return f"{prov}{alpha}{rest}"
    except Exception:
        return None

def parse_ccpd_filename(filepath):
    basename = os.path.basename(filepath)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.split('-')
    if len(parts) < 5:
        return None
    try:
        bbox_str = parts[2]
        bbox = [list(map(int, pt.split('&'))) for pt in bbox_str.split('_')]
        xmin, ymin = bbox[0]
        xmax, ymax = bbox[1]

        vertices_str = parts[3]
        pts = [list(map(int, pt.split('&'))) for pt in vertices_str.split('_')]
        # CCPD order: [bottom-right, bottom-left, top-left, top-right]
        # Reorder to clockwise from top-left: [top-left, top-right, bottom-right, bottom-left]
        points = [pts[2], pts[3], pts[0], pts[1]]

        plate_str = decode_plate_number(parts[4])
        if not plate_str:
            return None
        return {
            "filepath": filepath,
            "xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
            "points": points, "plate": plate_str
        }
    except Exception:
        return None

def process_ccpd(ccpd_dir, output_dir, max_det=30000, max_rec=300000):
    print("=" * 60)
    print(f"CCPD root: {ccpd_dir}")
    print(f"Output:    {output_dir}")
    print("=" * 60)

    image_paths = []
    subdirs = ["ccpd_base","ccpd_tilt","ccpd_db","ccpd_weather","ccpd_challenge","ccpd_blur","ccpd_fn"]
    for s in subdirs:
        found = glob.glob(os.path.join(ccpd_dir, s, "*.jpg"))
        print(f"  {s}: {len(found)} images")
        image_paths.extend(found)

    if not image_paths:
        print("[WARN] Recursive search...")
        image_paths = glob.glob(os.path.join(ccpd_dir, "**", "*.jpg"), recursive=True)

    print(f"Total: {len(image_paths)} images")
    if not image_paths:
        print("[ERROR] No images found.")
        return

    random.seed(42)
    random.shuffle(image_paths)

    det_out_dir = os.path.join(output_dir, "det")
    rec_out_dir = os.path.join(output_dir, "rec")
    rec_img_dir = os.path.join(rec_out_dir, "crop_images")
    os.makedirs(det_out_dir, exist_ok=True)
    os.makedirs(rec_img_dir, exist_ok=True)

    det_list = []
    rec_list = []

    print("\nProcessing...")
    for filepath in tqdm(image_paths):
        info = parse_ccpd_filename(filepath)
        if not info:
            continue

        # Detection: reference original image by absolute path
        if len(det_list) < max_det:
            line = f"{info['filepath']}\t" + json.dumps(
                [{"points": info["points"], "transcription": info["plate"]}],
                ensure_ascii=False)
            det_list.append(line)

        # Recognition: crop plate region with 2px margin
        if len(rec_list) < max_rec:
            img = cv2.imread(info["filepath"])
            if img is not None:
                h, w, _ = img.shape
                ymin = max(0, info["ymin"] - 2)
                ymax = min(h, info["ymax"] + 2)
                xmin = max(0, info["xmin"] - 2)
                xmax = min(w, info["xmax"] + 2)
                crop = img[ymin:ymax, xmin:xmax]
                if crop.size > 0:
                    crop_filename = f"crop_{len(rec_list):06d}.jpg"
                    crop_filepath = os.path.join(rec_img_dir, crop_filename)
                    cv2.imwrite(crop_filepath, crop)
                    rel_path = f"crop_images/{crop_filename}"
                    rec_list.append(f"{rel_path}\t{info['plate']}")

        if len(det_list) >= max_det and len(rec_list) >= max_rec:
            break

    print(f"\nDet: {len(det_list)}, Rec: {len(rec_list)}")

    # Split 9:1
    random.shuffle(det_list)
    det_split = int(len(det_list) * 0.9)
    with open(os.path.join(output_dir, "det_train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(det_list[:det_split]) + "\n")
    with open(os.path.join(output_dir, "det_val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(det_list[det_split:]) + "\n")
    print(f"det train: {det_split}, val: {len(det_list)-det_split}")

    random.shuffle(rec_list)
    rec_split = int(len(rec_list) * 0.9)
    with open(os.path.join(output_dir, "rec_train_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rec_list[:rec_split]) + "\n")
    with open(os.path.join(output_dir, "rec_val_list.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(rec_list[rec_split:]) + "\n")
    print(f"rec train: {rec_split}, val: {len(rec_list)-rec_split}")

    # Generate 73-char dictionary
    plate_chars = provinces + alphabets + ["0","1","2","3","4","5","6","7","8","9","挂","学","警","领","港","澳","使","超"]
    dict_file = os.path.join(output_dir, "plate_dict.txt")
    with open(dict_file, "w", encoding="utf-8") as f:
        for c in plate_chars:
            f.write(c + "\n")
        f.write("\n")  # trailing empty line (REQUIRED by PaddleOCR)
    print(f"Dict: {len(plate_chars)} chars -> {dict_file}")
    print("DONE.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ccpd_dir", default="/data/CCPD2019")
    parser.add_argument("--output_dir", default="/data/plate_train_data")
    parser.add_argument("--max_det", type=int, default=30000)
    parser.add_argument("--max_rec", type=int, default=300000)
    args = parser.parse_args()
    process_ccpd(args.ccpd_dir, args.output_dir, args.max_det, args.max_rec)
```

### 5.2 执行数据处理

```bash
cd /data/plate_train_data

python3 prepare_ccpd_data.py \
  --ccpd_dir /data/CCPD2019 \
  --output_dir /data/plate_train_data \
  --max_det 30000 \
  --max_rec 300000
```

> 说明: `--max_rec 300000` 处理 30 万张识别数据（CCPD2019 约 33 万张，取绝大部分）。`--max_det 30000` 检测用 3 万张（检测数据需求量小于识别）。处理时间约 30-60 分钟。

### 5.3 验证数据格式（必须做）

```bash
# 1. 验证字典: 73 字符 + 末尾空行 = 74 行
wc -l /data/plate_train_data/plate_dict.txt
# 预期: 74 plate_dict.txt（73 字符各占 1 行 + 1 个空行）

# 2. 验证标注格式
head -3 /data/plate_train_data/rec_train_list.txt
# 预期: crop_images/crop_000123.jpg<TAB>皖A12345

head -3 /data/plate_train_data/det_train_list.txt
# 预期: /data/CCPD2019/ccpd_base/025-90_93-182&499_463&563-463&563_182&563_182&499_463&499-0_0_22_27_27_33_16-125-79.jpg<TAB>[{"points": [[x,y],...], "transcription": "皖A12345"}]

# 3. 验证数据量
wc -l /data/plate_train_data/rec_train_list.txt   # ~27 万
wc -l /data/plate_train_data/rec_val_list.txt     # ~3 万
wc -l /data/plate_train_data/det_train_list.txt   # ~2.7 万
wc -l /data/plate_train_data/det_val_list.txt     # ~3 千

# 4. 抽查裁剪图片是否正确（人眼确认车牌内容与标注一致）
ls /data/plate_train_data/rec/crop_images/ | head -5
```

---

## 6. 阶段 3: 部署配置文件

将两个 yml 配置文件放到 PaddleOCR 的对应目录:

```bash
cp /data/plate_train_data/plate_det_tiny_gpu.yml /data/PaddleOCR/configs/det/PP-OCRv6/
cp /data/plate_train_data/plate_rec_small_gpu.yml /data/PaddleOCR/configs/rec/PP-OCRv6/
```

### 6.1 检测配置 `plate_det_tiny_gpu.yml`

将以下内容保存到 `/data/plate_train_data/plate_det_tiny_gpu.yml`:

```yaml
Global:
  model_name: plate_det_tiny_gpu
  debug: false
  use_gpu: true
  use_ema: true
  ema_decay: 0.9998
  ema_decay_type: threshold
  epoch_num: &epoch_num 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: /data/plate_train_data/output/plate_det_tiny
  save_epoch_step: 10
  eval_batch_step: [0, 200]
  cal_metric_during_train: false
  checkpoints: null
  pretrained_model: /data/plate_train_data/pretrained/PP-OCRv6_tiny_det_pretrained
  save_inference_dir: null
  use_visualdl: false
  infer_img: null
  save_res_path: /data/plate_train_data/output/plate_det_tiny/predicts_db.txt
  d2s_train_image_shape: [3, 640, 640]
  distributed: true

Architecture:
  model_type: det
  algorithm: DB
  Transform: null
  Backbone:
    name: PPLCNetV4
    det: true
    model_size: tiny
  Neck:
    name: RepLKFPN
    out_channels: 64
    dilated_kernel_size: 5
    shortcut: true
  Head:
    name: DBHead
    k: 50
    fix_nan: true
    aux_in_channels: 64

Loss:
  name: DBLoss
  main_loss_type: DiceFocalLoss
  alpha: 5
  beta: 10
  focal_alpha: 0.25
  focal_gamma: 2.5
  aux_weight_p4: 0.2
  aux_weight_p3: 0.3
  aux_weight_p2: 0.4

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.001
    warmup_epoch: 2
  regularizer:
    name: L2
    factor: 1.0e-06

PostProcess:
  name: DBPostProcess
  thresh: 0.2
  box_thresh: 0.45
  max_candidates: 1000
  unclip_ratio: 1.5

Metric:
  name: DetMetric
  main_indicator: hmean

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /
    label_file_list:
      - /data/plate_train_data/det_train_list.txt
    ratio_list: [1.0]
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - CopyPaste: null
    - ColorJitter:
        brightness: 0.4
        contrast: 0.4
        saturation: 0.7
        hue: 0.1
    - IaaAugment:
        augmenter_args:
        - type: Fliplr
          args:
            p: 0.5
        - type: Affine
          args:
            p: 0.5
            rotate:
            - -15
            - 15
            fit_output: true
        - type: Resize
          args:
            size:
            - 0.5
            - 1.5
    - RandomCrop:
        size:
        - 640
        - 640
        max_tries: 50
        keep_ratio: true
    - MakeBorderMap:
        shrink_ratio: 0.4
        thresh_min: 0.3
        thresh_max: 0.7
        total_epoch: *epoch_num
    - MakeShrinkMap:
        shrink_ratio: 0.4
        min_text_size: 8
        total_epoch: *epoch_num
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - threshold_map
        - threshold_mask
        - shrink_map
        - shrink_mask
  loader:
    shuffle: true
    drop_last: false
    batch_size_per_card: 4
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: /
    label_file_list:
      - /data/plate_train_data/det_val_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - DetLabelEncode: null
    - DetResizeForTest: null
    - NormalizeImage:
        scale: 1./255.
        mean:
        - 0.485
        - 0.456
        - 0.406
        std:
        - 0.229
        - 0.224
        - 0.225
        order: hwc
    - ToCHWImage: null
    - KeepKeys:
        keep_keys:
        - image
        - shape
        - polys
        - ignore_tags
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 1
    num_workers: 2
profiler_options: null
```

### 6.2 识别配置 `plate_rec_small_gpu.yml`

将以下内容保存到 `/data/plate_train_data/plate_rec_small_gpu.yml`:

> **关键修正**: RecAug 改回无参数官方写法；batch_size 32 适配 10GB；eval_batch_step [0,500]。

```yaml
Global:
  model_name: plate_rec_small_gpu
  debug: false
  use_gpu: true
  epoch_num: 100
  log_smooth_window: 20
  print_batch_step: 10
  save_model_dir: /data/plate_train_data/output/plate_rec_small
  save_epoch_step: 5
  eval_batch_step: [0, 500]
  cal_metric_during_train: true
  pretrained_model: /data/plate_train_data/pretrained/PP-OCRv6_small_rec_pretrained
  checkpoints: null
  save_inference_dir: null
  use_visualdl: false
  infer_img: null
  character_dict_path: /data/plate_train_data/plate_dict.txt
  max_text_length: &max_text_length 10
  infer_mode: false
  use_space_char: false
  distributed: true
  save_res_path: /data/plate_train_data/output/plate_rec_small/predicts.txt
  d2s_train_image_shape: [3, 48, 160]

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  lr:
    name: Cosine
    learning_rate: 0.0005
    warmup_epoch: 5
  regularizer:
    name: L2
    factor: 3.0e-05

Architecture:
  model_type: rec
  algorithm: SVTR_LCNet
  Transform:
  Backbone:
    name: PPLCNetV4
    model_size: small
  Head:
    name: MultiHead
    head_list:
      - CTCHead:
          Neck:
            name: lightsvtr
            dims: 120
            depth: 2
            mlp_ratio: 2.0
            local_kernel: 7
          Head:
            fc_decay: 0.00001
      - NRTRHead:
          nrtr_dim: 384
          max_text_length: *max_text_length

Loss:
  name: MultiLoss
  loss_config_list:
    - CTCLoss:
    - NRTRLoss:

PostProcess:
  name: CTCLabelDecode

Metric:
  name: RecMetric
  main_indicator: acc

Train:
  dataset:
    name: SimpleDataSet
    data_dir: /data/plate_train_data/rec/
    label_file_list:
    - /data/plate_train_data/rec_train_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - RecAug:
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - RecResizeImg:
        image_shape: [3, 48, 160]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  loader:
    shuffle: true
    batch_size_per_card: 32
    drop_last: true
    num_workers: 4

Eval:
  dataset:
    name: SimpleDataSet
    data_dir: /data/plate_train_data/rec/
    label_file_list:
    - /data/plate_train_data/rec_val_list.txt
    transforms:
    - DecodeImage:
        img_mode: BGR
        channel_first: false
    - MultiLabelEncode:
        gtc_encode: NRTRLabelEncode
    - RecResizeImg:
        image_shape: [3, 48, 160]
    - KeepKeys:
        keep_keys:
        - image
        - label_ctc
        - label_gtc
        - length
        - valid_ratio
  loader:
    shuffle: false
    drop_last: false
    batch_size_per_card: 32
    num_workers: 2
```

---

## 7. 阶段 4: 检测模型训练

### 7.1 启动训练（单卡）

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
  -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml \
  -o Global.pretrained_model=/data/plate_train_data/pretrained/PP-OCRv6_tiny_det_pretrained
```

### 7.2 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| epoch_num | 100 | 检测通常 50-100 epoch 收敛 |
| batch_size_per_card | 4 | 10GB 显存安全值（640x640 输入） |
| learning_rate | 0.001 | Cosine 衰减，warmup 2 epoch |
| save_epoch_step | 10 | 每 10 epoch 保存一次 |
| eval_batch_step | [0, 200] | 前 200 step 不评估，之后每 200 step 评估 |
| pretrained | PP-OCRv6_tiny_det_pretrained | 官方预训练权重 |

### 7.3 训练时间估算

- 3 万张 det 数据，batch 4，约 7500 step/epoch
- 100 epoch 约 75 万 step
- 单卡（如 RTX 3060/3090）约 **6-12 小时**
- 车牌检测简单，通常 30-50 epoch hmean 就能达到 0.9+，可提前停止

### 7.4 训练指标观察

日志中关注:
```
ppocr INFO: epoch: [30/100], hmean: 0.923, precision: 0.951, recall: 0.898
```
- **hmean > 0.90** 表示检测达标
- **hmean > 0.95** 优秀，可停止训练

### 7.5 从断点恢复训练（如中断）

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
  -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml \
  -o Global.checkpoints=/data/plate_train_data/output/plate_det_tiny/latest
```

---

## 8. 阶段 5: 识别模型训练

### 8.1 启动训练（单卡）

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.pretrained_model=/data/plate_train_data/pretrained/PP-OCRv6_small_rec_pretrained
```

### 8.2 训练参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| epoch_num | 100 | 识别通常 20-30 epoch 收敛 |
| batch_size_per_card | 32 | 10GB 显存安全值（48x160 输入） |
| learning_rate | 0.0005 | Cosine 衰减，warmup 5 epoch |
| save_epoch_step | 5 | 每 5 epoch 保存 |
| eval_batch_step | [0, 500] | 前 500 step 不评估 |
| max_text_length | 10 | 车牌最长 8 位，10 够用 |
| use_space_char | false | 车牌无空格 |
| image_shape | [3, 48, 160] | 车牌宽高比约 3:1，160/48≈3.3 |
| character_dict_path | plate_dict.txt | 73 字符统一字典 |

### 8.3 训练时间估算

- 27 万张 rec 数据，batch 32，约 8438 step/epoch
- 100 epoch 约 84 万 step
- 单卡约 **8-15 小时**
- 车牌数据简单，通常 20-30 epoch acc 就能达到 0.95+，可提前停止

### 8.4 训练指标观察

```
ppocr INFO: epoch: [20/100], acc: 0.956, norm_edit_dis: 0.982
```
- **acc > 0.95** 字符准确率 95%+
- **norm_edit_dis > 0.98** 编辑距离接近完美
- **acc > 0.98** 可停止训练

### 8.5 从断点恢复训练

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.checkpoints=/data/plate_train_data/output/plate_rec_small/latest
```

### 8.6 显存不足时的应急调整

如果训练时 OOM，将 `batch_size_per_card` 降到 16，同时 `num_workers` 降到 2:

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.pretrained_model=/data/plate_train_data/pretrained/PP-OCRv6_small_rec_pretrained \
  Train.loader.batch_size_per_card=16 \
  Train.loader.num_workers=2 \
  Eval.loader.batch_size_per_card=16
```

---

## 9. 阶段 6: 评估验证

### 9.1 检测模型评估

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py \
  -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml \
  -o Global.checkpoints=/data/plate_train_data/output/plate_det_tiny/best_accuracy
```

预期输出:
```
ppocr INFO: hmean: 0.92x, precision: 0.9x, recall: 0.9x
```

### 9.2 识别模型评估

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.checkpoints=/data/plate_train_data/output/plate_rec_small/best_accuracy
```

预期输出:
```
ppocr INFO: acc: 0.9x, norm_edit_dis: 0.9x
```

### 9.3 单张图推理验证（必须做）

```bash
cd /data/PaddleOCR

# 识别单张裁剪车牌
CUDA_VISIBLE_DEVICES=0 python3 tools/infer_rec.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.infer_img=/data/plate_train_data/rec/crop_images/crop_000001.jpg \
  Global.pretrained_model=/data/plate_train_data/output/plate_rec_small/best_accuracy
```

预期输出:
```
result: 皖A12345  0.998
```

> 如果输出乱码或为空，说明字典不一致或训练有问题，回到阶段 2 检查。

---

## 10. 阶段 7: 导出推理模型

### 10.1 导出检测推理模型

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py \
  -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml \
  -o Global.checkpoints=/data/plate_train_data/output/plate_det_tiny/best_accuracy \
  Global.save_inference_dir=/data/plate_train_data/output/plate_det_tiny_inference \
  Global.export_with_pir=False \
  Global.infer_mode=True
```

### 10.2 导出识别推理模型

```bash
cd /data/PaddleOCR

CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.checkpoints=/data/plate_train_data/output/plate_rec_small/best_accuracy \
  Global.save_inference_dir=/data/plate_train_data/output/plate_rec_small_inference \
  Global.export_with_pir=False \
  Global.infer_mode=True
```

### 10.3 验证导出结果

```bash
ls /data/plate_train_data/output/plate_det_tiny_inference/
# 预期: inference.pdmodel  inference.pdiparams  inference.pdiparams.info  inference.yml

ls /data/plate_train_data/output/plate_rec_small_inference/
# 预期: 同上
```

---

## 11. 阶段 8: 转换为 PaddleLite .nb 格式

### 11.1 修复 pdmodel（PaddleLite 兼容性补丁）

> **必须做**: PaddleLite v2.12 对 swish/hardsigmoid 算子的属性有严格要求，缺少 beta/slope/offset 属性会崩溃。

将以下脚本保存到 `/data/plate_train_data/fix_pdmodel.py`:

```python
#!/usr/bin/env python3
"""Fix pdmodel for PaddleLite compatibility (swish/hardsigmoid attributes)."""
import sys
import paddle.base.proto.framework_pb2 as framework_pb2

def fix_pdmodel(pdmodel_path):
    program = framework_pb2.ProgramDesc()
    with open(pdmodel_path, 'rb') as f:
        program.ParseFromString(f.read())

    fixed = 0
    for block in program.blocks:
        for op in block.ops:
            if op.type == 'swish':
                if not any(a.name == 'beta' for a in op.attrs):
                    a = op.attrs.add()
                    a.name = 'beta'; a.type = 1; a.f = 1.0
                    fixed += 1
            elif op.type in ['hard_sigmoid', 'hardsigmoid']:
                if not any(a.name == 'slope' for a in op.attrs):
                    a = op.attrs.add()
                    a.name = 'slope'; a.type = 1; a.f = 0.2
                    fixed += 1
                if not any(a.name == 'offset' for a in op.attrs):
                    a = op.attrs.add()
                    a.name = 'offset'; a.type = 1; a.f = 0.5
                    fixed += 1

    with open(pdmodel_path, 'wb') as f:
        f.write(program.SerializeToString())
    print(f"[OK] Fixed {fixed} ops in {pdmodel_path}")

if __name__ == "__main__":
    fix_pdmodel(sys.argv[1])
```

执行修复:

```bash
cd /data/plate_train_data

python3 fix_pdmodel.py /data/plate_train_data/output/plate_det_tiny_inference/inference.pdmodel
python3 fix_pdmodel.py /data/plate_train_data/output/plate_rec_small_inference/inference.pdmodel
```

### 11.2 转换为 .nb

```bash
# 检测模型 -> .nb
paddle_lite_opt \
  --model_file /data/plate_train_data/output/plate_det_tiny_inference/inference.pdmodel \
  --param_file /data/plate_train_data/output/plate_det_tiny_inference/inference.pdiparams \
  --optimize_out /data/plate_train_data/output/plate_det_small \
  --valid_targets arm \
  --optimize_out_type naive_buffer

# 识别模型 -> .nb
paddle_lite_opt \
  --model_file /data/plate_train_data/output/plate_rec_small_inference/inference.pdmodel \
  --param_file /data/plate_train_data/output/plate_rec_small_inference/inference.pdiparams \
  --optimize_out /data/plate_train_data/output/plate_rec_small \
  --valid_targets arm \
  --optimize_out_type naive_buffer

# 验证生成
ls -lh /data/plate_train_data/output/*.nb
# 预期: plate_det_small.nb  plate_rec_small.nb
```

> **注意**: 如果 `paddle_lite_opt` 通过 Python subprocess 调用崩溃，直接在 bash 命令行运行（如上）。

---

## 12. 阶段 9: Android 部署

### 12.1 拷贝模型和字典到 Android 工程

```bash
# 假设 Android 工程在本机（非服务器），从服务器下载 .nb 和字典
# 在本机执行:
scp user@gpu_server:/data/plate_train_data/output/plate_det_small.nb \
  /path/to/android_project/app/src/main/assets/models/detect/

scp user@gpu_server:/data/plate_train_data/output/plate_rec_small.nb \
  /path/to/android_project/app/src/main/assets/models/recognize/

scp user@gpu_server:/data/plate_train_data/plate_dict.txt \
  /path/to/android_project/app/src/main/assets/models/recognize/
```

### 12.2 Android assets 目录结构

```
app/src/main/assets/models/
├── detect/
│   └── plate_det_small.nb
└── recognize/
    ├── plate_rec_small.nb
    └── plate_dict.txt        ← 必须是训练用的 73 字符字典（同一份文件）
```

### 12.3 Android 端预处理要求（与训练一致）

| 项目 | 检测 | 识别 |
|------|------|------|
| 输入尺寸 | 动态 min 边 640，对齐 32 倍数 | 48 x 160（保持宽高比，右侧 padding） |
| 归一化 | ImageNet: mean=[0.485,0.456,0.406] std=[0.229,0.224,0.225] | (x - 0.5) / 0.5 |
| 通道顺序 | BGR | BGR |
| 图片类型 | uint8 -> float32 | uint8 -> float32 |

### 12.4 字典一致性最终检查

```bash
# 在服务器上
md5sum /data/plate_train_data/plate_dict.txt

# 在 Android 工程中
md5sum app/src/main/assets/models/recognize/plate_dict.txt

# 两个 md5 必须完全一致
```

---

## 13. 常见问题排查

### Q1: 训练 loss 不下降
1. 检查学习率（rec 推荐 0.0005，det 推荐 0.001）
2. 检查数据标注: `head -10 rec_train_list.txt` 确认车牌字符串正确
3. 确认预训练权重路径正确: `ls -lh /data/plate_train_data/pretrained/`
4. 确认字典内容: `cat plate_dict.txt` 应为 73 字符

### Q2: 导出后推理乱码
1. **检查字典**: 导出用的字典必须和训练完全一致（`character_dict_path` 指向同一文件）
2. **检查预处理**: 推理时图像预处理必须与训练一致（见 12.3）
3. 先用 `tools/infer_rec.py` 验证（Python 端），再用 .nb 验证（Android 端）

### Q3: 转 .nb 失败/崩溃
1. 用 PaddleLite **v2.12**（v2.13-rc 有 rec 模型转换崩溃 bug）
2. 先运行 `fix_pdmodel.py` 补丁脚本
3. 直接在 bash 命令行运行 `paddle_lite_opt`，不要通过 Python subprocess 调用

### Q4: Android 端识别不准
1. 先确认 Python 端推理准确（排除模型问题）
2. 检查 Android 端预处理（见 12.3）:
   - 检测输入尺寸（动态 min 边 640，对齐 32 倍数）
   - 识别输入（48x160，保持宽高比 + 右侧 padding）
   - 归一化参数
   - 通道顺序（BGR）
3. 检查字典 md5 是否一致

### Q5: 显存不足 OOM
- rec: 降 `batch_size_per_card` 到 16 或 8
- det: 降 `batch_size_per_card` 到 2
- 减少 `num_workers`（不影响精度，只影响数据加载速度）

### Q6: 之前训练的残留如何彻底清除
```bash
# 清理旧输出
rm -rf /data/plate_train_data/output
rm -rf /data/PaddleOCR/output
# 清理旧处理数据（含旧字典）
rm -rf /data/plate_train_data
# 重建目录从头开始（见阶段 1）
```

---

## 14. 完整执行顺序速查

```bash
# === 阶段 0: 环境清理 ===
rm -rf /data/plate_train_data
rm -rf /data/PaddleOCR/output
cd /data/PaddleOCR && git checkout . && git pull origin develop

# === 阶段 1: 目录 & 预训练权重 ===
mkdir -p /data/plate_train_data/pretrained
cd /data/plate_train_data/pretrained
wget -O PP-OCRv6_small_rec_pretrained.pdparams https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams
wget -O PP-OCRv6_tiny_det_pretrained.pdparams https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_tiny_det_pretrained.pdparams

# === 阶段 2: 数据准备 ===
cd /data/plate_train_data
python3 prepare_ccpd_data.py --ccpd_dir /data/CCPD2019 --output_dir /data/plate_train_data --max_det 30000 --max_rec 300000
wc -l plate_dict.txt  # 预期 74

# === 阶段 3: 部署配置 ===
cp plate_det_tiny_gpu.yml /data/PaddleOCR/configs/det/PP-OCRv6/
cp plate_rec_small_gpu.yml /data/PaddleOCR/configs/rec/PP-OCRv6/

# === 阶段 4: 检测训练 ===
cd /data/PaddleOCR
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml -o Global.pretrained_model=/data/plate_train_data/pretrained/PP-OCRv6_tiny_det_pretrained

# === 阶段 5: 识别训练 ===
CUDA_VISIBLE_DEVICES=0 python3 tools/train.py -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml -o Global.pretrained_model=/data/plate_train_data/pretrained/PP-OCRv6_small_rec_pretrained

# === 阶段 6: 评估 ===
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml -o Global.checkpoints=/data/plate_train_data/output/plate_det_tiny/best_accuracy
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml -o Global.checkpoints=/data/plate_train_data/output/plate_rec_small/best_accuracy

# === 阶段 7: 导出 ===
CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py -c configs/det/PP-OCRv6/plate_det_tiny_gpu.yml -o Global.checkpoints=/data/plate_train_data/output/plate_det_tiny/best_accuracy Global.save_inference_dir=/data/plate_train_data/output/plate_det_tiny_inference Global.export_with_pir=False Global.infer_mode=True
CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml -o Global.checkpoints=/data/plate_train_data/output/plate_rec_small/best_accuracy Global.save_inference_dir=/data/plate_train_data/output/plate_rec_small_inference Global.export_with_pir=False Global.infer_mode=True

# === 阶段 8: 转 .nb ===
cd /data/plate_train_data
python3 fix_pdmodel.py output/plate_det_tiny_inference/inference.pdmodel
python3 fix_pdmodel.py output/plate_rec_small_inference/inference.pdmodel
paddle_lite_opt --model_file output/plate_det_tiny_inference/inference.pdmodel --param_file output/plate_det_tiny_inference/inference.pdiparams --optimize_out output/plate_det_small --valid_targets arm --optimize_out_type naive_buffer
paddle_lite_opt --model_file output/plate_rec_small_inference/inference.pdmodel --param_file output/plate_rec_small_inference/inference.pdiparams --optimize_out output/plate_rec_small --valid_targets arm --optimize_out_type naive_buffer

# === 阶段 9: Android 部署 ===
# 拷贝 plate_det_small.nb, plate_rec_small.nb, plate_dict.txt 到 Android assets
```

---

## 15. 验收标准

| 指标 | 目标 | 说明 |
|------|------|------|
| det hmean | > 0.90 | 检测 F1 |
| rec acc | > 0.95 | 字符准确率 |
| rec norm_edit_dis | > 0.98 | 编辑距离 |
| Python 推理 | 输出正确车牌 | 单张图验证 |
| .nb 推理 | 与 Python 一致 | Android 端验证 |
| 字典 md5 | 三处一致 | 训练/导出/Android |
