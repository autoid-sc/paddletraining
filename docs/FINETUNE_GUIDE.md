# PaddleOCR 微调训练指南

## 概述

本指南说明如何在 PaddleOCR 官方预训练模型基础上，使用本地 50 张图片进行微调，以提升 VIN 码识别准确性。

## 为什么需要微调？

当前模型问题：
- 字符集仅包含 12 个字符：`0123456789:A`
- 识别结果受限，无法识别其他字符
- 模型输出始终为这 12 个字符的组合

微调优势：
- 使用官方预训练模型（已在大量数据上训练）
- 基于本地数据微调，适配特定场景
- 支持完整字符集

## 数据准备

### 1. 训练数据结构

```
img/tessdata_train/
├── 017c3a4025bd472d90e855d9b62ee903_1_1652407464.jpg
├── 017c3a4025bd472d90e855d9b62ee903_1_1652407464.gt.txt
├── 017c3a4025bd472d90e855d9b62ee903_2_1652407464.jpg
└── 017c3a4025bd472d90e855d9b62ee903_2_1652407464.gt.txt
```

- 每张图片对应一个 `.gt.txt` 标注文件
- 标注文件内容为纯文本（例如：`2022050508:50:54`）

### 2. 自动提取字符集

运行脚本自动分析数据并生成字符集：

```bash
python tools/extract_charset.py
```

输出：
- 字符统计
- 生成的字符集
- 字典文件：`img/paddle_rec_train/full_charset.txt`

### 3. 准备训练数据

运行脚本转换为 PaddleOCR 格式：

```bash
python tools/prepare_train_data.py
```

输出：
- 训练图片目录：`img/paddle_rec_train/train_data/train_images`
- 训练列表文件：`img/paddle_rec_train/train_data/train_list.txt`

## 微调训练

### 前置要求

1. **Python 环境**
   - Python 3.7+
   - PaddlePaddle
   - PaddleOCR

2. **克隆 PaddleOCR 工具**

```bash
cd tools
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

3. **安装依赖**

```bash
cd tools/PaddleOCR
pip install -r requirements.txt
```

### 方法一：使用自动化脚本（推荐）

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle
tools\run_finetune.bat
```

脚本会自动完成：
- ✓ 检查环境
- ✓ 提取字符集
- ✓ 准备训练数据
- ✓ 下载预训练模型
- ✓ 解压预训练模型
- ✓ 配置训练参数
- ✓ 显示训练命令

### 方法二：手动执行

1. **准备数据**

```bash
python tools/extract_charset.py
python tools/prepare_train_data.py
```

2. **运行训练**

```bash
cd tools/PaddleOCR
python tools/train.py \
  -c configs/rec/ch_ppocr_v4/rec_ch_ppocr_v4.yml \
  -o Global.pretrained_model=../output/finetune_vincode/pretrained_rec/ch_PP-OCRv4_rec_infer/inference.pdparams \
     Global.character_dict_path=ppocr/utils/dict/vincode_full.txt \
     Global.save_model_dir=../output/finetune_vincode \
     Global.eval_batch_step=[] \
     Train.loader.batch_size_per_card=8 \
     Train.loader.num_workers=2
```

## 训练参数说明

| 参数 | 说明 | 推荐值 |
|------|------|----------|
| `pretrained_model` | 预训练模型路径 | 官方 PP-OCRv4 模型 |
| `character_dict_path` | 字符字典 | 生成的 full_charset.txt |
| `save_model_dir` | 输出目录 | output/finetune_vincode |
| `batch_size_per_card` | 批大小 | 8-16（根据显存调整）|
| `num_workers` | 数据加载线程数 | 2-4 |

## 导出模型

训练完成后，导出推理模型：

```bash
cd tools/PaddleOCR
python tools/export_model.py \
  -c configs/rec/ch_ppocr_v4/rec_ch_ppocr_v4.yml \
  - -o Global.pretrained_model=../output/finetune_vincode/bbest_accuracy.pdparams \
     Global.export_inference_dir=../output/infer_model \
     Global.use_space_char=True
```

## 转换为 PaddleLite 模型

使用 PaddleLite 转换为 `.nb` 模型：

```bash
python paddle_lite_opt.py \
  --model_file=../output/infer_model/inference.pdmodel \
  --param_file=../output/infer_model/inference.pdiparams \
  --optimize_out=../output/nb_models/recognize \
  --valid_targets=arm \
  --optimize_out_type=naive的后缀
```

## 部署到 Android

1. 复制模型到 assets

```bash
cp output/nb_models/recognize.nb app/src/main/assets/models/recognize/
cp tools/extract_charset.py app/src/main/assets/models/recognize/full_charset.txt
```

2. 更新代码中的字典路径

```kotlin
// PaddleOCRService.kt
DICT_FILE = "full_charset.txt"
```

3. 重新编译安装

## 快速开始

一键执行：

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle
tools\run_finetune.bat
```

然后按照提示执行训练命令。

## 故障排查

### Q: 找不到 PaddleOCR 工具

**A:** 先克隆仓库

```bash
cd tools
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

### Q: 训练数据格式错误

**A:** 确保每张图片有对应的 `.gt.txt` 文件

### Q: 显存不足

**A:** 减小 batch_size

```yaml
Train:
  loader:
    batch_size_per_card: 4  # 改为 4
```

### Q: 字符集不匹配

**A:** 确保使用生成的 `full_charset.txt` 作为字典

## 参考资料

- [PaddleOCR 官方文档](https://github.com/PaddlePaddle/PaddleOCR)
- [微调训练教程](https://github.com/PaddlePaddle/PaddleOCR/blob/doc/doc_ch/finetune.md)
- [PaddleLite 文档](https://paddle-lite.readthedocs.io/)
