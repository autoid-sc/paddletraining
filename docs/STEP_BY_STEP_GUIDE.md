# PaddleOCR 微调训练 - 完整教程

## 目标

使用 PaddleOCR 官方预训练模型 + 本地 50 张图片，生成新的 VIN 码识别模型。

---

## 第一步：获取官方预训练模型

### 1.1 PaddleOCR 官方模型

PaddleOCR 提供多个版本的预训练模型，我们使用 **PP-OCRv4**（最新版本）。

**模型下载地址：**
```
检测模型（Det）: https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_det_infer.tar
识别模型（Rec）: https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
```

### 1.2 创建下载目录

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle
mkdir -p models\pretrained
```

### 1.3 下载识别模型（只需要识别模型进行微调）

**方法 1：使用浏览器下载**
- 访问：https://paddleocr.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar
- 下载到：`models/pretrained/`

**方法 2：使用 Python 下载**

创建 `tools/download_pretrain.py`:

```python
import os
import urllib.request

# 下载地址
URL = "https://paddleocr.bj.bcebos.bj.bcebos.com/PP-OCRv4/chinese/ch_PP-OCRv4_rec_infer.tar"
SAVE_DIR = r"E:\xuhrWorkProject\floating_ocr_ai_paddle\models\pretrained"
SAVE_FILE = os.path.join(SAVE_DIR, "ch_PP-OCRv4_rec_infer.tar")

os.makedirs(SAVE_DIR, exist_ok=True)

print(f"正在下载预训练模型...")
print(f"URL: {URL}")
print(f"保存到: {SAVE_FILE}")

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = downloaded * 100.0 / total_size
        print(f"\r下载进度: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        import sys
        sys.stdout.flush()

urllib.request.urlretrieve(URL, SAVE_FILE, reporthook=report_progress)
print("\n✓ 下载完成！")
```

运行：
```bash
python tools/download_pretrain.py
```

### 1.4 解压模型

```bash
cd models\pretrained
tar -xf ch_PP-OCRv4_rec_infer.tar
```

解压后得到：
```
models/pretrained/ch_PP-OCRv4_rec_infer/
├── inference.pdmodel      # 模型结构文件
├── inference.pdiparams   # 模型参数文件
└── inference.pdparams     # 完整模型（用于继续训练）
```

**重要：** 用于微调的文件是 `inference.pdparams`

---

## 第二步：准备训练数据

### 2.1 检查本地数据

你的数据在：
```
img/tessdata_train/
├── xxx.jpg          # 图片
└── xxx.gt.txt       # 标注（包含文本内容）
```

### 2.2 提取字符集

运行脚本自动分析所有标注文件，提取出现的所有字符：

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle
python tools/extract_charset.py
```

输出：
```
字符统计:
  '0': 173
  '1': 65
  '2': 137
  '3': 22
  ...
  'A': 22

生成的字符集（12个字符）:
0
1
2
3
4
5
6
7
8
9
:
A

字典已保存到: img/paddle_rec_train/full_charset.txt
```

### 2.3 转换训练数据格式

PaddleOCR 需要的训练格式：
```
train_images/
├── image1.jpg
├── image2.jpg
└── ...

train_list.txt
```

`train_list.txt` 格式：
```
图片完整路径\t标签内容
图片完整路径\t标签内容
...
```

运行准备脚本：
```bash
python tools/prepare_train_data.py
```

输出：
```
训练数据准备完成:
  - 训练图片目录: img/paddle_rec_train/train_data/train_images
  - 训练列表文件: img/paddle_rec_train/train_data/train_list.txt
  - 训练样本数: 50
```

---

## 第三步：安装 PaddleOCR 工具

### 3.1 克隆 PaddleOCR 仓库

```bash
cd tools
git clone https://github.com/PaddlePaddle/PaddleOCR.git
```

如果 git 太慢，可以从浏览器下载：
https://github.com/PaddlePaddle/PaddleOCR/archive/refs/heads/develop.zip

### 3.2 安装 Python 依赖

**激活虚拟环境：**
```bash
cd E:\xuhraterWorkProject\floating_ocr_ai_paddle
.venv_paddle2\Scripts\activate
```

**安装 PaddleOCR 依赖：**
```bash
cd tools\PaddleOCR
pip install -r requirements.txt
```

**安装 PaddlePaddle（GPU 版本）：**
```bash
pip install paddlepaddle-gpu
```

**或安装 CPU 版本：**
```bash
pip install paddlepaddle
```

### 3.3 验证安装

```bash
python -c "import paddle; print(paddle.version.show())"
```

应该输出 Paddle 版本信息。

---

## 第四步：创建训练配置文件

### 4.1 复制官方配置

PaddleOCR 提供了完整的配置模板，我们基于中文识别模型配置修改。

**官方配置位置：**
```
tools/PaddleOCR/configs/rec/ch_ppocr_v4/rec_ch_ppocr_v4.yml
```

### 4.2 创建自定义配置

创建 `tools/finetune_config.yml`:

```yaml
Global:
  use_gpu: true                    # 使用 GPU 训练（无 GPU 改为 false）
  epoch_num: 10                   # 训练轮数
  log_smooth_window: 20
  print_batch_step: 5
  save_model_dir: ../../output/finetune_vincode  # 输出目录
  save_epoch_step: 5               # 每 5 轮保存一次
  eval_batch_step: [1000]         # 每 1000 步评估一次
  cal_metric_during_train: true
  pretrained_model: null           # 将通过命令行参数指定
  checkpoints: null
  export_inference: false

  # 字典设置（重要！）
  character_dict_path: ppocr/utils/dict/vincode_full.txt  # 我们生成的字典
  max_text_length: 25
  use_space_char: true
  drop_last: true

Optimizer:
  name: Adam
  beta1: 0.9
  beta2: 0.999
  clip_norm: 5.0
  lr:
    name: Cosine                 # 余弦退火学习率
    learning_rate: 0.0001        # 较小学习率用于微调
    warmup_epoch: 2

# 模型架构 - 使用 PP-OCRv4 的 SVTR_LCNet
Architecture:
  model_type: rec
  name: SVTR_LCNet

  Stem:
    name: ConvBNLayer
    kernel_size: 3
    stride: 1
    padding: 1
    num_channels: 3
    num_filters: 32
    use_bn: true
    act: "hard_swish"

  PatchEmbed:
    name: PatchEmbed
    img_size: [32, 100]
    patch_size: [4, 4]
    in_channels: 32
    out_channels: 128

  Block:
    name: Block
    depth: 2
    num_heads: 2
    mlp_ratio: 2
    qkv_bias: true
    qk_scale: null
    drop_rate: 0.1
    attn_drop_rate: 0.1
    drop_path_rate: 0.1

  Neck:
    name: SequencePooling
    embedding_dim: 768

  Head:
    name: CTCHead
    fc_type: 'default'
    encoder_type: 'svtr'
    mid_channels: 96
    nr_class: 6625                # 官方模型类别数（会自动根据字典调整）
    use_ctc_decode: true

Loss:
  name: CTCLoss
  use_focal_loss: false

PostProcess:
  name: CTCPostProcess

Metric:
  name: RecMetric
  main_indicator: acc

# 训练数据配置
Train:
  dataset:
    name: SimpleDataSet
    data_dir: ../../img/paddle_rec_train/train_data  # 训练数据目录
    label_file_list: ["../../img/paddle_rec_train/train_data/train_list.txt"]  # 训练列表
    ratio: 10
    transforms:
      - DecodeImage:
          img_mode: RGB
          channel_first: false
      - RecResizeImg:
          image_shape: [3, 48, 320]  # 输入图像尺寸
      - KeepKeys:
          keep_keys: ['image', 'label']

  loader:
    shuffle: true
    batch_size_per_card: 8         # 批大小（显存不足改小）
    batch_size: 8
    drop_last: true
    num_workers: 2                 # 数据加载线程数
    use_shared_memory: false

# 评估数据配置（使用训练数据）
Eval:
  dataset:
    name: SimpleDataSet
    data_dir: ../../img/paddle_rec_train/train_data
    label_file_list: ["../../img/paddle_rec_train/train_data/train_list.txt"]
    transforms:
      - DecodeImage:
          img_mode: RGB
          channel_first: false
      - RecResizeImg:
          image_shape: [3, 48, 320]
      - KeepKeys:
          keep_keys: ['image', 'label']

  loader:
    shuffle: false
    batch_size_per_card: 8
    batch_size: 8
    drop_last: false
    num_workers: 2
    use_shared_memory: false
```

### 4.3 准备配置文件

1. 复制字典到 PaddleOCR 工具目录：

```bash
copy img\paddle_rec_train\full_charset.txt tools\PaddleOCR\ppocr\utils\dict\vincode_full.txt
```

2. 复制配置文件：

```bash
copy tools\finetune_config.yml tools\PaddleOCR\configs\rec\finetune_vincode.yml
```

---

## 第五步：运行训练

### 5.1 准备训练命令

**关键参数说明：**
- `pretrained_model`: 预训练模型路径（.pdparams 文件）
- `character_dict_path`: 字典文件路径
- `save_model_dir`: 输出目录
- `batch_size_per_card`: 批大小（显存不足改小）

### 5.2 执行训练

```bash
cd tools\PaddleOCR

python tools/train.py \
  -c configs/rec/finetune_vincode.yml \
  -o Global.pretrained_model=../../models/pretrained/ch_PP-OCRv4_rec_infer/inference.pdparams \
     Global.character_dict_path=ppocr/utils/dict/vincode_full.txt \
     Global.save_model_dir=../../output/finetune_vincode \
     Train.loader.batch_size_per_card=8 \
     Train.loader.num_workers=2
```

### 5.3 训练过程输出

```
2024-06-09 10:00:00,342 - INFO - epoch 1/10, iter 1/7, loss: 3.2456, lr: 0.00004
2024-06-09 10:00:02,156 - INFO - epoch 1/10, iter 2/7, loss: 2.8765, lr: 0.00007
2024-06-09 10:00:04,234 - INFO - epoch 1/10, iter 3/7, loss: 2.5432, lr: 0.00010
...
2024-06-09 10:00:30,567 - INFO - Eval result: acc: 0.9500
2024-06-09 10:00:30,568 - INFO - save model in ../../output/finetune_vincode/iter_100.pdparams
...
```

### 5.4 训练完成

训练完成后，输出目录结构：
```
output/finetune_vincode/
├── best_accuracy.pdmodel      # 最佳模型结构
├── best_accuracy.pdiparams   # 最佳模型参数
├── best_accuracy.pdparams     # 最佳模型
├── latest.pdmodel           # 最新模型
└── ...
```

**最佳模型：** `best_accuracy.pdparams`

---

## 第六步：导出推理模型

### 6.1 导出命令

```bash
cd tools\PaddleOCR

python tools/export_model.py \
  -c configs/rec/finetune_vincode.yml \
  -o Global.pretrained_model=../../output/finetune_vincode/best_accuracy.pdparams \
     Global.export_inference_dir=../../output/infer_model \
     Global.save_inference_dir=../../output/infer_model
```

### 6.2 导出结果

```
output/infer_model/
├── inference.pdmodel
└── inference.pdiparams
```

---

## 第七步：转换为 PaddleLite 格式

### 7.1 安装 PaddleLite

```bash
pip install paddlelite
```

### 7.2 转换命令

创建 `tools/convert_to_lite.py`:

```python
from paddlelite.lite import *

# 创建转换配置
opt = LiteOpt()

# 输入模型
opt.set_model_file(r"E:\xuhrWorkProject\floating_ocr_ai_paddle\output\infer_model\inference.pdmodel")
opt.set_param_file(r"E:\xuhrWorkProject\floating_ocr_aiari_paddle\output\infer_model\inference.pdiparams")

# 输出配置
opt.set_model_dir(r"E:\xuhrWorkProject\floating_ocr_ai_paddle\output\nb_models")
opt.set_model_name("recognize")

# 目标平台
opt.set_valid_targets("arm")  # Android ARM 平台

# 精度设置
opt.set_precision("fp32")  # 使用 float32 精度

# 运行转换
opt.run()

print("✓ 模型转换完成！")
print("输出文件: output/nb_models/recognize.nb")
```

运行：
```bash
python tools/convert_to_lite.py
```

---

## 第八步：部署到 Android

### 8.1 复制模型文件

```bash
copy output\nb_models\recognize.nb app\src\main\assets\models\recognize\
copy img\paddle_rec_train\full_charset.txt app\src\main\assets\models\recognize\
```

### 8.2 更新 Android 代码

修改 `PaddleOCRService.kt`:

```kotlin
private const val RECOGNIZE_MODEL_FILE = "recognize.nb"
private const val DICT_FILE = "full_charset.txt"  // 改为新的字典
```

### 8.3 重新编译安装

```bash
cd app
./gradlew assembleDebug
./gradlew installDebug
```

---

## 常见问题

### Q1: 训练时显存不足？

**解决方法：** 减小 batch_size

```bash
-o Train.loader.batch_size_per_card=4
```

### Q2: 找不到预训练模型？

**解决方法：** 检查文件路径是否正确

```bash
dir models\pretrained\ch_PP-OCRv4_rec_infer\
```

应该看到：
```
inference.pdmodel
inference.pdiparams
inference.pdparams
```

### Q3: 字符集不匹配？

**解决方法：** 确保使用生成的字典文件

```bash
python tools/extract_charset.py  # 重新生成
```

### Q4: 训练速度太慢？

**解决方法：**
- 使用 GPU 训练
- 减小 num_workers
- 使用更少的 epoch

---

## 快速参考

### 一键运行准备

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle

# 1. 提取字符集
python tools/extract_charset.py

# 2. 准备训练数据
python tools/prepare_train_data.py

# 3. 复制文件
copy img\paddle_rec_train\full_charset.txt tools\PaddleOCR\ppocr\utils\dict\vincode_full.txt
copy tools\finetune_config.yml tools\PaddleOCR\configs\rec\finetune_vincode.yml
```

### 开始训练

```bash
cd tools\PaddleOCR
python tools/train.py \
  -c configs/rec/finetune_vincode.yml \
  -o Global.pretrained_model=../../models/pretrained/ch_PP-OCRv4_rec_infer/inference.pdparams \
     Global.character_dict_path=ppocr/utils/dict/vincode_full.txt \
     Global.save_model_dir=../../output/finetune_vincode
```

---

## 总结

流程图：

```
官方预训练模型 (PP-OCRv4)
         ↓
    解压 .tar
         ↓
获取 inference.pdparams
         ↓
    加载预训练权重
         ↓
本地 50 张训练数据
         ↓
提取字符集 full_charset.txt
         ↓
    微调训练 (10 epochs)
         ↓
生成 best_accuracy.pdparams
         ↓
    导出推理模型
         ↓
转换为 recognize.nb
         ↓
   部署到 Android
```

关键点：
1. 使用官方预训练模型作为起点
2. 字典必须与训练数据完全匹配
3. 微调使用较小学习率
4. 训练后转换为 PaddleLite 格式
