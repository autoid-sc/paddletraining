# Tesseract (Tessdata) & PaddleOCR (v4/v5) 训练与移动端部署全流程指南

本指南系统整理了从零开始进行 **Tesseract (Tessdata)** 和 **PaddleOCR (包含 PP-OCRv4/v5)** 模型训练、转换、以及移动端部署的完整流程，并重点针对真实业务场景中常见的 **多角度文本（正反、颠倒、侧视：0°/90°/180°/270°）** 提供了系统化解决方案，最后对 `ncnn` (集成 `nihui/ncnn-android-ppocrv5`) 与 `Paddle Lite` 两个移动端推理框架的转化与部署方案进行了对比分析。

---

## 目录
0. [深度学习 OCR 技术演进与核心算法选型](#0-深度学习-ocr-技术演进与核心算法选型)
1. [Tessdata (Tesseract LSTM) 端到端微调训练流程](#1-tessdata-tesseract-lstm-端到端微调训练流程)
2. [PaddleOCR (PP-OCRv4/v5) 全套模型训练流程](#2-paddleocr-pp-ocrv4v5-全套模型训练流程)
3. [多角度文本识别（0°/90°/180°/270°）深度技术方案](#3-多角度文本识别090180270深度技术方案)
4. [移动端高效部署：Paddle Lite 与 ncnn 双重管道](#4-移动端高效部署paddle-lite-与-ncnn-双重管道)
5. [Android 混合开发（JNI 与 NDK 架构设计）](#5-android-混合开发jni-与-ndk-架构设计)

---

## 0. 深度学习 OCR 技术演进与核心算法选型

### 0.1 传统图像处理 OCR 与 深度学习 OCR 对比
传统的 OCR 技术（如早期 Tesseract 3.x 或基于 OpenCV 的模板匹配）依赖于手工特征和严苛的规则过滤：
1. **二值化分割**：利用 Otsu 等算法将图像转为黑白。
2. **字符切分**：采用投影法或连通域寻找单个字符边界。**若字符发生粘连、断裂或重叠，切分必将失败**。
3. **分类识别**：利用 HOG、SIFT 特征配合 SVM 分类器对单字进行识别。

深度学习 OCR（如 PP-OCR 系列）将检测与识别任务转化为**端到端学习问题**：
* **免切分字符**：通过引入 **CTC（联结性时间分类）** 损失函数，使网络可以直接对整行文本图像进行特征提取与对齐解码，彻底规避了单字切分的行业难题。
* **高鲁棒性**：借助卷积神经网络（CNN）多尺度视觉特征，对自然场景中的反光、阴影、污损、低对比度有极强的自适应能力。

### 0.2 核心检测算法选型：为什么选择 DBNet / DBNet++？
在文本定位（Text Detection）中，常见的两阶段（Faster R-CNN）或单阶段（YOLO）回归网络虽然适合检测独立物体，但难以完美拟合任意角度、弯曲的长文本行。
* **DBNet（Differentiable Binarization）**：将文本检测建模为像素级二值化分割。
* **可微二值化（DB）**：传统分割网络（如 U-Net）在生成概率图后，需通过固定阈值进行硬二值化处理，该步骤不可导，无法端到端联合训练。DBNet 将二值化操作设计为可微的近似阶跃函数，使网络可以直接对二值化图进行反向传播和联合优化。
* **优势**：速度极快，且可输出多边形顶点，即使条码或 VIN 码倾斜、折射、呈侧视畸变，也能生成精准的文本行轮廓。

### 0.3 核心识别算法选型：CRNN 与 SVTR_LCNet 的演进
* **CRNN (CNN + RNN + CTC)**：
  经典结构使用 CNN 提取图像特征图，转为特征序列后送入双向 LSTM（双向循环神经网络）捕捉上下文依赖，最后通过 CTC 解码。其痛点是 **RNN 在移动端 CPU 上无法并行计算**，且对超长文本的特征提取存在瓶颈。
* **SVTR_LCNet (PP-OCRv4/v5)**：
  PP-OCRv4 和 v5 识别网络移除了沉重的 RNN 层，采用基于 **SVTR (Single Visual Model for Text Recognition)** 的视觉 Transformer 自注意力机制，同时将 Backbone 替换为极致优化的轻量级 **LCNet**（融入了 Depthwise 深度可分离卷积与轻量注意力机制）。
* **优势**：推理时呈纯 CNN / Transformer 块结构，可高度并发计算，推理速度大幅提升，整句识别率提升明显。

### 0.4 开发框架与移动端预测引擎的抉择
* **开发框架（PyTorch vs. PaddlePaddle）**：
  PyTorch 适合学术探索，但其在边缘端部署时算子转换（如 PyTorch ➔ ONNX ➔ TFLite）常遇到算子不兼容。PaddlePaddle 原生自带面向工业落地的完整全家桶，内置多阶段 PP-OCR 轻量网络及高效标注工具 `PPOCRLabel`。
* **端侧引擎（Paddle Lite vs. ncnn）**：
  在 Android 设备部署中，`ncnn` 相比 `Paddle Lite` 拥有更极致的 ARM NEON 汇编级指令集调优，并提供免依赖的 `opencv-mobile` 链接支持。其原生支持的 **Vulkan API**，能完美调用 Android 设备的 GPU 算力，大幅降低 CPU 开销并减缓持续扫描时设备的发热问题。

---


## 1. Tessdata (Tesseract LSTM) 端到端微调训练流程

Tesseract 从 4.0 版本起引入了基于 LSTM（长短期记忆网络）的 OCR 引擎，其自定义模型训练主要依靠命令行工具集。

### 1.1 环境与前置准备
1. **安装 Tesseract 软件**：确保安装带有开发工具包的 Tesseract 5.x 版本（包含 `lftraining`、`combine_tessdata`、`unicharset_extractor` 等可执行文件）。
2. **准备训练基础包**：从官方获取对应语言的 `.traineddata` 文件（如 `eng.traineddata` 或 `chi_sim.traineddata`），作为微调的基底模型。

### 1.2 数据集结构与命名规范
Tesseract LSTM 训练的数据核心由 **成对的图片文件 (`.png`/`.tiff`)** 和 **标注文本文件 (`.gt.txt`)** 组成。
* **命名规范**：文件名前缀必须完全一致，且包含语言和字体/序列标识。
  * `eng.customfont.exp0.png`
  * `eng.customfont.exp0.gt.txt` （内容为该图片中展示的纯文本，如 `2022050508:50:54`）

### 1.3 核心步骤命令

#### 步骤一：提取特征文件 `.lstmf`
利用 Tesseract 执行 LSTM 训练提取模式，为每张图片生成 `.lstmf` 格式的中间特征文件。
```bash
# 对单张图片进行特征提取
tesseract eng.customfont.exp0.png eng.customfont.exp0 -l eng --psm 6 lstm.train
```
*批量执行脚本（Windows PowerShell）*：
```powershell
Get-ChildItem -Filter *.png | ForEach-Object {
    $baseName = $_.BaseName
    tesseract $_.FullName $baseName -l eng --psm 6 lstm.train
}
```

#### 步骤二：提取字符集 `unicharset`
利用所有 `.gt.txt` 文件提取模型所需的字符字典。
```bash
unicharset_extractor --output_unicharset unicharset --norm_mode 1 *.gt.txt
```

#### 步骤三：创建 Starter 基础数据包
将生成的 `unicharset` 与基础语言包解压出的模型配置合并，创建一个初始的 `.traineddata`。
```bash
# 从官方模型提取 LSTM 基础参数
combine_tessdata -e eng.traineddata eng.lstm

# 组合 unicharset 生成 starter.traineddata
combine_tessdata -u unicharset starter.traineddata
```

#### 步骤四：运行微调训练 `lftraining`
指定输出目录、训练列表文件、字符集和基底模型，启动训练。
```bash
# 1. 创建训练列表
ls *.lstmf > train_list.txt

# 2. 启动训练
lftraining \
  --model_output ./output/custom_ocr \
  --continue_from ./eng.lstm \
  --traineddata ./starter.traineddata \
  --train_listfile ./train_list.txt \
  --max_iterations 1000
```
* 参数解释：
  * `--continue_from`：要继承权重的 LSTM 网络层。
  * `--max_iterations`：训练最大迭代次数。

#### 步骤五：停止训练并打包输出 `.traineddata`
训练达到预期的目标误差率后，将检查点打包成最终可在安卓端调用的模型文件。
```bash
lftraining \
  --stop_training \
  --continue_from ./output/custom_ocr_checkpoint \
  --traineddata ./starter.traineddata \
  --model_output ./output/eng_custom.traineddata
```

### 1.4 Android 部署与调用
在 Android 中使用 `TessTwo` 或 `Tess-Two` 库：
1. 将 `eng_custom.traineddata` 放置在 Android 的外部存储或 Assets 中。
2. 启动时将其拷贝至 `filesDir/tessdata/eng_custom.traineddata`（Tesseract 强要求父目录名为 `tessdata`）。
3. 调用示例：
```kotlin
val tessBaseAPI = TessBaseAPI()
tessBaseAPI.init(context.filesDir.absolutePath, "eng_custom")
tessBaseAPI.pageSegMode = TessBaseAPI.PageSegMode.PSM_SINGLE_LINE
tessBaseAPI.setImage(bitmap)
val recognizedText = tessBaseAPI.utf8Text
tessBaseAPI.end()
```

---

## 2. PaddleOCR (PP-OCRv4/v5) 全套模型训练流程

PaddleOCR 采用高精度、高效率的三阶段架构：**文本检测 (DBNet) ➔ 方向分类 (Cls) ➔ 文本识别 (SVTR_LCNet)**。

```
                     ┌───────────┐
                     │ 原始图像   │
                     └─────┬─────┘
                           │
                           ▼
                  ┌────────────────┐
                  │ 1. 文本检测     │  (DBNet / DBNet++)
                  │    (Detect)    │
                  └────────┬───────┘
                           │ 裁剪文本框 (RoI)
                           ▼
                  ┌────────────────┐
                  │ 2. 方向分类     │  (MobileNetV3 Cls 0°/180°)
                  │ (Direction Cls)│  *选配：4-class 90°/270°
                  └────────┬───────┘
                           │ 旋转矫正至水平
                           ▼
                  ┌────────────────┐
                  │ 3. 文本识别     │  (SVTR_LCNet / SVTR_LCNetV3)
                  │  (Recognize)   │
                  └────────┬───────┘
                           │ CTC 解码
                           ▼
                     ┌───────────┐
                     │ 文本结果   │
                     └───────────┘
```

### 2.1 文本检测模型训练 (DBNet)
DBNet（Differentiable Binarization）利用分割网络直接预测文本边界，对弯曲、倾斜文本有极强的适应性。

* **标注工具**：推荐使用 [PPOCRLabel](https://github.com/PaddlePaddle/PaddleOCR/tree/main/PPOCRLabel)。
* **标注格式**：`train_list.txt`（以 JSON Lines 格式组织每张图片的标注数据）
  ```
  img/det_001.jpg \t [{"transcription": "202205", "points": [[10, 20], [100, 20], [100, 50], [10, 50]]}]
  ```
* **训练配置**：修改 `configs/det/det_mv3_db.yml` 中的数据路径及图像缩放大小。
* **执行命令**：
  ```bash
  python tools/train.py -c configs/det/det_mv3_db.yml -o Global.use_gpu=True
  ```

### 2.2 文本识别模型训练 (SVTR_LCNet)
PP-OCRv4/v5 识别模型基于 SVTR（Single Visual Model for Text Recognition）以及轻量级神经网络 LCNet 进行了重构。

1. **生成字典文件**：
   从训练标注中提取所有非重复字符，并保存为 `vincode_full.txt`，每一行一个字符。
2. **数据格式准备**：
   包含切分好的文本行图片，以及 `train_list.txt`，格式为：`相对路径 \t 文本内容`。
   ```
   rec_images/crop_01.jpg \t 2022050508:50:54
   ```
3. **关键配置参数 (`configs/rec/finetune_vincode.yml`)**：
   ```yaml
   Global:
     pretrained_model: ./pretrain/ch_PP-OCRv4_rec_train/best_accuracy
     character_dict_path: ./ppocr/utils/dict/vincode_full.txt
     use_space_char: true
   Train:
     dataset:
       name: SimpleDataSet
       data_dir: ./train_data
       label_file_list: ["./train_data/train_list.txt"]
   ```
4. **训练与导出**：
   ```bash
   # 训练
   python tools/train.py -c configs/rec/finetune_vincode.yml -o Global.epoch_num=100
   # 导出推理模型
   python tools/export_model.py -c configs/rec/finetune_vincode.yml -o Global.pretrained_model=./output/best_accuracy -o Global.save_inference_dir=./output/infer_rec
   ```

### 2.3 方向分类模型训练 (MobileNetV3 Classifier)
默认的二进制分类模型主要用于识别文本是否颠倒（即旋转了 180°）。

* **标注格式**：
  每一行指定切分文本图路径和方向索引（`0` 代表 0 度正向，`1` 代表 180 度反向）。
  ```
  cls_images/crop_01.jpg \t 0
  cls_images/crop_02.jpg \t 1
  ```
* **配置参数**：修改 `configs/cls/cls_mv3.yml` 的类别数量（默认为 `2` 门类）。
* **执行命令**：
  ```bash
  python tools/train.py -c configs/cls/cls_mv3.yml
  ```

---

## 3. 多角度文本识别（0°/90°/180°/270°）深度技术方案

在真实物理扫描、手持 PDA 等现场场景中，条码/VIN码文本可能呈 **正向、倒挂、左侧横排、右侧竖排**。系统必须能鲁棒地处理全角度文本。以下为四种主流架构解决方案：

### 方案 A：训练 4 分类（0°/90°/180°/270°）方向分类器（强烈推荐，从源头解决）
默认 PaddleOCR 的 `ch_ppocr_mobile_v2.0_cls` 模型仅支持 2 分类（0 和 180°）。我们可以通过**拓展输出 Head** 将其改写为 4 分类器。

#### 1. 数据集构建与数据增强
对现有的文本行图片进行离线旋转生成四种方向数据：
* 正向图片：旋转 0° ➔ 标签：`0`
* 右旋图片：旋转 90° ➔ 标签：`1`
* 倒置图片：旋转 180° ➔ 标签：`2`
* 左旋图片：旋转 270° ➔ 标签：`3`

#### 2. 修改 Paddle 配置文件
复制 `configs/cls/cls_mv3.yml` 配置文件，并修改核心参数：
```yaml
Architecture:
  name: ClsModel
  algorithm: CLS
  Transform: null
  Backbone:
    name: MobileNetV3
    scale: 0.35
    model_name: small
  Head:
    name: ClsHead
    class_num: 4       # 从 2 改为 4，适配四个旋转角度
```
训练完毕后，将其导出为推理模型，并在推理阶段将对应的 90°、180°、270° 图像按照分类结果进行 OpenCV 矩阵逆向旋转旋转至正向（0°），再送入识别模型。

---

### 方案 B：数据增强（在检测和识别层面对旋转免疫）
如果不希望在流程中多加一个分类模型，可以通过**极大规模的数据增强**使检测和识别模型具有旋转自适应能力。

#### 1. 检测模型旋转增强
在 `configs/det/det_mv3_db.yml` 的 `Train.transforms` 下配置旋转数据增强，使检测模型能准确定位任意角度文本框：
```yaml
- EastRandomCropData:
    size: [640, 640]
    max_tries: 50
    keep_ratio: true
- RandomRotate:      # 随机旋转增强
    angle: [-180, 180]
```

#### 2. 识别模型旋转增强
在 `configs/rec/finetune_vincode.yml` 中添加随机小角度旋转和翻转：
```yaml
- RecAug:
    use_horizontal_flip: false
    use_vertical_flip: false
    angle: [-90, 90]  # 允许最多 90 度的随机倾斜矫正
```

---

### 方案 C：利用 DBNet 多边形检测与透视变换（适用于侧视、倾斜）
当文本从斜侧面拍摄时（产生透视形变），直接用平行矩形裁剪会产生大量背景干扰。

```
   侧视倾斜的 VIN 码:
   A (x1, y1)                      B (x2, y2)
     ┌────────────────────────────┐
      \   2 0 2 2 0 5 0 5 0 8      \
       └────────────────────────────┘
       D (x4, y4)                     C (x3, y3)
```

#### 1. 提取多边形顶点
DBNet 的后处理可以输出 4 个顶点的具体坐标 $(x_1, y_1), (x_2, y_2), (x_3, y_3), (x_4, y_4)$（而不是简单的外接矩形 `Rect`）。
#### 2. OpenCV 透视变换（GetPerspectiveTransform）
在 JNI C++ 端或 Python 端，计算目标矩形并进行仿射剪裁，直接将侧视倾斜拉伸为正视矩形：
```cpp
cv::Point2f src_pts[4];
src_pts[0] = cv::Point2f(x1, y1);
src_pts[1] = cv::Point2f(x2, y2);
src_pts[2] = cv::Point2f(x3, y3);
src_pts[3] = cv::Point2f(x4, y4);

// 计算文本行最佳宽度与高度
float width_top = cv::norm(src_pts[1] - src_pts[0]);
float width_bottom = cv::norm(src_pts[2] - src_pts[3]);
float max_width = std::max(width_top, width_bottom);

float height_left = cv::norm(src_pts[3] - src_pts[0]);
float height_right = cv::norm(src_pts[2] - src_pts[1]);
float max_height = std::max(height_left, height_right);

cv::Point2f dst_pts[4];
dst_pts[0] = cv::Point2f(0, 0);
dst_pts[1] = cv::Point2f(max_width, 0);
dst_pts[2] = cv::Point2f(max_width, max_height);
dst_pts[3] = cv::Point2f(0, max_height);

cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
cv::Mat cropped_aligned;
cv::warpPerspective(original_image, cropped_aligned, M, cv::Size(max_width, max_height));
```
该步骤可彻底消除侧视形变对识别网络造成的精度负面影响。

---

### 方案 D：应用层多角度轮询与投票机制（免改动模型，快速上线）
在一些对耗时非极其敏感的非实时识别场景中，可以直接在代码层通过 OpenCV 旋转输入源：
1. **输入图片** ➔ **运行检测与识别** ➔ **得到结果 1 与置信度 $S_1$**。
2. 将输入图片逆时针旋转 90° ➔ **运行检测与识别** ➔ **得到结果 2 与置信度 $S_2$**。
3. 将输入图片旋转 180° ➔ **得到结果 3 与置信度 $S_3$**。
4. 将输入图片旋转 270° ➔ **得到结果 4 与置信度 $S_4$**。
5. **决策过滤**：选择置信度最高且内容符合 VIN 码格式规范（如包含 17 位字符，或满足特定正则）的结果上报。

---

## 4. 移动端高效部署：Paddle Lite 与 ncnn 双重管道

移动端深度学习推理有两种高效管道：**Paddle Lite**（百度官方 ARM 推理库）和 **ncnn**（腾讯开源的移动端极致优化框架）。

### 4.1 管道对比

| 维度 | Paddle Lite | ncnn (推荐，结合 `nihui/ncnn-android-ppocrv5`) |
| :--- | :--- | :--- |
| **体积与依赖** | 较大。需要包含 `libpaddle_light_api_shared.so`，包含较多辅助算子。 | **极小**。只依赖编译出的单个轻量动态链接库，且支持静态裁剪。 |
| **GPU 加速** | OpenCL 支持有限，配置较繁琐。 | **原生支持 Vulkan API**，在大部分中高端移动 GPU 上加速显著。 |
| **模型格式** | `.nb` 专有格式（Naive Buffer）。 | `.param` (网络结构文本) 与 `.bin` (权重二进制)。 |
| **开发活跃度** | 官方逐渐合流到 Paddle-Inference，Lite 分支更新缓慢。 | 社区极其活跃，针对 ARM NEON 指令集及多核 CPU 进行了汇编级调优。 |

---

### 4.2 管道一：Paddle Lite 转换与部署
当前项目的 `FloatingOcrService.kt` 使用此方式。

```
[Paddle paramsCheckpoint (.pdparams)]
                 │
                 ▼ (PaddleOCR export_model.py)
[Paddle Inference Model (.pdmodel & .pdiparams)]
                 │
                 ▼ (paddle_lite_opt 离线转换)
[PaddleLite Model (.nb)]
                 │
                 ▼ (Deploy to assets)
[Android: PaddleLite C++/Java SDK]
```

#### 转换命令
```bash
pip install paddlelite

# optimize_out 会在指定位置自动生成 detect.nb
paddle_lite_opt \
  --model_file=./infer_det/inference.pdmodel \
  --param_file=./infer_det/inference.pdiparams \
  --optimize_out=./app/src/main/assets/models/detect/detect \
  --valid_targets=arm \
  --optimize_out_type=naive_buffer
```

---

### 4.3 管道二：ncnn 转换与部署（利用 ONNX 作为桥梁）
`nihui/ncnn-android-ppocrv5` 的核心运行逻辑。

```
[Paddle Inference Model (.pdmodel & .pdiparams)]
                 │
                 ▼ (paddle2onnx)
[ONNX Model (.onnx)]
                 │
                 ▼ (onnx-simplifier)
[Simplified ONNX (.onnx)]
                 │
                 ▼ (onnx2ncnn)
[ncnn Model (.param & .bin)]
                 │
                 ▼ (ncnnoptimize 精度压缩与融合)
[Optimized ncnn Model (.param & .bin)]
```

#### 第一步：Paddle 转 ONNX
使用 `paddle2onnx` 命令行工具转换模型：
```bash
pip install paddle2onnx

paddle2onnx \
  --model_dir ./infer_rec \
  --model_filename inference.pdmodel \
  --params_filename inference.pdiparams \
  --save_file ./onnx_models/rec.onnx \
  --opset_version 11 \
  --enable_onnx_checker True
```

#### 第二步：ONNX 模型结构简化
由于 Paddle 转换出来的 ONNX 包含部分冗余的 reshape/squeeze 算子，需通过 `onnxsim` 移除：
```bash
pip install onnxscript onnx-simplifier

onnxsim ./onnx_models/rec.onnx ./onnx_models/rec_sim.onnx
```

#### 第三步：ONNX 转 ncnn
利用 ncnn 官方编译的 `onnx2ncnn` 工具进行转换：
```bash
# 生成 rec.param 和 rec.bin
onnx2ncnn ./onnx_models/rec_sim.onnx ./ncnn_models/rec.param ./ncnn_models/rec.bin
```

#### 第四步：模型优化与量化 (ncnnoptimize)
对网络中无用算子进行剪枝，并支持将 FP32 转换到 FP16 存储，能减少一半的模型包体积，并大幅提高手机带宽利用率：
```bash
# 最后一项参数中，0 代表保持 FP32/FP16 自动混合精度，1 代表强制强制 FP32
ncnnoptimize ./ncnn_models/rec.param ./ncnn_models/rec.bin ./ncnn_models/rec_opt.param ./ncnn_models/rec_opt.bin 0
```

---

## 5. Android 混合开发（JNI 与 NDK 架构设计）

为获得极致性能，OCR 的核心前处理（灰度化、二值化、轮廓查找）、模型推理（Vulkan NDK 调用）和后处理（CTC解码、DBNet多边形拟合）全部需要放在 JNI C++ 层运行，从而避免 Java/Kotlin 与 Native 之间频繁拷贝大块图像内存带来的延迟。

### 5.1 NDK 工程目录设计 (以 `ncnn-android-ppocrv5` 为标准)
```
app/src/main/
├── cpp/
│   ├── CMakeLists.txt        # 编译配置文件
│   ├── ocr_jni.cpp           # JNI 暴露的 C++ 接口函数
│   ├── ppocr_engine.cpp      # PP-OCR 核心处理逻辑类
│   ├── ppocr_engine.h
│   └── include/              # ncnn & opencv-mobile 头文件
├── assets/
│   └── models/
│       ├── det.param         # 检测模型结构
│       ├── det.bin           # 检测模型权重
│       ├── rec.param         # 识别模型结构
│       ├── rec.bin           # 识别模型权重
│       └── dict.txt          # 识别词典
```

### 5.2 核心 C++ 代码解析

#### 1. JNI 接口编写 (`ocr_jni.cpp`)
接收 Android Bitmap 引用，在底层获取像素指针并转换为 OpenCV 的 Mat：
```cpp
#include <jni.h>
#include <android/bitmap.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include "ppocr_engine.h"

extern "C" JNIEXPORT jstring JNICALL
Java_com_seuic_floating_network_NcnnOcrService_recognizeBitmap(
        JNIEnv* env, jobject thiz, jobject bitmap) {
    
    AndroidBitmapInfo info;
    void* pixels = nullptr;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0 ||
        AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        return env->NewStringUTF("");
    }

    // 将 Raw 像素直接初始化为 cv::Mat (假设格式为 RGBA_8888)
    cv::Mat mat_rgba(info.height, info.width, CV_8UC4, pixels);
    cv::Mat mat_rgb;
    cv::cvtColor(mat_rgba, mat_rgb, cv::COLOR_RGBA2RGB);

    // 解锁像素指针
    AndroidBitmap_unlockPixels(env, bitmap);

    // 实例化底层的 PP-OCR 预测引擎
    static PPOCREngine* g_ocr_engine = new PPOCREngine();
    
    // 执行全流程识别
    std::string result = g_ocr_engine->run(mat_rgb);

    return env->NewStringUTF(result.c_str());
}
```

#### 2. C++ 端 ncnn 推理调度 (`ppocr_engine.cpp`)
加载模型并配置多线程参数与 Vulkan GPU 推理：
```cpp
#include "ppocr_engine.h"
#include <ncnn/net.h>

bool PPOCREngine::init(AAssetManager* mgr) {
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;               // 绑核多线程控制
    opt.use_vulkan_compute = true;     // 开启 Vulkan GPU 加速机制

    det_net.opt = opt;
    rec_net.opt = opt;

    // 从 Android AssetManager 加载
    if (det_net.load_param(mgr, "models/det.param") != 0 ||
        det_net.load_model(mgr, "models/det.bin") != 0) {
        return false;
    }
    
    if (rec_net.load_param(mgr, "models/rec.param") != 0 ||
        rec_net.load_model(mgr, "models/rec.bin") != 0) {
        return false;
    }
    return true;
}

std::string PPOCREngine::run(const cv::Mat& src) {
    // 1. 调用 DBNet 检测出边界框集合 std::vector<TextBox>
    std::vector<TextBox> boxes = detect(src);
    
    std::string final_text = "";
    
    for (const auto& box : boxes) {
        // 2. 切图并针对倾斜/侧视做 warpPerspective 透视仿射对齐
        cv::Mat aligned_crop = crop_and_align(src, box);
        
        // 3. 将图像送入 ncnn 运行识别
        ncnn::Mat in = ncnn::Mat::from_pixels_resize(
            aligned_crop.data, 
            ncnn::Mat::PIXEL_RGB, 
            aligned_crop.cols, 
            aligned_crop.rows, 
            320, 48
        );
        
        // 归一化均值和方差
        const float mean_vals[3] = {127.5f, 127.5f, 127.5f};
        const float norm_vals[3] = {1.0f / 127.5f, 1.0f / 127.5f, 1.0f / 127.5f};
        in.substract_mean_normalize(mean_vals, norm_vals);

        ncnn::Extractor ex = rec_net.create_extractor();
        ex.input("in_tensor", in);
        
        ncnn::Mat out;
        ex.extract("out_tensor", out);
        
        // 4. CTC 解码得到字符串
        std::string word = decode_ctc(out);
        final_text += word;
    }
    
    return final_text;
}
```

### 5.3 CMakeLists.txt 库依赖声明
为完成 Native 构建，必须正确链接 `ncnn`、`opencv-mobile` 以及 NDK 提供的系统图像操作库 `jnigraphics`：
```cmake
cmake_minimum_required(VERSION 3.10.2)
project("ncnn_ocr")

# 设置 ncnn 路径（需解压 ncnn-android-vulkan SDK）
set(ncnn_DIR ${CMAKE_SOURCE_DIR}/sdk/ncnn-android-vulkan/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# 设置 opencv-mobile 路径
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/sdk/opencv-mobile/sdk/native/jni)
find_package(OpenCV REQUIRED)

add_library(ncnn_ocr SHARED
            ocr_jni.cpp
            ppocr_engine.cpp)

target_link_libraries(ncnn_ocr
                      ${ncnn_LIBS}
                      ${OpenCV_LIBS}
                      jnigraphics
                      log)
```

---

## 6. 总结与最佳实践

在向生产线和终端应用部署自定义 OCR 服务时，应遵守以下工程准则：
1. **优先保障数据多样性**：对于扫描角度不确定的 VIN 码识别，**预先在识别和检测数据集中加入大角度旋转增强**（方案 B）是最稳妥的做法，能降低模型复杂度。
2. **多机型适配优先考虑 ncnn**：当应用需要在硬件配置极其参差的低端手持终端上流畅运行时，基于 **ncnn + FP16 自动混合精度** 导出的 PP-OCRv5 模型，能在最大程度上减少功耗与内存抖动。
3. **关键场景采用级联双检**：
   * 在主 CPU 推理通路使用极速 PaddleOCR 或 ncnn 模型进行快速捕获。
   * 对于扫描错误、校验和失败（如 17 位 VIN 码校验不通过）的脏图像帧，调用备用 Tesseract/Tessdata 模块进行二次确认。
