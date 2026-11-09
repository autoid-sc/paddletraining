# PP-OCRv6 车牌模型导出与 Demo 对齐设计

## 日期
2026-07-27

## 背景与现状

项目 `floating_ocr_ai_paddle` 是一个 Android 端 OCR 应用，使用 PaddleLite 在端侧推理 PP-OCRv6 模型，支持三种模型层级：中文（CHINESE）、英文（ENGLISH）、车牌（PLATE）。

车牌检测与识别模型已基于 30 万张图片训练完成：
- 检测 checkpoint：`output/plate_det_tiny/best_accuracy.pdparams`
- 识别 checkpoint：`output/plate_rec_small/best_accuracy.pdparams`
- 字典：`tools/PaddleOCR/plate/plate_dict.txt`（75 行，74 个有效字符 + 1 个 CTC blank）

**当前问题：**
1. 识别模型 `.nb` 转换失败——`plate_rec_small.nb` 未出现在 `app/src/main/assets/models/recognize/`。根因：PaddleOCRv6 导出的 `inference.pdmodel` 中 `swish` / `hard_sigmoid` 算子缺失默认属性（`beta` / `slope` / `offset`），PaddleLite 不兼容。已有 `fix_swish_pdmodel.py` / `fix_pdmodel_attrs.py` 补丁脚本但未集成进主管线。
2. Demo 预处理与训练配置不一致：
   - 检测：Demo 用 480×480 固定 resize + 仅 `×1/255` 归一化；训练用 `DetResizeForTest`（min 边 736，32 倍数）+ ImageNet mean/std 归一化。
   - 识别：Demo 用 48×320 强制 resize；训练用 `image_shape: [3,48,160]` + 保持宽高比 + 右侧 padding。
3. 缺少转换前的 Python 预检验，问题会推迟到 Android 端才暴露。

## 目标

1. 完善 `export_and_deploy_plate_models.py`，让 det + rec 两个 `.nb` 一键生成并部署到 assets。
2. 在转 `.nb` 前用 Python 跑 CCPD 样本图验证识别模型能正确识别。
3. 对齐 `PaddleOCRService.kt` 的 PLATE 分支预处理，使其与训练配置一致。
4. 保持全套 PaddleOCRv6 架构不变，只动预处理与导出流程。
5. 向后兼容：中文/英文模型分支不受影响。

## 非目标

- 不修改 PaddleOCRv6 模型架构（Backbone / Neck / Head 不动）。
- 不新增独立 demo 模块。
- 不改动训练流程。

## 架构与文件变更

| 文件 | 操作 | 说明 |
|------|------|------|
| `tools/export_and_deploy_plate_models.py` | 修改 | 集成 pdmodel 补丁 + 调用预检验 |
| `tools/verify_plate_inference.py` | 新增 | Python 端预检验脚本 |
| `app/src/main/java/com/seuic/floating/network/PaddleOCRService.kt` | 修改 | 对齐 PLATE 分支预处理 |

### 数据流

```
best_accuracy.pdparams
        │
        ▼  export_model.py (PaddleOCR tools)
inference.pdmodel + .pdiparams
        │
        ▼  verify_plate_inference.py  (仅 rec，跑 CCPD 样本)
   (通过验证)
        │
        ▼  fix_pdmodel_attrs (内联补丁，det + rec 都执行)
patched inference.pdmodel
        │
        ▼  paddle_lite_opt --valid_targets arm
plate_det_small.nb + plate_rec_small.nb
        │
        ▼  copy to assets
app/src/main/assets/models/{detect,recognize}/
        │
        ▼  PaddleOCRService.kt (PLATE 分支)
   Android 端车牌识别
```

## 详细设计

### 1. 导出主管线变更（export_and_deploy_plate_models.py）

在现有 `export_model()` 成功后、`optimize_model()` 之前，插入两个步骤：

#### 1.1 预检验（仅 rec）

调用 `verify_plate_inference.py` 作为子进程，传入导出目录、字典路径、CCPD 样本目录。退出码 0 通过，非 0 中止部署。det 模型跳过预检验（检测框难以自动判定通过）。

#### 1.2 pdmodel 补丁（det + rec 都执行）

新增 `fix_pdmodel_attrs(pdmodel_path)` 函数，内联 `fix_pdmodel_attrs.py` 的逻辑：

- 用 `paddle.base.proto.framework_pb2.ProgramDesc` 解析 `inference.pdmodel`
- 遍历所有 op：
  - `swish` 算子缺 `beta` 属性 → 补 `beta=1.0`（type=FLOAT）
  - `hard_sigmoid` 算子缺 `slope` → 补 `slope=0.2`；缺 `offset` → 补 `offset=0.5`
- 回写 pdmodel

向后兼容：对不含这些算子的模型无副作用（空扫描）。

#### 1.3 流程顺序

```
export_model(name, config, ckpt, save_dir)
  ↓
if name == "Recognition":
    verify_plate_inference(save_dir, dict, ccpd_dir)  # 新增
  ↓
fix_pdmodel_attrs(save_dir/inference.pdmodel)          # 新增
  ↓
optimize_model(...)  # paddle_lite_opt
```

### 2. Python 预检验脚本（verify_plate_inference.py，新增）

- **依赖**：项目现有 `.venv_paddle2` 环境（含 paddlepaddle、numpy、cv2）
- **输入**：
  - 导出目录（含 `inference.pdmodel` + `inference.pdiparams`）
  - 字典路径
  - CCPD 样本目录（默认 `docs/CCPD2019/ccpd_base`）
- **流程**：
  1. `paddle.jit.load` 加载推理模型
  2. 从 CCPD `ccpd_base` 随机取 3 张图
  3. 解析 CCPD 文件名得到车牌真值与车牌 bbox（CCPD 命名规范：`...-bbox四点-plate字符索引-亮度-模糊度.jpg`）
  4. 按 bbox 裁剪车牌区域
  5. 预处理（复刻 `resize_norm_img`）：保持宽高比缩放到高 48，宽 ≤ 160，右侧 zero-padding 到 48×160，`(pixel/255 - 0.5)/0.5` 归一化，NCHW
  6. 推理 → CTC 解码（blank index = 0，char index = argmax - 1）
  7. 打印 `真值 vs 预测`
- **判定**：3 张图识别结果均非空 → 退出码 0；否则退出码 1
- **CCPD 文件名真值解码**：文件名 plate 部分（如 `0_0_22_10_26_29_24`）是字典索引，需用 `ppocr_keys_v1.txt` 或车牌字典映射回字符。预检验脚本直接用车牌字典反向映射。

### 3. Demo 预处理对齐（PaddleOCRService.kt，PLATE 分支）

#### 3.1 检测预处理（PLATE 分支）

| 项 | 现状 | 改为 |
|----|------|------|
| 输入尺寸 | 固定 480×480 | 动态：保持宽高比，min(h,w)=640，对齐 32 倍数 |
| 归一化 | `×1/255` | `×1/255` 后减 mean `[0.485,0.456,0.406]` 除 std `[0.229,0.224,0.225]` |
| 通道顺序 | RGB（从 RGBA 转来） | BGR（与训练 `img_mode: BGR` 一致） |
| Tensor shape | 固定 `[1,3,480,480]` | 动态 `[1,3,resized_h,resized_w]` |

实现要点：
- `inputTensor.resize(longArrayOf(1, 3, resizedH.toLong(), resizedW.toLong()))`
- DBNet 后处理的 `scale` 改为 `ratio_h, ratio_w`（分别对应 h/w 缩放比），用于把检测框映射回原图

#### 3.2 识别预处理（PLATE 分支）

| 项 | 现状 | 改为 |
|----|------|------|
| 输入宽度 | 固定 320 | 160（与训练 `image_shape: [3,48,160]` 一致） |
| Resize | 强制 48×320 | 保持宽高比缩放到高 48，宽 ≤ 160，右侧 zero-padding 到 48×160 |
| 归一化 | `(x/255 - 0.5)/0.5` | 不变（已与训练一致） |

实现要点（复刻 `resize_norm_img`）：
```kotlin
val imgH = 48
val imgW = 160
val ratio = roiWidth.toFloat() / roiHeight
var resizedW = ceil(imgH * ratio).toInt()
if (resizedW > imgW) resizedW = imgW
val resizedMat = Mat()
Imgproc.resize(roiMat, resizedMat, Size(resizedW.toDouble(), imgH.toDouble()))
// 归一化 (x/255 - 0.5)/0.5，填充到 [3, imgH, imgW]
```

#### 3.3 影响范围

- 仅 `OcrConfig.OcrModelTier.PLATE` 分支生效
- 中文 / 英文分支保持现有 480×480 / 48×320 逻辑不变
- 通过 `when (OcrConfig.ocrModelTier.value)` 分支隔离

## 验证方式

1. **Python 预检验**：导出后立即跑 CCPD 3 张图，打印 `真值 vs 预测`
2. **.nb 生成检查**：`plate_det_small.nb` 和 `plate_rec_small.nb` 都出现在 assets 对应目录
3. **Android 端**：选择「车牌场景模型」层级，对车牌图识别，输出应包含省份汉字 + 字母数字

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| pdmodel 补丁对新版 Paddle 导出的模型可能有其他缺失属性 | 补丁函数设计为幂等、可扩展；若 paddle_lite_opt 仍报错，按报错新增属性补全分支 |
| CCPD 文件名真值解码可能因字典差异与训练字典不一致 | 预检验脚本直接用车牌字典（`plate_dict.txt`）反向映射，与训练一致 |
| 动态输入尺寸在部分老旧 PaddleLite 版本可能不支持 | 项目现有 PaddleLite 已支持 `inputTensor.resize`，无兼容风险 |
| det 输入从 480 改为 640 会增加移动端耗时 | 640 仍小于训练默认 736，是精度/性能折中；后续可配置化 |

## 不改动项

- PaddleOCRv6 模型架构（DB + PPLCNetV4 检测，SVTR_LCNet + MultiHead 识别）
- 训练配置文件（`plate_det_tiny.yml`、`plate_rec_small.yml`）
- 中文 / 英文模型分支的预处理逻辑
- `OcrConfig.kt` 的 PLATE 枚举定义
