# floating_ocr_ai_paddle

基于 PaddleOCR / PaddleLite 的 Android 悬浮框 OCR 工具，聚焦车牌、印章等场景的
训练、模型导出与端侧推理。

## 目录结构

| 目录 | 说明 |
| --- | --- |
| `app/` | Android 应用（Kotlin），悬浮窗相机 + PaddleLite 推理 |
| `openCVLibrary345/` | OpenCV 3.4.5 Android 模块 |
| `tools/` | 数据处理、模型转换、导出部署脚本 |
| `tools/paddleocr_custom/` | 对 PaddleOCR 上游的本地改动补丁（源码克隆不入库） |
| `gpu_server_crpd_training/` | CRPD 车牌检测/识别 GPU 训练脚本与配置 |
| `gpu_server_plate_training*/` | CCPD / 车牌训练环境与配置 |
| `docs/` | 训练与部署文档、方案设计 |

## 端侧模型

已导出的 PaddleLite 模型位于 `app/src/main/assets/models/`：

- `detect/`：通用检测 `detect_small.nb`、车牌检测 `plate_det_small.nb`
- `recognize/`：中文识别 `rec_small.nb`、英文识别 `rec_en_small.nb`、车牌识别 `plate_rec_small.nb`
- 字典：`ch_dict.txt`、`en_dict.txt`、`plate_dict.txt`、`ch_dict_full.txt`

## 构建

1. 在 `local.properties` 配置 `sdk.dir`（该文件不入库）。
2. 签名配置读取 `gradle.properties` 中的 `KEY_PATH` / `KEY_ALIAS` / `KEY_PASSWORD` /
   `STORE_PASSWORD`，签名文件 `app/platform.keystore` 由本地提供，不入库。
3. 构建：

```bash
./gradlew :app:assembleDebug
```

## 模型训练与导出

训练、微调与导出流程见：

- `docs/MODEL_TRAINING_AND_DEPLOYMENT_GUIDE.md`
- `docs/STEP_BY_STEP_GUIDE.md`
- `docs/QUICK_START.md`
- `docs/FINETUNE_GUIDE.md`

训练数据集、预训练权重、推理中间产物和压缩包均未入库，按文档中的脚本自行准备。

## 仓库约定

- 不入库：签名文件、`local.properties`、构建产物、虚拟环境、数据集、模型权重与压缩包。
- PaddleOCR 定制改动以补丁形式保存在 `tools/paddleocr_custom/paddleocr_local.patch`。
