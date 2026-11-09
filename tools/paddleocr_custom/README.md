# PaddleOCR 本地定制补丁

仓库不直接包含 `tools/PaddleOCR` 上游克隆（约 544MB，含 .git 与训练数据）。
此目录保存对该克隆的全部本地改动，便于在任意机器上复现训练/导出的代码环境。

## 上游基线

- 仓库：https://github.com/PaddlePaddle/PaddleOCR
- 基线提交：`211989f046cc1878460f9e65574690c00a127a1a`
- 补丁文件：`paddleocr_local.patch`（含 12 个修改文件 + 11 个新增文件）

## 何时需要应用

只有在需要重新训练或重新导出模型到 PaddleLite（`.nb`）时才需要完整 PaddleOCR 源码。
Android 端推理只依赖 `app/src/main/assets/models` 下已导出的 `.nb` 模型。

## 应用方式

```bash
git clone https://github.com/PaddlePaddle/PaddleOCR.git tools/PaddleOCR
cd tools/PaddleOCR
git checkout 211989f046cc1878460f9e65574690c00a127a1a
git apply ../paddleocr_custom/paddleocr_local.patch
```

补丁内容概要：

- 车牌/印章检测与识别配置：`configs/det/PP-OCRv6/plate_*`、`configs/rec/PP-OCRv6/plate_*`、`configs/rec/PP-OCRv6/stamp_*`
- 字符字典：`plate/plate_dict.txt`、`ppocr/utils/dict/vincode_full.txt`
- 导出链路修正：`ppocr/utils/export_model.py`、`tools/export_model.py`
- 识别头/数据增强/EMA 相关改动：`ppocr/modeling/heads/*`、`ppocr/data/imaug/*`、`ppocr/utils/ema.py`

训练数据集（CRPD/CCPD、合成车牌）体积过大，未纳入仓库；采集/筛选脚本见 `gpu_server_crpd_training/`。
