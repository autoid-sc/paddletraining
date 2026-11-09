# CRPD 车牌 PP-OCRv6 微调训练包

基于 PP-OCRv6 small 模型，使用 CRPD 数据集（single + double + multi）进行车牌检测和识别的微调训练。
一键脚本，开箱即用。

## 目录结构

```
gpu_server_crpd_training/
├── run_training.sh                # 一键训练脚本 (主入口)
├── plate_dict.txt                 # 车牌字符字典 (73字符)
├── prepare_crpd_det_data.py       # 检测数据准备脚本
├── prepare_crpd_rec_data.py       # 识别数据准备脚本
├── fix_pdmodel.py                 # pdmodel 修复 (PaddleLite 兼容)
├── opt_linux_x86                  # PaddleLite v2.14 opt 工具
├── pretrained/                    # 预训练权重 (已内置)
│   ├── PP-OCRv6_small_det_pretrained.pdparams
│   └── PP-OCRv6_small_rec_pretrained.pdparams
└── configs/
    ├── crpd_det_small_gpu.yml     # 检测训练配置
    └── crpd_rec_small_gpu.yml     # 识别训练配置
```

## 服务器部署步骤

### 1. 前置准备（服务器端）

确保服务器已安装：
- **PaddlePaddle GPU 版** (推荐 2.6+)
  ```bash
  python3 -m paddle.utils.run_check  # 验证 GPU 可用
  ```
- **PaddleOCR 仓库**
  ```bash
  git clone https://github.com/PaddlePaddle/PaddleOCR.git /data/PaddleOCR
  cd /data/PaddleOCR && pip install -r requirements.txt
  ```

### 2. 上传文件

将整个 `gpu_server_crpd_training/` 目录上传到服务器，例如：
```bash
scp -r gpu_server_crpd_training/ user@server:/data/
```

同时将 CRPD 数据集上传到服务器：
```
/data/CRPD_all/
├── CRPD_single/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   └── test/
├── CRPD_double/
│   └── ...
└── CRPD_multi/
    └── ...
```

### 3. 修改配置

编辑 `run_training.sh` 顶部的参数：

```bash
PADDLE_OCR_DIR="/data/PaddleOCR"     # PaddleOCR 仓库路径
DATA_DIR="/data/crpd_train"          # 训练输出目录
CRPD_DIR="/data/CRPD_all"            # CRPD 数据集路径
GPU_IDS="0"                          # GPU 编号，多卡用逗号分隔如 "0,1,2,3"
```

### 4. 开始训练

```bash
cd /data/gpu_server_crpd_training
chmod +x run_training.sh
./run_training.sh
```

其他用法：
```bash
./run_training.sh --skip-det          # 跳过检测，只训识别
./run_training.sh --skip-rec          # 跳过识别，只训检测
./run_training.sh --clean-only        # 清理训练输出 (保留预训练权重)
```

## 训练流程

脚本共 8 个阶段，自动依次执行：

| 阶段 | 内容 | 说明 |
|------|------|------|
| 1 | 环境校验 | 检查 PaddleOCR/PaddlePaddle/CRPD 数据 |
| 2 | 目录 & 文件准备 | 部署配置、复制预训练权重和字典 |
| 3 | 检测数据准备 | CRPD → det_train/val_list.txt |
| 4 | 识别数据准备 | 透视裁剪 + 双行展开 → rec_train/val_list.txt |
| 5 | 检测模型训练 | PP-OCRv6 small det 微调 (80 epoch) |
| 6 | 识别模型训练 | PP-OCRv6 small rec 微调 (10 epoch) |
| 7 | 导出推理模型 | export_model.py → inference 模型 |
| 8 | 转 PaddleLite .nb | 供 Android 端部署 |

## 训练配置

### 检测模型
- 基础: PP-OCRv6 small (PPLCNetV4 + RepLKFPN + DBHead)
- Epoch: 80
- 学习率: 0.0005 (Cosine + warmup 2 epoch)
- Batch: 8 / GPU
- 数据增强: ColorJitter + IaaAugment + RandomCrop + CopyPaste

### 识别模型
- 基础: PP-OCRv6 small rec (PPLCNetV4 + LightSVTR + MultiHead)
- Epoch: 10
- 学习率: 0.0001 (Cosine + warmup 1 epoch)
- Batch: 16 / GPU
- 输入尺寸: 3×48×320
- 字典: 73 字符 (31省 + 24字母 + 10数字 + 8特殊)
- 双行黄牌: 自动展开为单行训练

## 输出文件

训练完成后在 `$DATA_DIR/output/` 下：

```
crpd_det_small/               # 检测训练权重
  ├── best_accuracy.pdparams  # 最佳权重
  └── latest.pdparams         # 最新权重
crpd_rec_small/               # 识别训练权重
crpd_det_small_inference/     # 检测推理模型 (pdmodel + pdiparams)
crpd_rec_small_inference/     # 识别推理模型
crpd_det_small.nb             # 检测 PaddleLite 模型 (Android用)
crpd_rec_small.nb             # 识别 PaddleLite 模型 (Android用)
```

## 部署到 Android

```
crpd_det_small.nb  -> app/src/main/assets/models/detect/
crpd_rec_small.nb  -> app/src/main/assets/models/recognize/
plate_dict.txt     -> app/src/main/assets/models/recognize/
```

**重要**：字典文件必须三处一致（训练配置、识别 assets、代码），否则识别结果会乱码。

## 常见问题

### Q: 显存不够怎么办？
A: 减小 `configs/crpd_det_small_gpu.yml` 和 `crpd_rec_small_gpu.yml` 中的 `batch_size_per_card`。

### Q: 训练太慢怎么办？
A:
- 调大 `MAX_DET_TRAIN` / `MAX_REC_TRAIN` 限制数据量
- 减少检测 epoch（80→50）
- 使用更多 GPU

### Q: 检测数据格式不对？
A: PaddleOCR 检测格式为 `图像路径\t[{"transcription":"...","points":[[x1,y1],...],"difficult":false},...]`，`prepare_crpd_det_data.py` 已自动转换。

### Q: 识别精度不够？
A:
- 增加 epoch 数 (10→20)
- 检查训练样本是否充足且均衡
- 确认字典与数据集完全匹配
- 尝试调高学习率或增加数据增强强度

### Q: PaddleLite 转换失败？
A: 脚本已内置 `fix_pdmodel.py` 修复 swish/hardsigmoid 属性。若仍失败，会自动回退到 pip 版 paddlelite。
