# PP-OCRv6 CCPD 车牌微调训练方案 v3

> 检测 small + 识别 small | 防过拟合调参 | 单卡/多卡 GPU 服务器
> 适用场景: 实际场所车牌识别（倾斜、模糊、反光、透视变形等真实场景）

完整设计文档: `../docs/superpowers/specs/2026-07-27-ppocrv6-ccpd-finetune-design.md`

---

## v3 改进点（对比 v2）

| 项 | v2 | v3 | 原因 |
|---|----|----|------|
| 检测模型 | tiny (2.1M) | **small (11M)** | tiny 准确率不够，假阳性多、定位不准 |
| 检测学习率 | 0.001 | **0.0005** | 微调，降低防过拟合 |
| 检测 epoch | 100 | **50** | 不用跑那么久，30-50 epoch 够了 |
| 识别学习率 | 0.0005 | **0.0002** | 上次过拟合严重（train acc=1.0, val acc=0.175），降 60% |
| 识别 epoch | 100 | **30** | best 在第 4 epoch，跑多了全是过拟合 |
| 识别 eval 频率 | 每 500 step | **每 200 step** | 更及时捕捉最佳模型 |
| 识别 warmup | 5 epoch | **3 epoch** | epoch 少了，warmup 也缩短 |
| 清理工具 | 手动删 | **`--clean-only` 一键清理** | 彻底清理训练残留，方便重训 |
| 预训练模型 | 服务器下载 | **本机预下载一并上传** | 避免网络问题，开箱即用 |

---

## 文件清单

```
gpu_server_plate_training_v2/
├── prepare_ccpd_data.py          # CCPD 数据处理
├── fix_pdmodel.py                # PaddleLite 兼容性补丁
├── run_full_training.sh          # 一键训练脚本(v3)
├── configs/
│   ├── plate_det_small_gpu.yml   # 检测配置(small版)
│   └── plate_rec_small_gpu.yml   # 识别配置(small版)
├── pretrained/                   # 预训练权重(本机已下载)
│   ├── PP-OCRv6_small_det_pretrained.pdparams   (11M)
│   └── PP-OCRv6_small_rec_pretrained.pdparams   (120M)
└── README.md                     # 本文件
```

---

## 快速开始

### 首次使用（从零开始）

```bash
# 1. 将本目录整体上传到 GPU 服务器
#    例如: scp -r gpu_server_plate_training_v2/ user@server:/data/plate_train_data_source/

# 2. 修改 run_full_training.sh 顶部变量
#    PADDLE_OCR_DIR: PaddleOCR 仓库路径
#    DATA_DIR: 训练数据输出目录
#    CCPD_DIR: CCPD2019 数据集路径
#    GPU_IDS: 使用的 GPU 编号 (如 "0" 或 "0,1,2")

# 3. 赋予执行权限
chmod +x run_full_training.sh

# 4. 执行全流程
./run_full_training.sh

# 5. 训练完成后，下载以下文件到 Android 项目
#    output/plate_rec_small.nb     -> app/src/main/assets/models/recognize/
#    output/plate_det_small.nb     -> app/src/main/assets/models/detect/
#    plate_dict.txt                 -> app/src/main/assets/models/recognize/
```

### 重新训练（清理后从头来）

```bash
# 1. 一键清理（只保留预训练权重）
./run_full_training.sh --clean-only

# 2. 重新开始
./run_full_training.sh
```

### 跳过已完成的阶段

```bash
./run_full_training.sh --skip-det          # 跳过检测训练
./run_full_training.sh --skip-rec          # 跳过识别训练
./run_full_training.sh --skip-det --skip-rec  # 只做数据+导出（需已有模型）
```

---

## 训练时间估算（2 卡 10GB）

| 阶段 | 时间 | 说明 |
|------|------|------|
| 数据准备 | 30-60 分钟 | 处理 30 万张识别数据 |
| 检测训练 | **3-6 小时** | 50 epoch, 约 3.7k step/epoch |
| 识别训练 | **4-8 小时** | 30 epoch, 约 8.4k step/epoch |
| 评估 + 导出 + 转 .nb | ~30 分钟 | |
| **合计** | **8-15 小时** | |

> 实际时间取决于 GPU 型号。以上基于 RTX 2080Ti/3060 级别估算。
>
> 训练日志会实时输出 `ips`（每秒样本数），可以自己估算:
> ```
> 剩余时间(秒) = 剩余 step 数 * batch_size / ips
> ```

---

## 关键参数说明

### 为什么把学习率降到 0.0002？

v2 用 0.0005 训练时，train acc 很快冲到 1.0，但验证集 best acc 只有 0.175（第 4 个 epoch），之后验证集持续下降。这是典型的**过拟合**：学习率太高，模型迅速记住了训练数据，但泛化到新数据的能力很差。

降到 0.0002 后，收敛会慢一些，但验证集准确率会更高、更稳定。

### 为什么识别只跑 30 epoch？

v2 跑了 100 epoch，但 best_accuracy 在第 4 个 epoch 就出现了，后面 96 个 epoch 全在过拟合。降到 30 epoch 可以省 2/3 的时间，而且效果不会更差（说不定更好，因为少了过拟合）。

### 为什么检测用 small 不用 tiny？

v2 用 tiny 版检测，Android 端出现了多检测框、定位不准的问题。small 版虽然模型大 5 倍（11M vs 2M），但准确率明显更高，对于车牌检测这种实际场景更可靠。

---

## 验证训练是否有效

训练过程中关注以下指标：

### 检测模型
```
ppocr INFO: epoch: [20/50], hmean: 0.9xx
```
- hmean > 0.90 合格，> 0.93 优秀

### 识别模型
```
ppocr INFO: cur metric, acc: 0.8xx, norm_edit_dis: 0.9xx
```
- acc > 0.80 合格，> 0.90 优秀
- **重点关注 cur metric（验证集），不是 train acc**
- 如果 train acc 到 0.99+ 但 val acc 还很低 → 过拟合，降学习率或加数据

### 导出后 Python 端验证（必须做）

```bash
cd /data/PaddleOCR

# 用验证集评估
CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.checkpoints=/data/plate_train_data/output/plate_rec_small/best_accuracy

# 单张图推理测试
CUDA_VISIBLE_DEVICES=0 python3 tools/infer_rec.py \
  -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
  -o Global.infer_img=/data/plate_train_data/rec/crop_images/crop_000001.jpg \
  Global.pretrained_model=/data/plate_train_data/output/plate_rec_small/best_accuracy
```

如果 Python 端验证准确率高，但 Android 端低 → 是部署/预处理问题，不是模型问题。

---

## Android 部署

```
app/src/main/assets/models/
├── detect/
│   └── plate_det_small.nb
└── recognize/
    ├── plate_rec_small.nb
    └── plate_dict.txt
```

**注意：** `plate_dict.txt` 必须和训练时用的是同一个文件（73 字符，顺序一致）。字典不匹配会导致识别乱码或全错。

---

## 常见问题

| 问题 | 解决 |
|------|------|
| 训练 loss 不降 | 检查学习率、数据标注、预训练路径 |
| 验证 acc 很低 | 降低学习率（如 0.0001）、增加数据增强、检查验证集标注 |
| 过拟合（train acc >> val acc） | 降学习率、减少 epoch、加数据增强 |
| 转 .nb 失败 | 用 PaddleLite v2.12，先跑 fix_pdmodel.py |
| 内存不足 | det batch 降 4/2，rec batch 降 16/8 |
| .nb 推理乱码 | 检查字典（必须是训练用的 plate_dict.txt） |

---

## 完全清理命令速查

如果想从完全干净的状态开始：

```bash
# 方式一：用脚本（推荐，只删训练产物，保留预训练权重）
./run_full_training.sh --clean-only

# 方式二：手动全部删除（包括预训练，完全重来）
rm -rf /data/plate_train_data/*
# 然后重新上传整个目录
```
