# PaddleOCR 微调训练 - 快速开始指南

## ✅ 准备工作已全部完成！

所有数据已准备就绪，现在可以开始训练。

---

## 📊 当前状态

| 项目 | 状态 | 路径 |
|------|------|------|
| 预训练模型 | ✅ 已准备 | `padddle/ch_PP-OCRv4_rec_infer/` |
| 训练数据 | ✅ 已准备 | 50 张图片 |
| 字符集字典 | ✅ 已生成 | 12 个字符 |
| PaddleOCR 工具 | ✅ 已存在 | `tools/PaddleOCR/` |
| 训练配置 | ✅ 已复制 | `tools/PaddleOCR/configs/rec/finetune_vincode.yml` |

---

## 🚀 开始训练

### 步骤 1：运行训练脚本

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle
tools\run_train.bat
```

### 步骤 2：选择训练模式

脚本会显示：
```
选择训练模式:
1. CPU 训练 (较慢，不需要 GPU)
2. GPU 训练 (快速，需要 NVIDIA 显卡)
```

**推荐选择：2（GPU 训练）**

### 步骤 3：输入训练轮数

```
训练轮数 [推荐 10，输入回车使用默认]:
```

直接按回车使用默认的 10 轮训练。

---

## ⏱️ 训练过程

脚本会执行以下命令：
```bash
python tools/train.py \
  -c configs/rec/finetune_vincode.yml \
  -o Global.use_gpu=true \
     Global.epoch_num=10 \
     Global.pretrained_model=../../padddle/ch_PP-OCRv4_rec_infer/inference.pdmodel \
     Global.character_dict_path=ppocr/utils/dict/vincode_full.txt \
     Global.save_model_dir=../../output/finetune_vincode \
     Train.loader.batch_size_per_card=8 \
     Train.loader.num_workers=2
```

### 训练日志示例：

```
epoch 1/10, iter 1/7, loss: 3.2456, lr: 0.00004
epoch 1/10, iter 2/7, loss: 2.8765, lr: 0.00007
epoch 1/10, iter 3/7, loss: 2.5432, lr: 0.00010
...
Eval result: acc: 0.9500
save model in ../../output/finetune_vincode/iter_50.pdparams
epoch 2/10, iter 1/7, loss: 2.1234, lr: 0.00008
...
Eval result: acc: 0.9800
save model in ../../output/finetune_vincode/best_accuracy.pdparams
...
```

### 训练时间估算：

| 模式 | 训练时间 |
|------|----------|
| CPU | 1-2 小时 |
| GPU | 10-30 分钟 |

---

## ✅ 训练完成后的步骤

### 步骤 4：导出并转换模型

训练完成后，运行：

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle
python tools\step5_export.py
```

这个脚本会自动：
1. ✓ 导出推理模型
2. ✓ 转换为 PaddleLite .nb 格式
3. ✓ 部署到 Android assets

### 步骤 5：更新 Android 代码

打开 `app/src/main/java/com/seuic/floating/network/PaddleOCRService.kt`，修改：

```kotlin
// 第 29 行，改为新的字典文件名
private const val DICT_FILE = "full_charset.txt"
```

### 步骤 6：重新编译安装

```bash
cd app
./gradlew clean assembleDebug
./gradlew installDebug
```

---

## 🔍 故障排查

### Q1: 运行 run_train.bat 提示"python不是内部或外部命令"

**解决方法：**
检查 Python 是否已安装并添加到 PATH。

运行：
```bash
python --version
```

### Q2: 导入 paddle 失败

**解决方法：**
先激活虚拟环境：
```bash
.venv_paddle2\Scripts\activate
```

然后再次运行训练脚本。

### Q3: 显存不足错误

**解决方法：**
修改批大小为 4：
- 编辑 `tools/run_train.bat`
- 找到 `set batch_size=8`
- 改为 `set batch_size=4`

### Q4: 训练准确率很低

**可能原因：**
1. 字符集不匹配（检查 vincode_full.txt）
2. 训练数据格式错误（检查 train_list.txt）
3. 预训练模型路径错误（检查 padddle 目录）

---

## 📝 训练参数说明

如果需要调整参数，可以修改 `tools/run_train.bat`：

```batch
set use_gpu=true      # 是否使用 GPU
set batch_size=8        # 批大小（显存不足改 4）
set epochs=10            # 训练轮数
```

---

## 📖 完整流程图

```
已完成的准备工作
        ↓
运行训练脚本 (tools\run_train.bat)
        ↓
选择模式 (GPU/CPU)
        ↓
执行训练 (10 epochs)
        ↓
训练完成 → best_accuracy.pdparams
        ↓
导出推理模型 (tools\step5_export.py)
        ↓
转换为 .nb 模型
        ↓
部署到 Android
        ↓
更新代码 → 重新编译 → 测试
```

---

## 🎯 现在开始吧！

运行以下命令开始训练：

```bash
cd E:\xuhrWorkProject\floating_ocr_ai_paddle
tools\run_train.bat
```

训练完成后，继续运行：

```bash
python tools\step5_export.py
```

---

需要帮助？查看详细教程：
```
docs\STEP_BY_STEP_GUIDE.md
```
