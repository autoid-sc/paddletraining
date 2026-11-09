# PP-OCRv6 CCPD 车牌微调训练 - 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `gpu_server_plate_training_v2/` 目录下创建一套完整、可直接部署到 GPU 服务器的 PP-OCRv6 车牌微调训练文件（脚本 + 配置 + 一键执行 + 说明文档）。

**Architecture:** 所有文件放在 `gpu_server_plate_training_v2/` 下，用户将该目录整体上传到 GPU 服务器的 `/data/plate_train_data/`，然后执行 `run_full_training.sh` 或按 README 分步执行。配置文件修复了旧版的 5 个问题（字典 73 字符统一、RecAug 参数、det 预训练名、eval_batch_step、batch_size 适配 10GB）。

**Tech Stack:** Python 3, PaddlePaddle GPU, PaddleOCR (PP-OCRv6), PaddleLite v2.12, Bash

**Spec:** `docs/superpowers/specs/2026-07-27-ppocrv6-ccpd-finetune-design.md`

---

## File Structure

| 文件 | 职责 |
|------|------|
| `gpu_server_plate_training_v2/prepare_ccpd_data.py` | CCPD 数据解析、车牌裁剪、标注生成、73 字符字典生成 |
| `gpu_server_plate_training_v2/configs/plate_det_tiny_gpu.yml` | 检测模型训练配置（DB + PPLCNetV4 tiny） |
| `gpu_server_plate_training_v2/configs/plate_rec_small_gpu.yml` | 识别模型训练配置（SVTR_LCNet small + MultiHead） |
| `gpu_server_plate_training_v2/fix_pdmodel.py` | PaddleLite 兼容性补丁（修复 swish/hardsigmoid 属性） |
| `gpu_server_plate_training_v2/run_full_training.sh` | 一键执行全流程脚本（环境清理→数据→训练→评估→导出→转nb） |
| `gpu_server_plate_training_v2/README.md` | 使用说明（更新旧版，指向新方案） |

---

### Task 1: 创建数据处理脚本 prepare_ccpd_data.py

**Files:**
- Create: `gpu_server_plate_training_v2/prepare_ccpd_data.py`

- [ ] **Step 1: 创建脚本文件**

写入完整内容（来自 spec 5.1 节），关键点：
- 73 字符字典（31 省 + 24 字母 + 10 数字 + 8 特殊）
- 末尾保留空行（PaddleOCR 要求）
- 裁剪车牌加 2px margin
- det 标注引用原图绝对路径（不复制图片）
- rec 标注用相对路径 `crop_images/crop_xxxxxx.jpg`
- 9:1 划分 train/val

- [ ] **Step 2: 验证 Python 语法**

Run: `python -c "import ast; ast.parse(open('gpu_server_plate_training_v2/prepare_ccpd_data.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: 验证字典字符数**

Run: `python -c "import sys; sys.path.insert(0,'gpu_server_plate_training_v2'); from prepare_ccpd_data import provinces, alphabets; print(len(provinces), len(alphabets), 31+24+10+8)"`
Expected: `31 24 73`

---

### Task 2: 创建检测配置 plate_det_tiny_gpu.yml

**Files:**
- Create: `gpu_server_plate_training_v2/configs/plate_det_tiny_gpu.yml`

- [ ] **Step 1: 创建配置文件**

写入完整内容（来自 spec 6.1 节），关键修正点：
- `pretrained_model: /data/plate_train_data/pretrained/PP-OCRv6_tiny_det_pretrained`（非旧的 `detect_tiny_pretrained`）
- `batch_size_per_card: 4`（适配 10GB）
- `eval_batch_step: [0, 200]`
- `data_dir: /`（label 里用绝对路径引用原图）
- `save_model_dir: /data/plate_train_data/output/plate_det_tiny`

- [ ] **Step 2: 验证 YAML 语法**

Run: `python -c "import yaml; yaml.safe_load(open('gpu_server_plate_training_v2/configs/plate_det_tiny_gpu.yml')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: 验证关键配置项**

Run: `python -c "import yaml; d=yaml.safe_load(open('gpu_server_plate_training_v2/configs/plate_det_tiny_gpu.yml')); assert d['Train']['loader']['batch_size_per_card']==4; assert 'PP-OCRv6_tiny_det_pretrained' in d['Global']['pretrained_model']; print('OK')"`
Expected: `OK`

---

### Task 3: 创建识别配置 plate_rec_small_gpu.yml

**Files:**
- Create: `gpu_server_plate_training_v2/configs/plate_rec_small_gpu.yml`

- [ ] **Step 1: 创建配置文件**

写入完整内容（来自 spec 6.2 节），关键修正点：
- `RecAug:`（无参数，官方写法，非旧的自定义 7 个 prob 参数）
- `batch_size_per_card: 32`（适配 10GB，非旧的 64）
- `eval_batch_step: [0, 500]`（非旧的 [0, 200]）
- `character_dict_path: /data/plate_train_data/plate_dict.txt`
- `use_space_char: false`
- `image_shape: [3, 48, 160]`

- [ ] **Step 2: 验证 YAML 语法**

Run: `python -c "import yaml; yaml.safe_load(open('gpu_server_plate_training_v2/configs/plate_rec_small_gpu.yml')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: 验证关键修正项**

Run: `python -c "import yaml; d=yaml.safe_load(open('gpu_server_plate_training_v2/configs/plate_rec_small_gpu.yml')); t=d['Train']['dataset']['transforms']; rec=[x for x in t if 'RecAug' in x][0]; assert rec['RecAug'] is None; assert d['Train']['loader']['batch_size_per_card']==32; assert d['Global']['eval_batch_step']==[0,500]; print('OK')"`
Expected: `OK`

---

### Task 4: 创建 pdmodel 补丁脚本 fix_pdmodel.py

**Files:**
- Create: `gpu_server_plate_training_v2/fix_pdmodel.py`

- [ ] **Step 1: 创建脚本文件**

写入完整内容（来自 spec 11.1 节），关键点：
- 修复 `swish` 算子缺少 `beta` 属性
- 修复 `hard_sigmoid`/`hardsigmoid` 缺少 `slope`/`offset` 属性
- 接受命令行参数（pdmodel 文件路径）

- [ ] **Step 2: 验证 Python 语法**

Run: `python -c "import ast; ast.parse(open('gpu_server_plate_training_v2/fix_pdmodel.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: 验证脚本可执行（dry run）**

Run: `python gpu_server_plate_training_v2/fix_pdmodel.py --help` 或 `python gpu_server_plate_training_v2/fix_pdmodel.py nonexistent.pdmodel`
Expected: 输出用法或报错（不是语法错误）

---

### Task 5: 创建一键执行脚本 run_full_training.sh

**Files:**
- Create: `gpu_server_plate_training_v2/run_full_training.sh`

- [ ] **Step 1: 创建脚本文件**

写入完整内容，包含 9 个阶段：
1. 环境清理（rm 旧 output/data）
2. 创建目录 + 下载预训练权重
3. 数据准备（python prepare_ccpd_data.py）
4. 部署配置到 PaddleOCR configs 目录
5. 检测训练
6. 识别训练
7. 评估
8. 导出推理模型
9. fix_pdmodel + 转 .nb

关键点：
- 开头 `#!/bin/bash` + `set -e`（任何步骤失败立即停止）
- 用变量定义路径（PADDLE_OCR_DIR, DATA_DIR, CCPD_DIR）
- 每个阶段有 echo 分隔标题
- 支持 `--skip-det` / `--skip-rec` 参数跳过已完成的阶段

- [ ] **Step 2: 验证 bash 语法**

Run: `bash -n gpu_server_plate_training_v2/run_full_training.sh`
Expected: 无输出（语法正确）

- [ ] **Step 3: 验证脚本可执行权限**

Run: `chmod +x gpu_server_plate_training_v2/run_full_training.sh && ls -l gpu_server_plate_training_v2/run_full_training.sh`
Expected: 文件有 x 权限

---

### Task 6: 更新 README.md

**Files:**
- Modify: `gpu_server_plate_training_v2/README.md`

- [ ] **Step 1: 重写 README**

用新的方案概要替换旧内容，关键点：
- 指向 spec 文档获取完整细节
- 列出 5 个关键修复
- 快速开始命令（一键脚本 + 分步命令）
- 文件清单
- 验收标准

- [ ] **Step 2: 验证文件存在且非空**

Run: `test -s gpu_server_plate_training_v2/README.md && echo OK`
Expected: `OK`

---

## Self-Review

### 1. Spec coverage

| Spec 章节 | 对应 Task |
|-----------|-----------|
| 阶段 0: 环境清理 | Task 5 (run_full_training.sh 阶段1) + Task 6 (README 说明) |
| 阶段 1: 目录 & 预训练权重 | Task 5 (阶段2) |
| 阶段 2: 数据准备 | Task 1 (脚本) + Task 5 (阶段3) |
| 阶段 3: 部署配置 | Task 2 + Task 3 + Task 5 (阶段4) |
| 阶段 4: 检测训练 | Task 5 (阶段5) |
| 阶段 5: 识别训练 | Task 5 (阶段6) |
| 阶段 6: 评估 | Task 5 (阶段7) |
| 阶段 7: 导出 | Task 5 (阶段8) |
| 阶段 8: 转 .nb | Task 4 (脚本) + Task 5 (阶段9) |
| 阶段 9: Android 部署 | Task 6 (README 说明) |
| 73 字典论证 | Task 1 (脚本内嵌) + Task 6 (README 说明) |
| 5 个问题修复 | Task 2 + Task 3 (配置内嵌) + Task 6 (README 列出) |

无遗漏。

### 2. Placeholder scan

无 TBD/TODO。所有文件内容来自 spec，已在 spec 中完整给出。

### 3. Type consistency

- `plate_dict.txt` 在 Task 1 脚本中生成，在 Task 3 配置中引用，路径 `/data/plate_train_data/plate_dict.txt` 一致。
- `PP-OCRv6_tiny_det_pretrained` 在 Task 2 配置和 Task 5 脚本中名称一致。
- `PP-OCRv6_small_rec_pretrained` 在 Task 3 配置和 Task 5 脚本中名称一致。
- `batch_size_per_card: 32` 在 Task 3 配置和 spec 中一致。
- `batch_size_per_card: 4` 在 Task 2 配置和 spec 中一致。

无类型/命名不一致。
