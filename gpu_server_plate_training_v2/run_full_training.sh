#!/bin/bash
# =============================================================================
# PP-OCRv6 CCPD 车牌微调训练 - 一键执行脚本
# 适用: 单卡 GPU (显存 >= 10GB)
# 用法:
#   ./run_full_training.sh              # 执行全流程
#   ./run_full_training.sh --skip-det   # 跳过检测训练
#   ./run_full_training.sh --skip-rec   # 跳过识别训练
#   ./run_full_training.sh --skip-det --skip-rec  # 仅数据准备+导出(需已有模型)
# =============================================================================
set -e

# ---------- 日志系统 ----------
LOG_DIR="/data/plate_train_data/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/training_$(date +%Y%m%d_%H%M%S).log"
TRAIN_START_TIME=$(date +%s)

# 所有输出同时写到终端和日志文件
exec > >(tee -a "$LOG_FILE") 2>&1

log_section() {
    echo ""
    echo "################################################################"
    echo "# $1"
    echo "# $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"
}

log_diagnostic() {
    echo ""
    echo "========== 诊断信息 =========="
    echo "--- 环境 ---"
    echo "hostname: $(hostname)"
    echo "date: $(date '+%Y-%m-%d %H:%M:%S')"
    python3 -c "import paddle; print(f'paddle: {paddle.__version__}, cuda: {paddle.is_compiled_with_cuda()}, gpu_count: {paddle.device.cuda.device_count()}')" 2>/dev/null || echo "paddle: N/A"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi: N/A"
    echo "--- 配置 ---"
    echo "PADDLE_OCR_DIR: $PADDLE_OCR_DIR"
    echo "DATA_DIR: $DATA_DIR"
    echo "CCPD_DIR: $CCPD_DIR"
    echo "GPU_IDS: $GPU_IDS"
    echo "SCRIPT_DIR: $SCRIPT_DIR"
    echo "LOG_FILE: $LOG_FILE"
    echo "==============================="
    echo ""
}

log_data_stats() {
    echo ""
    echo "========== 数据统计 =========="
    echo "--- 字典 ---"
    if [ -f "$DATA_DIR/plate_dict.txt" ]; then
        echo "plate_dict.txt: $(grep -c . "$DATA_DIR/plate_dict.txt") 字符, $(wc -l < "$DATA_DIR/plate_dict.txt") 行"
        echo "前5字符: $(head -5 "$DATA_DIR/plate_dict.txt" | tr '\n' ' ')"
        echo "后5字符: $(tail -5 "$DATA_DIR/plate_dict.txt" | tr '\n' ' ')"
    else
        echo "plate_dict.txt: 不存在!"
    fi
    echo "--- 标注文件 ---"
    for f in det_train_list.txt det_val_list.txt rec_train_list.txt rec_val_list.txt; do
        if [ -f "$DATA_DIR/$f" ]; then
            echo "$f: $(wc -l < "$DATA_DIR/$f") 行"
        else
            echo "$f: 不存在!"
        fi
    done
    echo "--- 裁剪图片 ---"
    if [ -d "$DATA_DIR/rec/crop_images" ]; then
        echo "crop_images: $(ls "$DATA_DIR/rec/crop_images/" | wc -l) 张"
        echo "抽样3张尺寸:"
        for img in $(ls "$DATA_DIR/rec/crop_images/" | head -3); do
            python3 -c "import cv2; im=cv2.imread('$DATA_DIR/rec/crop_images/$img'); print(f'  $img: {im.shape[1]}x{im.shape[0]}')" 2>/dev/null || echo "  $img: 读取失败"
        done
    fi
    echo "--- 预训练权重 ---"
    ls -lh "$DATA_DIR/pretrained/" 2>/dev/null || echo "pretrained/: 不存在!"
    echo "==============================="
    echo ""
}

log_training_result() {
    local model_name=$1
    local output_dir="$DATA_DIR/output/$model_name"
    echo ""
    echo "========== 训练结果: $model_name =========="
    if [ -f "$output_dir/train.log" ]; then
        echo "--- 最后10条训练指标 ---"
        grep -E "acc|hmean|norm_edit" "$output_dir/train.log" | tail -10
        echo "--- best metric ---"
        grep "best metric" "$output_dir/train.log" | tail -3
        echo "--- cur metric (最后3次验证) ---"
        grep "cur metric" "$output_dir/train.log" | tail -3
    else
        echo "train.log 不存在!"
    fi
    echo "--- best_accuracy.states ---"
    if [ -f "$output_dir/best_accuracy.states" ]; then
        python3 -c "
import pickle
with open('$output_dir/best_accuracy.states', 'rb') as f:
    s = pickle.load(f)
print(s)
" 2>/dev/null || cat "$output_dir/best_accuracy.states"
    fi
    echo "--- 模型文件 ---"
    ls -lh "$output_dir/"*.pdparams 2>/dev/null | tail -5
    echo "============================================"
    echo ""
}

log_export_result() {
    local model_name=$1
    local infer_dir="$DATA_DIR/output/${model_name}_inference"
    echo ""
    echo "========== 导出结果: $model_name =========="
    if [ -d "$infer_dir" ]; then
        ls -lh "$infer_dir/"
        echo "--- inference.yml ---"
        cat "$infer_dir/inference.yml" 2>/dev/null || echo "inference.yml 不存在"
    else
        echo "$infer_dir 不存在!"
    fi
    echo "==========================================="
    echo ""
}

log_nb_result() {
    echo ""
    echo "========== .nb 文件 =========="
    ls -lh "$DATA_DIR/output/"*.nb 2>/dev/null || echo "无 .nb 文件"
    echo "--- opt 工具版本 ---"
    if [ -x "$DATA_DIR/opt_v2.14" ]; then
        "$DATA_DIR/opt_v2.14" --version 2>&1 | head -1
    fi
    echo "==============================="
    echo ""
}

log_summary() {
    local end_time=$(date +%s)
    local elapsed=$((end_time - TRAIN_START_TIME))
    local hours=$((elapsed / 3600))
    local minutes=$(((elapsed % 3600) / 60))
    echo ""
    echo "################################################################"
    echo "# 训练全流程完成"
    echo "# 总耗时: ${hours}小时${minutes}分钟 (${elapsed}秒)"
    echo "# 日志文件: $LOG_FILE"
    echo "# $(date '+%Y-%m-%d %H:%M:%S')"
    echo "################################################################"
}

# 记录诊断信息
log_diagnostic

# ---------- 变量定义(按实际环境修改) ----------
PADDLE_OCR_DIR="/data/PaddleOCR"          # PaddleOCR 仓库路径
DATA_DIR="/data/plate_train_data"         # 训练数据根目录
CCPD_DIR="/data/CCPD2019"                 # CCPD2019 原始数据集
GPU_IDS="1,2,3,4"                       # GPU 编号(多卡逗号分隔,单卡写 "0")
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # 本脚本所在目录

# 预训练模型 URL
REC_PRETRAINED_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_rec_pretrained.pdparams"
DET_PRETRAINED_URL="https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv6_small_det_pretrained.pdparams"

# ---------- 参数解析 ----------
SKIP_DET=false
SKIP_REC=false
CLEAN_ONLY=false
for arg in "$@"; do
    case $arg in
        --skip-det) SKIP_DET=true ;;
        --skip-rec) SKIP_REC=true ;;
        --clean-only) CLEAN_ONLY=true ;;
        *) echo "未知参数: $arg"; exit 1 ;;
    esac
done

# ---------- 仅清理模式 ----------
if [ "$CLEAN_ONLY" = true ]; then
    echo "============================================================"
    echo "清理模式：删除所有训练输出、处理数据、下载的工具"
    echo "============================================================"
    echo ""
    echo "将删除以下内容:"
    echo "  $DATA_DIR/output/                 (训练输出 + 推理模型 + .nb)"
    echo "  $DATA_DIR/rec/                     (裁剪的车牌图片)"
    echo "  $DATA_DIR/det/                     (检测数据目录)"
    echo "  $DATA_DIR/det_train_list.txt       (检测训练标注)"
    echo "  $DATA_DIR/det_val_list.txt         (检测验证标注)"
    echo "  $DATA_DIR/rec_train_list.txt       (识别训练标注)"
    echo "  $DATA_DIR/rec_val_list.txt         (识别验证标注)"
    echo "  $DATA_DIR/plate_dict.txt           (字典文件)"
    echo "  $DATA_DIR/prepare_ccpd_data.py     (数据处理脚本)"
    echo "  $DATA_DIR/fix_pdmodel.py           (pdmodel 补丁脚本)"
    echo "  $DATA_DIR/inference_lite_lib*/     (PaddleLite 工具)"
    echo "  $DATA_DIR/paddle_*.tar.gz          (PaddleLite 压缩包)"
    echo "  $DATA_DIR/opt_*                    (PaddleLite opt 工具)"
    echo ""
    read -p "确认删除? [y/N]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消。"
        exit 0
    fi

    rm -rf "$DATA_DIR/output"
    rm -rf "$DATA_DIR/rec"
    rm -rf "$DATA_DIR/det"
    rm -f "$DATA_DIR/det_train_list.txt"
    rm -f "$DATA_DIR/det_val_list.txt"
    rm -f "$DATA_DIR/rec_train_list.txt"
    rm -f "$DATA_DIR/rec_val_list.txt"
    rm -f "$DATA_DIR/plate_dict.txt"
    rm -f "$DATA_DIR/prepare_ccpd_data.py"
    rm -f "$DATA_DIR/fix_pdmodel.py"
    rm -rf "$DATA_DIR/inference_lite_lib"*
    rm -f "$DATA_DIR/paddle_"*.tar.gz
    rm -f "$DATA_DIR/opt_"*
    # 注: pretrained/ 目录保留(预训练权重可复用)
    echo ""
    echo "[OK] 清理完成。预训练权重保留在 $DATA_DIR/pretrained/"
    echo "     如需重新训练，直接运行本脚本（不加 --clean-only）即可。"
    exit 0
fi

echo "============================================================"
echo "PP-OCRv6 CCPD 车牌微调训练"
echo "PaddleOCR: $PADDLE_OCR_DIR"
echo "Data:      $DATA_DIR"
echo "CCPD:      $CCPD_DIR"
echo "GPU:       $GPU_IDS"
echo "Skip det:  $SKIP_DET"
echo "Skip rec:  $SKIP_REC"
echo "============================================================"

# ---------- 阶段 1: 环境清理 ----------
echo ""
echo "========== [1/9] 环境清理 =========="
# 清理旧输出(消除上次训练残留)
rm -rf "$DATA_DIR/output"
rm -rf "$PADDLE_OCR_DIR/output"
echo "[OK] 已清理旧 output 目录"

# 校验 PaddleOCR 仓库
if [ ! -d "$PADDLE_OCR_DIR" ]; then
    echo "[ERROR] PaddleOCR 目录不存在: $PADDLE_OCR_DIR"
    exit 1
fi
if [ ! -f "$PADDLE_OCR_DIR/tools/train.py" ]; then
    echo "[ERROR] PaddleOCR/tools/train.py 不存在，请检查仓库完整性"
    exit 1
fi
echo "[OK] PaddleOCR 仓库校验通过"

# 校验 PaddlePaddle
python3 -c "import paddle; assert paddle.is_compiled_with_cuda(), 'PaddlePaddle 不是 GPU 版!'; print(f'[OK] PaddlePaddle {paddle.__version__}, GPU count: {paddle.device.cuda.device_count()}')"

# 校验 CCPD 数据集
if [ ! -d "$CCPD_DIR" ]; then
    echo "[ERROR] CCPD 数据集不存在: $CCPD_DIR"
    exit 1
fi
echo "[OK] CCPD 数据集存在"

# ---------- 阶段 2: 创建目录 & 下载预训练权重 ----------
echo ""
echo "========== [2/9] 创建目录 & 下载预训练权重 =========="
mkdir -p "$DATA_DIR/pretrained"
mkdir -p "$DATA_DIR/rec/crop_images"
mkdir -p "$DATA_DIR/output"

# 优先从本地 pretrained/ 目录复制(本机下载好一并上传,避免服务器下载)
LOCAL_PRETRAINED_DIR="$SCRIPT_DIR/pretrained"
if [ -d "$LOCAL_PRETRAINED_DIR" ]; then
    echo "检查本地预训练权重目录..."
    for f in "$LOCAL_PRETRAINED_DIR"/*.pdparams; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        if [ ! -f "$DATA_DIR/pretrained/$fname" ]; then
            echo "[COPY] 从本地复制 $fname"
            cp "$f" "$DATA_DIR/pretrained/$fname"
        else
            echo "[SKIP] $fname 已在目标目录"
        fi
    done
fi

# 下载并校验预训练权重(校验文件大小,防止下载不完整导致 EOFError)
download_pretrained() {
    local url=$1
    local dest=$2
    local min_size=$3  # 最小字节数

    if [ -f "$dest" ]; then
        local size
        size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
        if [ "$size" -ge "$min_size" ]; then
            echo "[SKIP] $(basename "$dest") 已存在 ($size bytes)"
            return 0
        fi
        echo "[WARN] $(basename "$dest") 文件不完整 ($size bytes < $min_size)，重新下载"
        rm -f "$dest"
    fi

    echo "下载 $(basename "$dest")..."
    wget -c -O "$dest" "$url"

    local new_size
    new_size=$(stat -c%s "$dest" 2>/dev/null || echo 0)
    if [ "$new_size" -lt "$min_size" ]; then
        echo "[ERROR] 下载失败: $(basename "$dest") 大小 $new_size < $min_size bytes"
        echo "        请检查网络后重新运行本脚本"
        exit 1
    fi
    echo "[OK] $(basename "$dest") 下载完成 ($new_size bytes)"
}

# rec 预训练权重 ~119MB, 最小要求 100MB
download_pretrained "$REC_PRETRAINED_URL" \
    "$DATA_DIR/pretrained/PP-OCRv6_small_rec_pretrained.pdparams" \
    100000000

# det 预训练权重 ~11MB (PPLCNetV4 small), 最小要求 8MB
download_pretrained "$DET_PRETRAINED_URL" \
    "$DATA_DIR/pretrained/PP-OCRv6_small_det_pretrained.pdparams" \
    8000000

echo "预训练权重文件:"
ls -lh "$DATA_DIR/pretrained/"

# ---------- 阶段 3: 数据准备 ----------
echo ""
echo "========== [3/9] 数据准备 (CCPD 处理) =========="
if [ ! -f "$DATA_DIR/plate_dict.txt" ]; then
    echo "运行 prepare_ccpd_data.py (首次执行约 30-60 分钟)..."
    cp "$SCRIPT_DIR/prepare_ccpd_data.py" "$DATA_DIR/prepare_ccpd_data.py"
    cd "$DATA_DIR"
    python3 prepare_ccpd_data.py \
        --ccpd_dir "$CCPD_DIR" \
        --output_dir "$DATA_DIR" \
        --max_det 30000 \
        --max_rec 200000 \
        --max_per_prov 5000
else
    echo "[SKIP] 数据已准备 (plate_dict.txt 已存在)"
fi

# 验证字典
DICT_LINES=$(wc -l < "$DATA_DIR/plate_dict.txt")
echo "字典行数: $DICT_LINES (预期 74 = 73 字符 + 1 空行)"
if [ "$DICT_LINES" -ne 74 ]; then
    echo "[ERROR] 字典行数不对! 预期 74, 实际 $DICT_LINES"
    exit 1
fi
echo "[OK] 字典验证通过 (73 字符, 不含·)"
log_data_stats

# ---------- 阶段 4: 部署配置文件 ----------
echo ""
echo "========== [4/9] 部署配置文件到 PaddleOCR =========="
mkdir -p "$PADDLE_OCR_DIR/configs/det/PP-OCRv6"
mkdir -p "$PADDLE_OCR_DIR/configs/rec/PP-OCRv6"
cp "$SCRIPT_DIR/configs/plate_det_small_gpu.yml" "$PADDLE_OCR_DIR/configs/det/PP-OCRv6/"
cp "$SCRIPT_DIR/configs/plate_rec_small_gpu.yml" "$PADDLE_OCR_DIR/configs/rec/PP-OCRv6/"
echo "[OK] 配置文件已部署"

# ---------- 训练配置概览 ----------
echo ""
echo "=================== 训练配置概览 (v4 微调) ==================="
echo "  基础模型: PP-OCRv6 small det+rec (官方预训练, 已能识别大部分车牌)"
echo "  微调目标: 提高车牌精度+专业度, 字典变小→推理更快"
echo "  检测 epoch: 50   | 学习率: 0.0005   | batch: 8/GPU"
echo "  识别 epoch: 5    | 学习率: 0.0001   | batch: 16/GPU"
echo "  识别 L2: 5e-5    | warmup: 1 epoch  | eval: 每100step"
echo "  GPU: $GPU_IDS ($(echo $GPU_IDS | tr ',' '\n' | wc -l) 卡)"
echo "  字典: 73 字符 (31省+24字母+10数字+8特殊, 无·)"
echo "  数据: 平衡采样(每省≤5000, ~15万张) + 透视矫正 + 彩色图"
echo "  字典文件: $DATA_DIR/plate_dict.txt"
echo "====================================================="
echo ""
# ---------- 阶段 5: 检测模型训练 ----------
echo ""
echo "========== [5/9] 检测模型训练 =========="
if [ "$SKIP_DET" = true ]; then
    echo "[SKIP] 检测训练已跳过"
else
    cd "$PADDLE_OCR_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m paddle.distributed.launch --gpus "$GPU_IDS" tools/train.py \
        -c configs/det/PP-OCRv6/plate_det_small_gpu.yml \
        -o Global.pretrained_model="$DATA_DIR/pretrained/PP-OCRv6_small_det_pretrained"
    echo "[OK] 检测训练完成"
    log_training_result "plate_det_small"
fi

# ---------- 阶段 6: 识别模型训练 ----------
echo ""
echo "========== [6/9] 识别模型训练 =========="
if [ "$SKIP_REC" = true ]; then
    echo "[SKIP] 识别训练已跳过"
else
    cd "$PADDLE_OCR_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m paddle.distributed.launch --gpus "$GPU_IDS" tools/train.py \
        -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
        -o Global.pretrained_model="$DATA_DIR/pretrained/PP-OCRv6_small_rec_pretrained"
    echo "[OK] 识别训练完成"
    log_training_result "plate_rec_small"
fi

# ---------- 阶段 7: 评估 ----------
echo ""
echo "========== [7/9] 模型评估 =========="
cd "$PADDLE_OCR_DIR"

if [ -f "$DATA_DIR/output/plate_det_small/best_accuracy.pdparams" ]; then
    echo "--- 检测模型评估 ---"
    CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py \
        -c configs/det/PP-OCRv6/plate_det_small_gpu.yml \
        -o Global.checkpoints="$DATA_DIR/output/plate_det_small/best_accuracy"
else
    echo "[WARN] 检测模型 best_accuracy 不存在，跳过评估"
fi

if [ -f "$DATA_DIR/output/plate_rec_small/best_accuracy.pdparams" ]; then
    echo "--- 识别模型评估 ---"
    CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py \
        -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
        -o Global.checkpoints="$DATA_DIR/output/plate_rec_small/best_accuracy"
else
    echo "[WARN] 识别模型 best_accuracy 不存在，跳过评估"
fi

# ---------- 阶段 8: 导出推理模型 ----------
echo ""
echo "========== [8/9] 导出推理模型 =========="
cd "$PADDLE_OCR_DIR"

if [ -f "$DATA_DIR/output/plate_det_small/best_accuracy.pdparams" ]; then
    echo "--- 导出检测推理模型 ---"
    CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py \
        -c configs/det/PP-OCRv6/plate_det_small_gpu.yml \
        -o Global.checkpoints="$DATA_DIR/output/plate_det_small/best_accuracy" \
        Global.save_inference_dir="$DATA_DIR/output/plate_det_small_inference" \
        Global.export_with_pir=False \
        Global.infer_mode=True
    echo "[OK] 检测推理模型已导出"
    log_export_result "plate_det_small"
else
    echo "[WARN] 检测模型不存在，跳过导出"
fi

if [ -f "$DATA_DIR/output/plate_rec_small/best_accuracy.pdparams" ]; then
    echo "--- 导出识别推理模型 ---"
    CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py \
        -c configs/rec/PP-OCRv6/plate_rec_small_gpu.yml \
        -o Global.checkpoints="$DATA_DIR/output/plate_rec_small/best_accuracy" \
        Global.save_inference_dir="$DATA_DIR/output/plate_rec_small_inference" \
        Global.export_with_pir=False \
        Global.infer_mode=True
    echo "[OK] 识别推理模型已导出"
    log_export_result "plate_rec_small"
else
    echo "[WARN] 识别模型不存在，跳过导出"
fi

# ---------- 阶段 9: 转 PaddleLite .nb ----------
echo ""
echo "========== [9/9] 转换为 PaddleLite .nb =========="
cp "$SCRIPT_DIR/fix_pdmodel.py" "$DATA_DIR/fix_pdmodel.py"

# PaddleLite opt 工具（必须与 Android 端运行时版本匹配！当前 Android 用 v2.14-rc）
OPT_BIN="$DATA_DIR/opt_v2.14"
if [ ! -x "$OPT_BIN" ]; then
    if [ -f "$SCRIPT_DIR/opt_linux_x86" ]; then
        echo "从包内复制 PaddleLite v2.14-rc opt 工具..."
        cp "$SCRIPT_DIR/opt_linux_x86" "$OPT_BIN"
        chmod +x "$OPT_BIN"
    else
        echo "下载 PaddleLite v2.14-rc opt 工具..."
        wget -O "$OPT_BIN" https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.14-rc/opt_linux_x86
        chmod +x "$OPT_BIN"
    fi
fi
echo "[OK] opt 工具: $("$OPT_BIN" --version 2>&1 | head -1)"
echo "[INFO] Android 运行时版本: v2.14-rc, opt 版本匹配!"

# 转换函数：先试 OPT_BIN，失败则回退到 pip 安装的 paddle_lite_opt
convert_nb() {
    local model_name=$1
    local infer_dir="$DATA_DIR/output/${model_name}_inference"
    local out_name="$DATA_DIR/output/${model_name}"

    if [ ! -f "$infer_dir/inference.pdmodel" ]; then
        echo "[WARN] $model_name 推理模型不存在，跳过"
        return 1
    fi

    echo "--- 修复 $model_name pdmodel ---"
    cd "$DATA_DIR"
    python3 fix_pdmodel.py "$infer_dir/inference.pdmodel"

    echo "--- 转换 $model_name 为 .nb (尝试 opt_v2.14) ---"
    if "$OPT_BIN" \
        --model_file "$infer_dir/inference.pdmodel" \
        --param_file "$infer_dir/inference.pdiparams" \
        --optimize_out "$out_name" \
        --valid_targets arm \
        --optimize_out_type naive_buffer 2>&1; then
        echo "[OK] $model_name .nb 已生成 (opt_v2.14)"
        return 0
    fi

    echo "[WARN] opt_v2.14 转换失败，尝试 pip paddle_lite_opt..."
    # 回退：pip 安装的 paddle_lite_opt
    if ! command -v paddle_lite_opt &>/dev/null; then
        echo "安装 paddlelite (pip)..."
        pip install paddlelite==2.13rc0 -q 2>/dev/null || true
    fi
    if command -v paddle_lite_opt &>/dev/null; then
        if paddle_lite_opt \
            --model_file "$infer_dir/inference.pdmodel" \
            --param_file "$infer_dir/inference.pdiparams" \
            --optimize_out "$out_name" \
            --valid_targets arm \
            --optimize_out_type naive_buffer 2>&1; then
            echo "[OK] $model_name .nb 已生成 (pip paddle_lite_opt)"
            return 0
        fi
    fi

    echo "[ERROR] $model_name 转换失败! Android 端将回退到通用检测模型"
    return 1
}

convert_nb "plate_det_small" || true
convert_nb "plate_rec_small" || true
log_nb_result

# ---------- 完成 ----------
echo ""
echo "============================================================"
echo "全流程完成!"
echo "============================================================"
echo ""
echo "输出文件:"
ls -lh "$DATA_DIR/output/"*.nb 2>/dev/null || echo "(无 .nb 文件)"
echo ""
echo "字典文件(部署到 Android 时必须用同一份):"
echo "  $DATA_DIR/plate_dict.txt"
echo ""
echo "部署到 Android:"
echo "  plate_det_small.nb  -> app/src/main/assets/models/detect/"
echo "  plate_rec_small.nb  -> app/src/main/assets/models/recognize/"
echo "  plate_dict.txt      -> app/src/main/assets/models/recognize/"
echo ""
echo "字典 MD5 (三处必须一致):"
md5sum "$DATA_DIR/plate_dict.txt"

log_summary
