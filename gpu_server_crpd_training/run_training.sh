#!/bin/bash
# =============================================================================
# PP-OCRv6 CRPD 车牌微调训练 - 一键执行脚本
# 数据集: CRPD (CRPD_single / CRPD_double / CRPD_multi)
# 模型: PP-OCRv6 small det + rec
# 适用: 单卡/多卡 GPU (显存 >= 10GB)
# 用法:
#   ./run_training.sh              # 全流程: 数据准备 + det训练 + rec训练 + 导出 + 转nb
#   ./run_training.sh --skip-det   # 跳过检测训练
#   ./run_training.sh --skip-rec   # 跳过识别训练
#   ./run_training.sh --clean-only # 清理训练输出
# =============================================================================
set -e

# ---------- 可配置参数 ----------
PADDLE_OCR_DIR="/data/PaddleOCR"          # PaddleOCR 仓库路径
DATA_DIR="/data/crpd_train"                # 训练数据根目录 (输出目录)
CRPD_DIR="/data/CRPD_all"                  # CRPD 原始数据集根目录 (含三个子集)
GPU_IDS="0,1,2,3"                          # GPU 编号 (多卡逗号分隔, 如 "0,1,2,3")

# 数据量上限 (0=不限制, 按需调整以防训练时间过长)
MAX_DET_TRAIN=0       # 检测训练集最大图数
MAX_DET_VAL=0         # 检测验证集最大图数
MAX_REC_TRAIN=0       # 识别训练集最大样本数
MAX_REC_VAL=5000      # 识别验证集最大样本数

# 牌型: 0蓝单行 1黄单行 2黄双行 3白
PLATE_TYPES="0,1,2,3"

# 用 test 集做验证 (CRPD 有独立的 test 和 val, 默认用 val)
USE_TEST_AS_VAL=false
# --------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$DATA_DIR/logs"
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
    echo "DATA_DIR:       $DATA_DIR"
    echo "CRPD_DIR:       $CRPD_DIR"
    echo "GPU_IDS:        $GPU_IDS"
    echo "PLATE_TYPES:    $PLATE_TYPES"
    echo "SCRIPT_DIR:     $SCRIPT_DIR"
    echo "LOG_FILE:       $LOG_FILE"
    echo "==============================="
    echo ""
}

log_data_stats() {
    echo ""
    echo "========== 数据统计 =========="
    echo "--- 字典 ---"
    if [ -f "$DATA_DIR/plate_dict.txt" ]; then
        echo "plate_dict.txt: $(wc -l < "$DATA_DIR/plate_dict.txt") 字符"
    else
        echo "plate_dict.txt: 不存在!"
    fi
    echo "--- 检测标注 ---"
    for f in det_train_list.txt det_val_list.txt; do
        if [ -f "$DATA_DIR/$f" ]; then
            echo "$f: $(wc -l < "$DATA_DIR/$f") 行"
        else
            echo "$f: 不存在!"
        fi
    done
    echo "--- 识别标注 ---"
    for f in rec_train_list.txt rec_val_list.txt; do
        if [ -f "$DATA_DIR/rec/$f" ]; then
            echo "$f: $(wc -l < "$DATA_DIR/rec/$f") 行"
        else
            echo "$f: 不存在!"
        fi
    done
    if [ -d "$DATA_DIR/rec/crop_images" ]; then
        echo "crop_images: $(ls "$DATA_DIR/rec/crop_images/" | wc -l) 张"
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
    echo "清理模式：删除所有训练输出和处理后的数据"
    echo "============================================================"
    echo ""
    echo "将删除以下内容:"
    echo "  $DATA_DIR/output/              (训练输出 + 推理模型 + .nb)"
    echo "  $DATA_DIR/rec/                  (裁剪的车牌图片 + 识别标注)"
    echo "  $DATA_DIR/det_train_list.txt    (检测训练标注)"
    echo "  $DATA_DIR/det_val_list.txt      (检测验证标注)"
    echo "  $DATA_DIR/logs/                 (训练日志)"
    echo ""
    echo "保留内容:"
    echo "  $DATA_DIR/pretrained/           (预训练权重)"
    echo "  $DATA_DIR/plate_dict.txt        (字典文件)"
    echo ""
    read -p "确认删除? [y/N]: " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "已取消。"
        exit 0
    fi

    rm -rf "$DATA_DIR/output"
    rm -rf "$DATA_DIR/rec"
    rm -f "$DATA_DIR/det_train_list.txt"
    rm -f "$DATA_DIR/det_val_list.txt"
    rm -rf "$DATA_DIR/logs"
    echo ""
    echo "[OK] 清理完成。"
    exit 0
fi

# ---------- 开始 ----------
log_diagnostic

echo "============================================================"
echo "PP-OCRv6 CRPD 车牌微调训练"
echo "PaddleOCR: $PADDLE_OCR_DIR"
echo "Data:      $DATA_DIR"
echo "CRPD:      $CRPD_DIR"
echo "GPU:       $GPU_IDS ($(echo $GPU_IDS | tr ',' '\n' | wc -l) 卡)"
echo "Skip det:  $SKIP_DET"
echo "Skip rec:  $SKIP_REC"
echo "============================================================"

# ---------- 阶段 1: 环境校验 ----------
log_section "[1/8] 环境校验"

# 清理旧输出
rm -rf "$DATA_DIR/output"
rm -rf "$PADDLE_OCR_DIR/output"
echo "[OK] 已清理旧 output 目录"

# 校验 PaddleOCR 仓库
if [ ! -d "$PADDLE_OCR_DIR" ]; then
    echo "[ERROR] PaddleOCR 目录不存在: $PADDLE_OCR_DIR"
    echo "        请先克隆: git clone https://github.com/PaddlePaddle/PaddleOCR.git $PADDLE_OCR_DIR"
    exit 1
fi
if [ ! -f "$PADDLE_OCR_DIR/tools/train.py" ]; then
    echo "[ERROR] PaddleOCR/tools/train.py 不存在，请检查仓库完整性"
    exit 1
fi
echo "[OK] PaddleOCR 仓库校验通过"

# 校验 PaddlePaddle
python3 -c "import paddle; assert paddle.is_compiled_with_cuda(), 'PaddlePaddle 不是 GPU 版!'; print(f'[OK] PaddlePaddle {paddle.__version__}, GPU count: {paddle.device.cuda.device_count()}')"

# 校验 CRPD 数据集
if [ ! -d "$CRPD_DIR" ]; then
    echo "[ERROR] CRPD 数据集不存在: $CRPD_DIR"
    exit 1
fi
found_any=false
for sub in CRPD_single CRPD_double CRPD_multi; do
    if [ -d "$CRPD_DIR/$sub" ]; then
        echo "[OK] $sub 存在"
        found_any=true
    else
        echo "[WARN] $sub 不存在 (将跳过)"
    fi
done
if [ "$found_any" = false ]; then
    echo "[ERROR] 未找到任何 CRPD 子集!"
    exit 1
fi

# ---------- 阶段 2: 目录 & 文件准备 ----------
log_section "[2/8] 目录 & 文件准备"
mkdir -p "$DATA_DIR/pretrained"
mkdir -p "$DATA_DIR/output"
mkdir -p "$DATA_DIR/rec"

# 复制预训练权重 (优先从本目录复制, 避免服务器下载)
if [ -d "$SCRIPT_DIR/pretrained" ]; then
    echo "从本地复制预训练权重..."
    for f in "$SCRIPT_DIR/pretrained"/*.pdparams; do
        [ -f "$f" ] || continue
        fname=$(basename "$f")
        if [ ! -f "$DATA_DIR/pretrained/$fname" ]; then
            cp "$f" "$DATA_DIR/pretrained/$fname"
            echo "  [COPY] $fname"
        else
            echo "  [SKIP] $fname 已存在"
        fi
    done
fi

# 校验预训练权重
for f in PP-OCRv6_small_det_pretrained.pdparams PP-OCRv6_small_rec_pretrained.pdparams; do
    if [ ! -f "$DATA_DIR/pretrained/$f" ]; then
        echo "[ERROR] 缺少预训练权重: $f"
        exit 1
    fi
    size=$(stat -c%s "$DATA_DIR/pretrained/$f" 2>/dev/null || echo 0)
    echo "  [OK] $f ($((size/1024/1024)) MB)"
done

# 复制字典
cp "$SCRIPT_DIR/plate_dict.txt" "$DATA_DIR/plate_dict.txt"
DICT_LINES=$(wc -l < "$DATA_DIR/plate_dict.txt")
echo "字典: $DICT_LINES 字符"

# 复制配置到 PaddleOCR
mkdir -p "$PADDLE_OCR_DIR/configs/det/PP-OCRv6"
mkdir -p "$PADDLE_OCR_DIR/configs/rec/PP-OCRv6"
cp "$SCRIPT_DIR/configs/crpd_det_small_gpu.yml" "$PADDLE_OCR_DIR/configs/det/PP-OCRv6/"
cp "$SCRIPT_DIR/configs/crpd_rec_small_gpu.yml" "$PADDLE_OCR_DIR/configs/rec/PP-OCRv6/"
echo "[OK] 配置文件已部署到 PaddleOCR"

# ---------- 阶段 3: 检测数据准备 ----------
log_section "[3/8] 检测数据准备"
if [ -f "$DATA_DIR/det_train_list.txt" ] && [ -f "$DATA_DIR/det_val_list.txt" ]; then
    echo "[SKIP] 检测数据已存在"
else
    cp "$SCRIPT_DIR/prepare_crpd_det_data.py" "$DATA_DIR/prepare_crpd_det_data.py"
    cd "$DATA_DIR"
    python3 prepare_crpd_det_data.py \
        --crpd_dir "$CRPD_DIR" \
        --output_dir "$DATA_DIR" \
        --max_train_per_subset $MAX_DET_TRAIN \
        --max_val_per_subset $MAX_DET_VAL \
        $($USE_TEST_AS_VAL && echo "--use_test_as_val")
    echo "[OK] 检测数据准备完成"
fi

# ---------- 阶段 4: 识别数据准备 ----------
log_section "[4/8] 识别数据准备"
if [ -f "$DATA_DIR/rec/rec_train_list.txt" ] && [ -f "$DATA_DIR/rec/rec_val_list.txt" ]; then
    echo "[SKIP] 识别数据已存在"
else
    cp "$SCRIPT_DIR/prepare_crpd_rec_data.py" "$DATA_DIR/prepare_crpd_rec_data.py"
    cd "$DATA_DIR"
    python3 prepare_crpd_rec_data.py \
        --crpd_dir "$CRPD_DIR" \
        --output_dir "$DATA_DIR/rec" \
        --max_train $MAX_REC_TRAIN \
        --max_val $MAX_REC_VAL \
        --types "$PLATE_TYPES" \
        --degrade_prob 0.5 \
        $($USE_TEST_AS_VAL && echo "--use_test_as_val")
    echo "[OK] 识别数据准备完成"
fi

log_data_stats

# ---------- 训练配置概览 ----------
echo ""
echo "=================== 训练配置概览 ==================="
echo "  基础模型: PP-OCRv6 small det + rec"
echo "  数据集:   CRPD (single + double + multi)"
echo "  字典:     $DICT_LINES 字符 (31省+24字母+10数字+8特殊)"
echo "  牌型:     $PLATE_TYPES (0蓝 1黄单 2黄双 3白)"
echo "  检测:     80 epoch | lr 0.0005 | batch 8/GPU"
echo "  识别:     10 epoch | lr 0.0001 | batch 16/GPU"
echo "  GPU:      $GPU_IDS"
echo "===================================================="
echo ""

# ---------- 阶段 5: 检测模型训练 ----------
log_section "[5/8] 检测模型训练"
if [ "$SKIP_DET" = true ]; then
    echo "[SKIP] 检测训练已跳过"
else
    cd "$PADDLE_OCR_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m paddle.distributed.launch --gpus "$GPU_IDS" tools/train.py \
        -c configs/det/PP-OCRv6/crpd_det_small_gpu.yml \
        -o Global.pretrained_model="$DATA_DIR/pretrained/PP-OCRv6_small_det_pretrained" \
        Global.save_model_dir="$DATA_DIR/output/crpd_det_small" \
        Train.dataset.label_file_list="['$DATA_DIR/det_train_list.txt']" \
        Eval.dataset.label_file_list="['$DATA_DIR/det_val_list.txt']"
    echo "[OK] 检测训练完成"
    log_training_result "crpd_det_small"
fi

# ---------- 阶段 6: 识别模型训练 ----------
log_section "[6/8] 识别模型训练"
if [ "$SKIP_REC" = true ]; then
    echo "[SKIP] 识别训练已跳过"
else
    cd "$PADDLE_OCR_DIR"
    CUDA_VISIBLE_DEVICES=$GPU_IDS python3 -m paddle.distributed.launch --gpus "$GPU_IDS" tools/train.py \
        -c configs/rec/PP-OCRv6/crpd_rec_small_gpu.yml \
        -o Global.pretrained_model="$DATA_DIR/pretrained/PP-OCRv6_small_rec_pretrained" \
        Global.save_model_dir="$DATA_DIR/output/crpd_rec_small" \
        Global.character_dict_path="$DATA_DIR/plate_dict.txt" \
        Train.dataset.data_dir="$DATA_DIR/rec/" \
        Train.dataset.label_file_list="['$DATA_DIR/rec/rec_train_list.txt']" \
        Eval.dataset.data_dir="$DATA_DIR/rec/" \
        Eval.dataset.label_file_list="['$DATA_DIR/rec/rec_val_list.txt']"
    echo "[OK] 识别训练完成"
    log_training_result "crpd_rec_small"
fi

# ---------- 阶段 7: 导出推理模型 ----------
log_section "[7/8] 导出推理模型"
cd "$PADDLE_OCR_DIR"

if [ -f "$DATA_DIR/output/crpd_det_small/best_accuracy.pdparams" ]; then
    echo "--- 导出检测推理模型 ---"
    CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py \
        -c configs/det/PP-OCRv6/crpd_det_small_gpu.yml \
        -o Global.checkpoints="$DATA_DIR/output/crpd_det_small/best_accuracy" \
        Global.save_inference_dir="$DATA_DIR/output/crpd_det_small_inference" \
        Global.export_with_pir=False \
        Global.infer_mode=True
    echo "[OK] 检测推理模型已导出"
    log_export_result "crpd_det_small"
else
    echo "[WARN] 检测模型不存在，跳过导出"
fi

if [ -f "$DATA_DIR/output/crpd_rec_small/best_accuracy.pdparams" ]; then
    echo "--- 导出识别推理模型 ---"
    CUDA_VISIBLE_DEVICES=0 python3 tools/export_model.py \
        -c configs/rec/PP-OCRv6/crpd_rec_small_gpu.yml \
        -o Global.checkpoints="$DATA_DIR/output/crpd_rec_small/best_accuracy" \
        Global.save_inference_dir="$DATA_DIR/output/crpd_rec_small_inference" \
        Global.character_dict_path="$DATA_DIR/plate_dict.txt" \
        Global.export_with_pir=False \
        Global.infer_mode=True
    echo "[OK] 识别推理模型已导出"
    log_export_result "crpd_rec_small"
else
    echo "[WARN] 识别模型不存在，跳过导出"
fi

# ---------- 阶段 8: 转 PaddleLite .nb ----------
log_section "[8/8] 转换为 PaddleLite .nb"
cp "$SCRIPT_DIR/fix_pdmodel.py" "$DATA_DIR/fix_pdmodel.py"

OPT_BIN="$DATA_DIR/opt_v2.14"
if [ ! -x "$OPT_BIN" ]; then
    if [ -f "$SCRIPT_DIR/opt_linux_x86" ]; then
        echo "从包内复制 PaddleLite v2.14 opt 工具..."
        cp "$SCRIPT_DIR/opt_linux_x86" "$OPT_BIN"
        chmod +x "$OPT_BIN"
    else
        echo "下载 PaddleLite v2.14 opt 工具..."
        wget -O "$OPT_BIN" https://github.com/PaddlePaddle/Paddle-Lite/releases/download/v2.14-rc/opt_linux_x86
        chmod +x "$OPT_BIN"
    fi
fi
echo "[OK] opt 工具: $("$OPT_BIN" --version 2>&1 | head -1)"

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

    echo "--- 转换 $model_name 为 .nb ---"
    if "$OPT_BIN" \
        --model_file "$infer_dir/inference.pdmodel" \
        --param_file "$infer_dir/inference.pdiparams" \
        --optimize_out "$out_name" \
        --valid_targets arm \
        --optimize_out_type naive_buffer 2>&1; then
        echo "[OK] $model_name .nb 已生成"
        return 0
    fi

    echo "[WARN] opt_v2.14 转换失败，尝试 pip paddle_lite_opt..."
    if ! command -v paddle_lite_opt &>/dev/null; then
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

    echo "[ERROR] $model_name .nb 转换失败!"
    return 1
}

convert_nb "crpd_det_small" || true
convert_nb "crpd_rec_small" || true
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
echo "  crpd_det_small.nb  -> app/src/main/assets/models/detect/"
echo "  crpd_rec_small.nb  -> app/src/main/assets/models/recognize/"
echo "  plate_dict.txt     -> app/src/main/assets/models/recognize/"
echo ""
echo "字典 MD5 (三处必须一致):"
md5sum "$DATA_DIR/plate_dict.txt"

log_summary
