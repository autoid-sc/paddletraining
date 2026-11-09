@echo off
chcp 65001 > nul
title PP-OCRv6 钢印模型训练控制台

echo ========================================================
echo        PP-OCRv6 钢印模型一键微调训练系统
echo ========================================================
echo.

set PROJECT_ROOT=E:\xuhrWorkProject\floating_ocr_ai_paddle
set PYTHON_EXE=%PROJECT_ROOT%\.venv_paddle2\Scripts\python.exe

:: 1. 运行数据预处理
echo [1/3] 正在检查并处理数据集 (划分为 9:1 并生成字典)...
"%PYTHON_EXE%" "%PROJECT_ROOT%\tools\prepare_train_data_stamp.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 数据预处理失败，请检查标注文件格式！
    pause
    exit /b %errorlevel%
)
echo [OK] 数据集准备完毕。
echo.

:: 2. 检查并下载预训练权重
echo [2/3] 正在检查预训练权重...
"%PYTHON_EXE%" "%PROJECT_ROOT%\tools\download_v6_pretrained.py"
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 预训练模型下载/检查失败！
    pause
    exit /b %errorlevel%
)
echo [OK] 预训练模型权重准备就绪。
echo.

:: 3. 选择训练任务
echo [3/3] 请选择您要启动的训练任务：
echo ========================================================
echo   [1] 启动 检测 (Detection) 模型微调训练
echo   [2] 启动 识别 (Recognition) 模型微调训练
echo   [3] 退出
echo ========================================================
set /p choice="请输入选择序号 (1/2/3): "

if "%choice%"=="1" (
    echo.
    echo [INFO] 正在启动检测模型微调训练...
    echo [TIPS] 默认使用 CPU 训练。如需使用 GPU 加速，可在训练命令行后追加 "-o Global.use_gpu=True"
    cd /d "%PROJECT_ROOT%\tools\PaddleOCR"
    "%PYTHON_EXE%" tools/train.py -c configs/det/PP-OCRv6/stamp_det_tiny.yml
    pause
) else if "%choice%"=="2" (
    echo.
    echo [INFO] 正在启动识别模型微调训练...
    echo [TIPS] 默认使用 CPU 训练。如需使用 GPU 加速，可在训练命令行后追加 "-o Global.use_gpu=True"
    cd /d "%PROJECT_ROOT%\tools\PaddleOCR"
    "%PYTHON_EXE%" tools/train.py -c configs/rec/PP-OCRv6/stamp_rec_small.yml
    pause
) else (
    echo 退出程序。
)
