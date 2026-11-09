@echo off
chcp 65001 > nul
title PP-OCRv6 钢印模型一键导出并部署工具

echo ========================================================
echo        PP-OCRv6 钢印模型一键导出并部署工具
echo ========================================================
echo.

set PROJECT_ROOT=%~dp0
:: Remove trailing backslash if exists
if "%PROJECT_ROOT:~-1%"=="\" set PROJECT_ROOT=%PROJECT_ROOT:~0,-1%

set PYTHON_EXE=%PROJECT_ROOT%\.venv_paddle2\Scripts\python.exe

if not exist "%PYTHON_EXE%" (
    echo [ERROR] 找不到 Python 解释器：%PYTHON_EXE%
    echo 请确认您的虚拟环境已经正确安装！
    pause
    exit /b 1
)

echo [INFO] 正在启动导出与部署脚本...
"%PYTHON_EXE%" "%PROJECT_ROOT%\tools\export_and_deploy_stamp_models.py"

echo.
echo ========================================================
echo                    操作已执行完毕！
echo ========================================================
pause
