@echo off
REM ============================================================================
REM  自对弈 + 训练循环脚本
REM  循环执行：
REM    1. 启动自对弈采样 (mcts_sample_buffer.py)
REM    2. 训练模型 (mcts_train_cnn_v11.py)
REM  任一环节失败则终止
REM ============================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ========================================
echo   自对弈 + 训练循环
echo   开始时间: %date% %time%
echo ========================================
echo.
echo [配置] 每轮自对弈局数: --games 30 --workers 10
echo [配置] checkpoint间隔: --updates-per-checkpoint 15
echo [配置] model自动识别: cnn_v11 最高代数
echo [配置] 终止条件: 任一环节失败则终止循环
echo.

set ROUND=0

:loop
set /a ROUND+=1
echo.
echo ========================================
echo   第 !ROUND! 轮
echo ========================================
echo.

REM ---- 步骤1: 启动自对弈采样 ----
echo [INFO] 开始自对弈采样...
python src/ai/scripts/mcts_sample_buffer.py --games 30 --workers 10
if errorlevel 1 (
    echo.
    echo [ERROR] 第 !ROUND! 轮: 自对弈失败，终止循环。
    goto :end
)
echo [ OK ] 第 !ROUND! 轮: 自对弈完成

echo.

REM ---- 步骤2: 训练模型 ----
echo [INFO] 开始训练模型...
python src/ai/scripts/mcts_train_cnn_v11.py --updates-per-checkpoint 15
if errorlevel 1 (
    echo.
    echo [ERROR] 第 !ROUND! 轮: 训练失败，终止循环。
    goto :end
)
echo [ OK ] 第 !ROUND! 轮: 训练完成

echo.
echo [INFO] 第 !ROUND! 轮完成，准备开始下一轮...
goto :loop

:end
echo.
echo ========================================
echo   循环结束
echo   结束时间: %date% %time%
echo   已完成: !ROUND! 轮
echo ========================================
endlocal
