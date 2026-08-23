@echo off
REM ============================================================================
REM  Self-play + Training Loop Script
REM  Loop executes:
REM    1. Start self-play sampling (mcts_sample_buffer.py)
REM    2. Train model (mcts_train_cnn_v11.py)
REM  Terminates if any step fails
REM ============================================================================
setlocal enabledelayedexpansion

cd /d "%~dp0"

echo ========================================
echo   Self-play + Training Loop
echo   Start time: %date% %time%
echo ========================================
echo.
echo [Config] Games per round: --games 30 --workers 10
echo [Config] Checkpoint interval: --updates-per-checkpoint 15
echo [Config] Model auto-detect: cnn_v11 highest generation
echo [Config] Stop condition: stops if any step fails
echo.

set ROUND=0

:loop
set /a ROUND+=1
echo.
echo ========================================
echo   Round !ROUND!
echo ========================================
echo.

REM ---- Step 1: Start self-play sampling ----
echo [INFO] Starting self-play sampling...
python src/ai/scripts/mcts_sample_buffer.py --games 10 --workers 10 --temp-steps 16
if errorlevel 1 (
    echo.
    echo [ERROR] Round !ROUND!: self-play failed, stopping loop.
    goto :end
)
echo [ OK ] Round !ROUND!: self-play completed

echo.

REM ---- Step 2: Train model ----
echo [INFO] Starting model training...
python src/ai/scripts/mcts_train_cnn_v11.py --updates-per-checkpoint 15 --eval-games 10
if errorlevel 1 (
    echo.
    echo [ERROR] Round !ROUND!: training failed, stopping loop.
    goto :end
)
echo [ OK ] Round !ROUND!: training completed

echo.
echo [INFO] Round !ROUND! completed, preparing next round...
goto :loop

:end
echo.
echo ========================================
echo   Loop ended
echo   End time: %date% %time%
echo   Completed: !ROUND! rounds
echo ========================================
endlocal
