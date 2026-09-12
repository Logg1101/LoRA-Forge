@echo off
title LoRA Forge
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found in "venv".
    echo Please ensure the venv folder exists at "%~dp0venv".
    pause
    exit /b 1
)

echo ========================================================
echo               Launching LoRA Forge v2.0
echo ========================================================
echo.

call venv\Scripts\activate.bat
python main.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Application closed with error code %ERRORLEVEL%.
    pause
)
