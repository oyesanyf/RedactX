@echo off
setlocal

:: RedactX Training Launcher for Python 3.12 (Bypasses Windows IFEO and Free-Threaded 3.14)
set "PY_DIR=C:\Users\oyesanyf\AppData\Local\Programs\Python\Python312"
set "PATH=%PY_DIR%;%PY_DIR%\Scripts;%PATH%"

cd /d "D:\harfile\RedactX"

echo =======================================================
echo  Launching RedactX VaultGemma Training (Python 3.12)
echo =======================================================

"%PY_DIR%\py312.exe" train.py ^
    --model google/vaultgemma-1b ^
    --force-vaultgemma ^
    --device cuda ^
    --recipe-60-20-20 ^
    --samples 2000 ^
    --epochs 5 ^
    --batch-size 16 ^
    --lr 3e-4 ^
    --lambda-consistency 1.0 ^
    --output-dir ./models/RedactX ^
    --model-name RedactX %*

endlocal
