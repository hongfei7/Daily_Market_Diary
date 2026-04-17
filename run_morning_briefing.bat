@echo off
setlocal

echo ========================================
echo   Daily Morning Briefing Launcher
echo   Hong Kong Research Desk Edition
echo ========================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python 3.8 or above was not found on this machine.
    pause
    exit /b 1
)

cd /d "%~dp0"

echo [1/3] Checking dependencies...
pip show pandas >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing project requirements...
    pip install -r market_diary/requirements.txt
)

echo.
echo [2/3] Generating the professional morning briefing...
python market_diary/main_professional.py %*

if errorlevel 1 (
    echo.
    echo [ERROR] Morning briefing generation failed.
    pause
    exit /b 1
)

echo.
echo [3/3] Opening the output folder...
start reports_professional

echo.
echo ========================================
echo   Morning briefing completed successfully.
echo ========================================
pause
