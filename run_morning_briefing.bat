@echo off
REM 投行研究院晨报生成脚本 - Windows 版本
REM 使用方法：双击运行或在命令行执行

echo ========================================
echo   投行研究院晨报生成器
echo   Morning Briefing Generator
echo ========================================
echo.

REM 检查 Python 是否安装
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到 Python，请先安装 Python 3.8+
    pause
    exit /b 1
)

REM 检查环境变量
if "%MINIMAX_API_KEY%"=="" (
    echo [警告] 未设置 MINIMAX_API_KEY 环境变量
    echo 请设置后重试：set MINIMAX_API_KEY=your_key_here
    pause
    exit /b 1
)

REM 切换到脚本所在目录
cd /d "%~dp0"

REM 检查依赖
echo [1/3] 检查依赖...
pip show pandas >nul 2>&1
if errorlevel 1 (
    echo [提示] 正在安装依赖...
    pip install -r market_diary/requirements.txt
)

REM 生成晨报
echo.
echo [2/3] 生成晨报...
python market_diary/main_professional.py %*

REM 检查结果
if errorlevel 1 (
    echo.
    echo [错误] 晨报生成失败
    pause
    exit /b 1
)

REM 打开报告目录
echo.
echo [3/3] 打开报告目录...
start reports_professional

echo.
echo ========================================
echo   晨报生成完成！
echo ========================================
pause
