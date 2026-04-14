#!/bin/bash
# 投行研究院晨报生成脚本 - Linux/Mac 版本
# 使用方法：chmod +x run_morning_briefing.sh && ./run_morning_briefing.sh

set -e

echo "========================================"
echo "  投行研究院晨报生成器"
echo "  Morning Briefing Generator"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "[错误] 未检测到 Python，请先安装 Python 3.8+"
    exit 1
fi

# 检查环境变量
if [ -z "$MINIMAX_API_KEY" ]; then
    echo "[警告] 未设置 MINIMAX_API_KEY 环境变量"
    echo "请设置后重试：export MINIMAX_API_KEY=your_key_here"
    exit 1
fi

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 检查依赖
echo "[1/3] 检查依赖..."
if ! python3 -c "import pandas" &> /dev/null; then
    echo "[提示] 正在安装依赖..."
    pip3 install -r market_diary/requirements.txt
fi

# 生成晨报
echo ""
echo "[2/3] 生成晨报..."
python3 market_diary/main_professional.py "$@"

# 打开报告目录
echo ""
echo "[3/3] 报告已生成"
echo "报告目录: $(pwd)/reports_professional"

# 在 Mac 上自动打开
if [[ "$OSTYPE" == "darwin"* ]]; then
    open reports_professional
fi

echo ""
echo "========================================"
echo "  晨报生成完成！"
echo "========================================"
