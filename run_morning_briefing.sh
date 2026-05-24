#!/bin/bash
# Daily Morning Briefing launcher for Linux and macOS.

set -euo pipefail

echo "========================================"
echo "  Daily Morning Briefing Launcher"
echo "  Hong Kong Research Desk Edition"
echo "========================================"
echo ""

if ! command -v python3 >/dev/null 2>&1; then
    echo "[ERROR] Python 3.10 or above was not found on this machine."
    exit 1
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 10) else 1)
PY
then
    echo "[ERROR] Python 3.10 or above is required."
    exit 1
fi

cd "$(dirname "$0")"

echo "[1/3] Checking dependencies..."
if ! python3 -c "import pandas" >/dev/null 2>&1; then
    echo "[INFO] Installing project requirements..."
    pip3 install -r market_diary/requirements.txt
fi

echo ""
echo "[2/3] Generating the professional morning briefing..."
python3 market_diary/main_professional.py "$@"

echo ""
echo "[3/3] Output directory:"
echo "$(pwd)/reports_professional"

if [[ "$OSTYPE" == "darwin"* ]]; then
    open reports_professional
fi

echo ""
echo "========================================"
echo "  Morning briefing completed successfully."
echo "========================================"
