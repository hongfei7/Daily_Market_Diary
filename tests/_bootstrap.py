from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MARKET_DIARY = ROOT / "market_diary"

for path in (ROOT, MARKET_DIARY):
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
