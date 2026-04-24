"""Daily Market Diary package.

Register historical ``modules`` and ``professional`` aliases for local scripts
or notebooks that may still import the old top-level module names.
"""

from __future__ import annotations

import importlib
import sys


_ALIASES = {
    "modules": "market_diary.modules",
    "professional": "market_diary.professional",
}

for alias, target in _ALIASES.items():
    if alias not in sys.modules:
        sys.modules[alias] = importlib.import_module(target)
