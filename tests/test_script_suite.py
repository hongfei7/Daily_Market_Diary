from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


TEST_DIR = Path(__file__).resolve().parent


SCRIPT_STYLE_TESTS = [
    "test_chart_appendix.py",
    "test_daily_one_chart.py",
    "test_date_semantics.py",
    "test_email_delivery.py",
    "test_hk_local_foundation.py",
    "test_llm_pipeline.py",
    "test_news_cache.py",
    "test_professional_workbench.py",
    "test_report_quality.py",
    "test_runtime_audit.py",
    "test_stockconnect_ah_premium.py",
    "test_text_normalizer.py",
    "test_trend_pack.py",
]


def _load_script_module(filename: str) -> ModuleType:
    path = TEST_DIR / filename
    module_name = f"_script_suite_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Unable to load script-style test module: {filename}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_style_smoke_suite() -> None:
    for filename in SCRIPT_STYLE_TESTS:
        module = _load_script_module(filename)
        main = getattr(module, "main", None)
        assert callable(main), f"{filename} does not expose a callable main()"
        main()
