"""
GitHub Actions smoke test for the professional morning-briefing pipeline.
"""

import os
import sys

from _bootstrap import MARKET_DIARY


def check_basic_imports() -> bool:
    print("=" * 60)
    print("Test 1: Third-party imports")
    print("=" * 60)

    required = ["pandas", "numpy", "matplotlib", "openai", "yfinance"]
    for module_name in required:
        try:
            __import__(module_name)
            print(f"OK  {module_name}")
        except Exception as exc:
            print(f"FAIL {module_name}: {exc}")
            return False
    return True


def check_project_imports() -> bool:
    print("=" * 60)
    print("Test 2: Project imports")
    print("=" * 60)

    sys.path.insert(0, str(MARKET_DIARY))
    modules = [
        "main_professional",
        "market_diary.main_professional",
        "market_diary.modules.data_fetcher",
        "market_diary.professional.analytics",
        "modules.data_fetcher",
        "modules.chart_features",
        "modules.china_rates",
        "modules.adapter_ah_premium",
        "modules.adapter_hkex_announce",
        "modules.adapter_shortsell",
        "modules.adapter_stockconnect",
        "modules.llm_client",
        "modules.macro_calendar",
        "modules.market_movers",
        "modules.hk_local_data",
        "modules.local_metrics",
        "modules.risk_radar",
        "modules.sector_news",
        "modules.text_normalizer",
        "professional.analytics",
        "professional.analytics_briefing",
        "professional.analytics_flows",
        "professional.analytics_hk_checks",
        "professional.analytics_macro",
        "professional.analytics_narrative",
        "professional.analytics_market",
        "professional.analytics_public_flow",
        "professional.analytics_sector",
        "professional.analytics_trackers",
        "professional.analytics_watchlist",
        "professional.attribution",
        "professional.config",
        "professional.daily_one_chart",
        "professional.dashboard",
        "professional.date_policy",
        "professional.email_builder",
        "professional.fact_checker",
        "professional.llm_enhancer",
        "professional.models",
        "professional.report_blocks",
        "professional.report_formatting",
        "professional.report_quality",
        "professional.report_builder",
        "professional.report_layout",
        "professional.relevance",
        "professional.report_sections",
        "professional.report_text",
        "professional.runtime_audit",
        "professional.trend_pack",
    ]

    for module_name in modules:
        try:
            __import__(module_name)
            print(f"OK  {module_name}")
        except Exception as exc:
            print(f"FAIL {module_name}: {exc}")
            return False
    return True


def check_api_env() -> bool:
    print("=" * 60)
    print("Test 3: API environment")
    print("=" * 60)

    from modules.llm_client import api_key_available

    api_key = (os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
    if not api_key:
        if api_key_available():
            print("Local API key file detected; skipping remote client validation.")
            return True
        print("No API key configured; skipping client validation.")
        return True

    print(f"API key detected (length={len(api_key)})")
    base_url = os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or ""
    if base_url:
        print(f"Base URL: {base_url}")
    if os.getenv("LLM_MODEL"):
        print(f"Model: {os.getenv('LLM_MODEL')}")
    return True


def main() -> int:
    print("\n" + "=" * 60)
    print("GitHub Actions Smoke Test")
    print("=" * 60 + "\n")

    results = [
        ("Third-party imports", check_basic_imports()),
        ("Project imports", check_project_imports()),
        ("API environment", check_api_env()),
    ]

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, passed in results:
        print(f"{name:24s} {'PASS' if passed else 'FAIL'}")

    failed = [name for name, passed in results if not passed]
    if failed:
        print(f"\nSmoke test failed: {', '.join(failed)}")
        return 1

    print("\nSmoke test passed")
    return 0


def test_basic_imports() -> None:
    assert check_basic_imports()


def test_project_imports() -> None:
    assert check_project_imports()


def test_api_env() -> None:
    assert check_api_env()


if __name__ == "__main__":
    raise SystemExit(main())
