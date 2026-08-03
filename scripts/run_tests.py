from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

SYSTEM_TESTS = [
    "tests/test_github_actions.py",
    "tests/test_analytics_briefing.py",
    "tests/test_analytics_flows.py",
    "tests/test_analytics_hk_checks.py",
    "tests/test_analytics_macro.py",
    "tests/test_analytics_narrative.py",
    "tests/test_analytics_public_flow.py",
    "tests/test_analytics_sector.py",
    "tests/test_analytics_trackers.py",
    "tests/test_analytics_watchlist.py",
    "tests/test_stockconnect_ah_premium.py",
    "tests/test_report_quality.py",
    "tests/test_runtime_audit.py",
    "tests/test_stage_report_archive.py",
    "tests/test_hk_local_foundation.py",
    "tests/test_hk_flow_attribution.py",
    "tests/test_daily_one_chart.py",
    "tests/test_market_data_resilience.py",
    "tests/test_llm_enhancer_resilience.py",
    "tests/test_llm_pipeline.py",
    "tests/test_skill_shadow.py",
    "tests/test_email_delivery.py",
    "tests/test_report_html.py",
    "tests/test_editorial_guards.py",
    "tests/test_fact_checker.py",
    "tests/test_provenance.py",
    "tests/test_source_health.py",
    "tests/test_performance.py",
    "tests/test_text_normalizer.py",
    "tests/test_chart_appendix.py",
    "tests/test_news_cache.py",
    "tests/test_date_semantics.py",
    "tests/test_trend_pack.py",
    "tests/test_report_gallery.py",
    "tests/test_repo_hygiene.py",
    "tests/test_professional_workbench.py",
]


def _run(command: list[str]) -> None:
    print(f"$ {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Daily Market Diary regression suite.")
    parser.add_argument(
        "--pytest",
        action="store_true",
        help="Run pytest collection after the script-based CI smoke suite.",
    )
    args = parser.parse_args(argv)

    for test_path in SYSTEM_TESTS:
        _run([sys.executable, test_path])

    if args.pytest:
        _run([sys.executable, "-m", "pytest", "-q"])

    print("All regression checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
