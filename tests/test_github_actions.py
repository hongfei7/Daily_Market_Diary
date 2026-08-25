"""
GitHub Actions smoke test for the professional morning-briefing pipeline.
"""

import os
import sys

from _bootstrap import MARKET_DIARY, ROOT
from scripts.validate_workflow_shell import validate_workflow_shell


WORKFLOW_PATH = ROOT / ".github" / "workflows" / "morning_briefing_professional.yml"


def check_basic_imports() -> bool:
    print("=" * 60)
    print("Test 1: Third-party imports")
    print("=" * 60)

    required = ["pandas", "numpy", "matplotlib", "openai", "yfinance", "pypdf", "websockets"]
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
        "professional.catalyst_radar",
        "professional.config",
        "professional.daily_one_chart",
        "professional.dashboard",
        "professional.date_policy",
        "professional.email_builder",
        "professional.fact_checker",
        "professional.llm_enhancer",
        "professional.models",
        "professional.performance",
        "professional.report_blocks",
        "professional.report_formatting",
        "professional.report_quality",
        "professional.report_builder",
        "professional.report_layout",
        "professional.relevance",
        "professional.report_sections",
        "professional.report_text",
        "professional.runtime_audit",
        "professional.source_health",
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

    api_key = (os.getenv("DEEPSEEK_API_KEY") or os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()
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


def check_scheduled_archive_publish() -> bool:
    print("=" * 60)
    print("Test 4: Scheduled archive publishing")
    print("=" * 60)

    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")
    if "github.event_name == 'schedule'" not in workflow or "inputs.publish_archive" not in workflow:
        print("FAIL scheduled runs must publish the report archive")
        return False
    if "inputs.include_raw_bundle" not in workflow:
        print("FAIL raw bundle input must be safe for scheduled runs")
        return False
    if "--include-all-charts" in workflow:
        print("FAIL production archive must retain only report-referenced charts")
        return False
    print("OK  scheduled runs publish the archive")
    return True


def check_workflow_guardrails() -> bool:
    print("=" * 60)
    print("Test 5: Workflow guardrails")
    print("=" * 60)

    morning = WORKFLOW_PATH.read_text(encoding="utf-8")
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    combined = "\n".join([morning, ci])
    required_actions = (
        "actions/checkout@v6",
        "actions/setup-python@v6",
        "actions/upload-artifact@v6",
    )
    for action in required_actions:
        if action not in combined:
            print(f"FAIL missing modern GitHub Action: {action}")
            return False
    if "FORCE_JAVASCRIPT_ACTIONS_TO_NODE24" in combined:
        print("FAIL Node 24 forcing env should not be needed with v6 actions")
        return False
    if "python scripts/run_tests.py --pytest" not in morning:
        print("FAIL morning workflow must run the full pytest-backed suite")
        return False
    required_llm_config = (
        "LLM_PRIMARY_PROVIDER",
        "MINIMAX_API_KEY",
        "DEEPSEEK_API_KEY",
        "deepseek-v4-pro",
        "https://api.deepseek.com",
        "MiniMax-M3",
    )
    for marker in required_llm_config:
        if marker not in morning:
            print(f"FAIL missing LLM provider fallback config: {marker}")
            return False
    if "LLM_PRIMARY_PROVIDER: ${{ secrets.MINIMAX_API_KEY != '' && 'minimax' || 'deepseek' }}" not in morning:
        print("FAIL MiniMax-M3 must be the production primary provider")
        return False
    if "secrets.DEEPSEEK_API_KEY != '' && 'deepseek' || 'minimax'" in morning:
        print("FAIL workflow must not force DeepSeek ahead of MiniMax-M3")
        return False
    if "continuing with delivery" in morning or "set +e" in morning:
        print("FAIL runtime audit must block automatic delivery")
        return False
    if "git reset --hard" in morning:
        print("FAIL scheduled publishing must not discard generated ledgers or rewrite archive state")
        return False
    required_sla_guards = (
        'cron: "17 21 * * *"',
        'cron: "47 22 * * *"',
        "timeout-minutes: 35",
        "cancel-in-progress: false",
        "--require-email-preview",
        "--require-wecom-preview",
        "Primary WeCom decision brief failed",
        "Full report attachment was not delivered to WeCom",
        "--mode summary",
        '--attachment "reports_professional/${BRIEFING_DATE}_morning_briefing.html"',
        "Render validated A4 PDF companion",
        "render_report_pdf.py",
        "steps.pdf.outcome == 'success'",
        "PDF companion was not delivered",
        "status=success",
        "archive_needed",
        "primary run did not succeed",
        "--quality-policy commute",
        "Notify WeCom when the report is unavailable",
        "send_wecom_incident.py",
        "wecom_${BRIEFING_DATE}_summary_receipt.json",
        "wecom_${BRIEFING_DATE}_file_receipt.json",
        "wecom_${BRIEFING_DATE}_pdf_receipt.json",
        "steps.audit.outcome == 'success'",
        "Notify WeCom when the run gate fails",
        "RUNS_FILE=\"$(mktemp)\"",
        "DMD_REPORT_LINK_ENABLED",
    )
    for marker in required_sla_guards:
        if marker not in morning:
            print(f"FAIL workflow is missing SLA guard: {marker}")
            return False
    if "steps.gate.outputs.archive_needed == 'true' && steps.archive.outcome == 'success'" not in morning:
        print("FAIL a recovery run must not link a newly generated report to an older immutable archive")
        return False
    if "export RUNS_JSON" in morning or "os.environ.get(\"RUNS_JSON\"" in morning:
        print("FAIL recovery gate must not put GitHub API payloads in the process environment")
        return False
    if "reports_professional/performance/*" not in morning:
        print("FAIL signal performance artifacts must be retained with each workflow run")
        return False
    if "reports_professional/*.pdf" not in morning:
        print("FAIL validated PDF companions must be retained as workflow artifacts")
        return False
    deliver_block = morning.split("deliver:", 1)[1].split("run_full_tests:", 1)[0]
    if "default: true" not in deliver_block:
        print("FAIL manual workflow dispatch must deliver to WeCom by default")
        return False
    print("OK  workflows use current actions and full regression coverage")
    return True


def main() -> int:
    print("\n" + "=" * 60)
    print("GitHub Actions Smoke Test")
    print("=" * 60 + "\n")

    results = [
        ("Third-party imports", check_basic_imports()),
        ("Project imports", check_project_imports()),
        ("API environment", check_api_env()),
        ("Scheduled archive publishing", check_scheduled_archive_publish()),
        ("Workflow guardrails", check_workflow_guardrails()),
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


def test_scheduled_archive_publish() -> None:
    assert check_scheduled_archive_publish()


def test_workflow_guardrails() -> None:
    assert check_workflow_guardrails()


def test_workflow_run_blocks_are_valid_bash() -> None:
    assert validate_workflow_shell(WORKFLOW_PATH) == []


if __name__ == "__main__":
    raise SystemExit(main())
