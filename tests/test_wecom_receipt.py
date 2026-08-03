from __future__ import annotations

import json
import tempfile
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from scripts.send_wecom_incident import build_incident_markdown
from scripts import send_report_wecom as wecom


def test_success_receipt_is_machine_verifiable() -> None:
    bundle = {
        "meta": {"briefing_date": "2026-08-03", "global_market_date": "2026-08-02", "hk_data_date": "2026-07-31"},
        "overview": {"risk_regime": "Mixed", "theme": "Conditional setup."},
        "llm_sections": {"one_line_market_pulse": "Conditional setup.", "risk_check": "Reassess on weaker breadth."},
        "report_quality": {"score": 82, "grade": "B", "release_recommendation": {"label": "Send with caveats"}},
        "today_forward": {"focus_lines": ["Confirm with Hong Kong breadth."]},
        "market_summary": {},
        "hk_quick_checks": [],
        "must_watch": [],
    }
    original = wecom._wecom_post
    wecom._wecom_post = lambda webhook_url, payload: {"errcode": 0, "errmsg": "ok"}
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            receipt = Path(tmpdir) / "summary.json"
            wecom.send_summary("https://example.invalid/send?key=test", bundle, "2026-08-03", receipt_path=receipt)
            payload = json.loads(receipt.read_text(encoding="utf-8"))
    finally:
        wecom._wecom_post = original

    assert payload["status"] == "ok"
    assert payload["channel"] == "wecom"
    assert payload["kind"] == "summary"
    assert payload["report_date"] == "2026-08-03"
    assert payload["response"] == {"errcode": 0, "errmsg": "ok"}
    assert "sent_at_utc" in payload


def test_incident_message_preserves_recovery_context() -> None:
    run_url = "https://github.com/example/repo/actions/runs/123"
    markdown = build_incident_markdown(
        "2026-08-03",
        "generation=success, preview=success, audit=failure",
        "05:42",
        run_url,
    )
    assert "2026-08-03" in markdown
    assert "audit=failure" in markdown
    assert "06:47 recovery run" in markdown
    assert run_url in markdown


def main() -> None:
    test_success_receipt_is_machine_verifiable()
    test_incident_message_preserves_recovery_context()
    print("WeCom receipt test passed")


if __name__ == "__main__":
    main()
