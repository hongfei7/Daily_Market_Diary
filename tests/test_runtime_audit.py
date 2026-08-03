import json
import os
import shutil
import sys
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from professional.runtime_audit import audit_generated_run


REPORT_BODY = """# Morning Research Workbench | 2026-04-14

## Visual Dashboard
![Research Dashboard](charts/dashboard_2026-04-14.png)

### 1.2 Global Asset Price Dashboard

| Asset | Last | Read |
| --- | --- | --- |
| China 10Y | 1.79% | China local rates anchor \\| Live public |

### 1.3 Hong Kong Key Data Quick Check

| Check | Value | Status |
| --- | --- | --- |
| Southbound / Northbound net flow | Southbound +HK$2.2bn \\| Northbound N/A | Live public |

### 2.3 Flow Tracker and Attribution
#### Stock Connect Southbound Active Names
#### AH Premium Dispersion

### 3.3 Daily One Chart
![Daily One Chart](charts/daily_one_chart_2026-04-14.png)

### Report Quality and Validation
- **Quality score:** 90.0/100
**Narrative fact-check guardrail**
**Adapter status**
"""


def _bundle():
    return {
        "meta": {"briefing_date": "2026-04-14"},
        "report_quality": {"score": 90.0, "warnings": []},
        "fact_check": {"status": "ok", "summary": "Checked 3 numeric claims; 0 numeric mismatch(es), 0 logic warning(s)."},
        "provenance_audit": {"status": "ok", "checked_records": 3, "unavailable_records": 0, "errors": [], "warnings": []},
        "source_health": {"status": "healthy", "critical_failures": []},
        "performance": {"status": "insufficient_history", "methodology": {"look_ahead_guard": True}},
        "llm_sections": {"task_meta": {"tasks": {"overnight_review": {"status": "ok"}}}},
    }


def main() -> None:
    root = Path(os.getcwd()) / "reports_professional" / "_runtime_audit_tmp"
    try:
        shutil.rmtree(root, ignore_errors=True)
        (root / "charts").mkdir(parents=True)
        (root / "raw").mkdir(parents=True)
        (root / "2026-04-14_morning_briefing.md").write_text(REPORT_BODY, encoding="utf-8")
        (root / "raw" / "2026-04-14_bundle.json").write_text(json.dumps(_bundle()), encoding="utf-8")
        (root / "charts" / "dashboard_2026-04-14.png").write_bytes(b"png")
        (root / "charts" / "daily_one_chart_2026-04-14.png").write_bytes(b"png")
        (root / "2026-04-14_email_preview.html").write_text(
            "<html><body>Report quality Deep-read setup Hong Kong local checks</body></html>", encoding="utf-8"
        )

        audit = audit_generated_run(root, "2026-04-14", require_llm=True, require_email_preview=True)
        assert audit["status"] == "ok"
        assert not audit["errors"]

        broken_report = REPORT_BODY.replace("\\|", "|")
        (root / "2026-04-14_morning_briefing.md").write_text(broken_report, encoding="utf-8")
        broken_audit = audit_generated_run(root, "2026-04-14", require_llm=True, require_email_preview=True)
        assert broken_audit["status"] == "error"
        assert any("Malformed markdown table" in item for item in broken_audit["errors"])

        (root / "2026-04-14_morning_briefing.md").write_text(REPORT_BODY + "\nThis line was clipped...\n", encoding="utf-8")
        clipped_audit = audit_generated_run(root, "2026-04-14", require_llm=True, require_email_preview=True)
        assert clipped_audit["status"] == "error"
        assert any("clipped text" in item for item in clipped_audit["errors"])

        (root / "2026-04-14_morning_briefing.md").write_text(
            REPORT_BODY + "\nA headline may contain an ellipsis... without being clipped.\n",
            encoding="utf-8",
        )
        natural_ellipsis_audit = audit_generated_run(root, "2026-04-14", require_llm=True, require_email_preview=True)
        assert natural_ellipsis_audit["status"] == "ok"

        (root / "2026-04-14_morning_briefing.md").write_text(REPORT_BODY + "\nThis line was clipped [trimmed]\n", encoding="utf-8")
        trimmed_audit = audit_generated_run(root, "2026-04-14", require_llm=True, require_email_preview=True)
        assert trimmed_audit["status"] == "error"
        assert any("clipped text" in item for item in trimmed_audit["errors"])

        (root / "2026-04-14_morning_briefing.md").write_text(REPORT_BODY + "\nUnexpected non-English token: 찜흙\n", encoding="utf-8")
        language_audit = audit_generated_run(root, "2026-04-14", require_llm=True, require_email_preview=True)
        assert language_audit["status"] == "error"
        assert any("non-English" in item for item in language_audit["errors"])

        bundle = _bundle()
        bundle["provenance_audit"] = {"status": "error", "errors": ["market_data: missing provenance records"]}
        (root / "raw" / "2026-04-14_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")
        (root / "2026-04-14_morning_briefing.md").write_text(REPORT_BODY, encoding="utf-8")
        provenance_audit = audit_generated_run(root, "2026-04-14", require_llm=True, require_email_preview=True)
        assert provenance_audit["status"] == "error"
        assert any("provenance" in item.lower() for item in provenance_audit["errors"])
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("Runtime audit test passed")


if __name__ == "__main__":
    main()
