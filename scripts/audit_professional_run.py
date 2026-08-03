from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "market_diary"))

from professional.runtime_audit import audit_generated_run, format_audit_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a generated professional morning briefing run.")
    parser.add_argument("--report-date", required=True, help="Briefing date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default="reports_professional", help="Directory containing generated outputs.")
    parser.add_argument("--require-llm", action="store_true", help="Fail the audit if LLM task metadata is missing or errored.")
    parser.add_argument("--require-email-preview", action="store_true", help="Fail the audit if the email preview HTML file is missing.")
    parser.add_argument("--require-wecom-preview", action="store_true", help="Fail if the primary WeCom summary or HTML preview is missing or invalid.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    audit = audit_generated_run(
        output_dir=(ROOT / args.output_dir).resolve(),
        report_date=args.report_date,
        require_llm=args.require_llm,
        require_email_preview=args.require_email_preview,
        require_wecom_preview=args.require_wecom_preview,
    )
    print(format_audit_summary(audit))
    return 0 if audit.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
