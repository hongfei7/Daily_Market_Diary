"""Post-generation audit helpers for the professional morning briefing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REQUIRED_REPORT_SECTIONS = [
    "## Visual Dashboard",
    "### 1.2 Global Asset Price Dashboard",
    "### 1.3 Hong Kong Key Data Quick Check",
    "### 2.3 Flow Tracker and Attribution",
    "#### Stock Connect Southbound Active Names",
    "#### AH Premium Dispersion",
    "### 3.3 Daily One Chart",
    "### Report Quality and Validation",
]

FORBIDDEN_PHRASES = [
    "Pending adapter",
    "not wired in this sprint",
    "waiting for a locked public historical source",
    "No stable historical public flow endpoint was selected in this sprint",
]


def _count_unescaped_pipes(text: str) -> int:
    count = 0
    escaped = False
    for char in text:
        if escaped:
            escaped = False
            continue
        if char == "\\":
            escaped = True
            continue
        if char == "|":
            count += 1
    return count


def _audit_tables(report_text: str) -> List[str]:
    errors: List[str] = []
    lines = report_text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line.startswith("|"):
            index += 1
            continue
        block: List[str] = []
        while index < len(lines) and lines[index].strip().startswith("|"):
            block.append(lines[index].rstrip())
            index += 1
        if len(block) < 2:
            continue
        expected = _count_unescaped_pipes(block[0])
        for row_idx, row in enumerate(block[1:], start=2):
            if _count_unescaped_pipes(row) != expected:
                errors.append(f"Malformed markdown table near row {row_idx} of a table block: `{row[:120]}`")
                break
    return errors


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, dict) else {}


def audit_generated_run(
    output_dir: str | Path,
    report_date: str,
    require_llm: bool = False,
    require_email_preview: bool = False,
) -> Dict[str, Any]:
    root = Path(output_dir)
    report_path = root / f"{report_date}_morning_briefing.md"
    bundle_path = root / "raw" / f"{report_date}_bundle.json"
    dashboard_path = root / "charts" / f"dashboard_{report_date}.png"
    daily_chart_path = root / "charts" / f"daily_one_chart_{report_date}.png"
    email_preview_path = root / f"{report_date}_email_preview.html"

    errors: List[str] = []
    warnings: List[str] = []

    for label, path in [
        ("report", report_path),
        ("bundle", bundle_path),
        ("dashboard", dashboard_path),
        ("daily chart", daily_chart_path),
    ]:
        if not path.exists():
            errors.append(f"Missing required {label} file: {path}")
        elif path.is_file() and path.stat().st_size == 0:
            errors.append(f"Required {label} file is empty: {path}")

    if require_email_preview and not email_preview_path.exists():
        errors.append(f"Missing required email preview file: {email_preview_path}")
    elif email_preview_path.exists() and email_preview_path.stat().st_size == 0:
        errors.append(f"Email preview file is empty: {email_preview_path}")

    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    bundle = _load_json(bundle_path) if bundle_path.exists() else {}

    for marker in REQUIRED_REPORT_SECTIONS:
        if marker not in report_text:
            errors.append(f"Missing required report section: {marker}")

    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in report_text.lower():
            errors.append(f"Report still contains forbidden placeholder phrase: {phrase}")

    errors.extend(_audit_tables(report_text))

    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report_text)
    if len(image_refs) >= 2 and image_refs[0] == image_refs[-1]:
        errors.append("Visual Dashboard and Daily One Chart appear to reference the same image path.")

    if bundle:
        if not bundle.get("report_quality"):
            errors.append("Bundle is missing report_quality diagnostics.")
        if not bundle.get("fact_check"):
            errors.append("Bundle is missing fact_check diagnostics.")

        meta = bundle.get("meta", {}) or {}
        if str(meta.get("briefing_date", "")) != report_date:
            warnings.append(f"Bundle briefing_date `{meta.get('briefing_date')}` does not match requested report date `{report_date}`.")

        report_quality = bundle.get("report_quality", {}) or {}
        if report_quality.get("warnings"):
            warnings.extend(str(item) for item in (report_quality.get("warnings", []) or [])[:8])

        fact_check = bundle.get("fact_check", {}) or {}
        if fact_check.get("status") == "warning":
            warnings.append("Fact-check guardrail reported warnings.")

        llm_sections = bundle.get("llm_sections", {}) or {}
        task_meta = (llm_sections.get("task_meta", {}) or {}).get("tasks", {}) or {}
        if require_llm:
            if not task_meta:
                errors.append("LLM task metadata is missing for a run that required LLM validation.")
            else:
                errored = [name for name, item in task_meta.items() if isinstance(item, dict) and item.get("status") == "error"]
                if errored:
                    errors.append(f"LLM tasks returned error status: {', '.join(errored)}")
        elif not task_meta:
            warnings.append("LLM overlay was not run; runtime audit only covers deterministic layers.")

    if email_preview_path.exists():
        html = email_preview_path.read_text(encoding="utf-8")
        for marker in ("Report quality", "Hong Kong local checks", "Deep-read setup"):
            if marker not in html:
                warnings.append(f"Email preview is missing expected marker: {marker}")

    status = "ok" if not errors else "error"
    return {
        "status": status,
        "report_date": report_date,
        "errors": errors,
        "warnings": warnings,
        "checked_files": {
            "report": str(report_path),
            "bundle": str(bundle_path),
            "dashboard": str(dashboard_path),
            "daily_chart": str(daily_chart_path),
            "email_preview": str(email_preview_path),
        },
    }


def format_audit_summary(audit: Dict[str, Any]) -> str:
    lines = [f"Audit status: {audit.get('status', 'unknown')}"]
    errors = audit.get("errors", []) or []
    warnings = audit.get("warnings", []) or []
    lines.append(f"Errors: {len(errors)}")
    lines.extend(f"- ERROR: {item}" for item in errors)
    lines.append(f"Warnings: {len(warnings)}")
    lines.extend(f"- WARN: {item}" for item in warnings)
    return "\n".join(lines)
