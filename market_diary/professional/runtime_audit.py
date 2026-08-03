"""Post-generation audit helpers for the professional morning briefing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


REQUIRED_REPORT_SECTION_GROUPS = [
    ("Visual dashboard", ("## Visual Dashboard",)),
    ("Global asset dashboard", ("### 1.2 Global Asset Price Dashboard",)),
    (
        "Hong Kong quick check",
        (
            "### 1.3 Hong Kong Key Data Quick Check",
            "### 1.3 Hong Kong Weekly Tape Quick Check",
            "### 1.3 Hong Kong Last Cash-Tape Quick Check (Reference)",
        ),
    ),
    ("Flow tracker", ("### 2.3 Flow Tracker and Attribution",)),
    ("Stock Connect active names", ("#### Stock Connect Southbound Active Names",)),
    ("A/H premium dispersion", ("#### AH Premium Dispersion",)),
    ("Daily one chart", ("### 3.3 Daily One Chart",)),
    ("Report quality", ("### Report Quality and Validation",)),
]

FORBIDDEN_PHRASES = [
    "Pending adapter",
    "not wired in this sprint",
    "waiting for a locked public historical source",
    "No stable historical public flow endpoint was selected in this sprint",
]

NON_ENGLISH_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
CLIPPED_CELL_RE = re.compile(r"(\.\.\.|…|\[trimmed\])\s*(?:\||$)", re.IGNORECASE)


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


def _audit_table_spacing(report_text: str) -> List[str]:
    errors: List[str] = []
    lines = report_text.splitlines()
    index = 0
    while index < len(lines):
        if not lines[index].strip().startswith("|"):
            index += 1
            continue
        start = index
        while index < len(lines) and lines[index].strip().startswith("|"):
            index += 1
        if start > 0 and lines[start - 1].strip():
            errors.append(f"Markdown table is not separated from previous block near line {start + 1}.")
        if index < len(lines) and lines[index].strip():
            errors.append(f"Markdown table is not separated from following block near line {index + 1}.")
    return errors


def _audit_english_only(report_text: str) -> List[str]:
    errors: List[str] = []
    for line_number, line in enumerate(report_text.splitlines(), start=1):
        match = NON_ENGLISH_SCRIPT_RE.search(line)
        if match:
            errors.append(f"Report contains non-English CJK/Kana/Hangul text near line {line_number}: `{line[:120]}`")
            if len(errors) >= 5:
                break
    return errors


def _audit_clipped_text(report_text: str) -> List[str]:
    errors: List[str] = []
    for line_number, line in enumerate(report_text.splitlines(), start=1):
        if CLIPPED_CELL_RE.search(line.rstrip()):
            errors.append(f"Report appears to contain clipped text near line {line_number}: `{line[:120]}`")
            if len(errors) >= 5:
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

    for label, markers in REQUIRED_REPORT_SECTION_GROUPS:
        if not any(marker in report_text for marker in markers):
            errors.append(f"Missing required report section group: {label} ({' OR '.join(markers)})")

    for phrase in FORBIDDEN_PHRASES:
        if phrase.lower() in report_text.lower():
            errors.append(f"Report still contains forbidden placeholder phrase: {phrase}")

    errors.extend(_audit_tables(report_text))
    errors.extend(_audit_table_spacing(report_text))
    errors.extend(_audit_english_only(report_text))
    errors.extend(_audit_clipped_text(report_text))

    image_refs = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", report_text)
    if len(image_refs) >= 2 and image_refs[0] == image_refs[-1]:
        errors.append("Visual Dashboard and Daily One Chart appear to reference the same image path.")

    if bundle:
        if not bundle.get("report_quality"):
            errors.append("Bundle is missing report_quality diagnostics.")
        if not bundle.get("fact_check"):
            errors.append("Bundle is missing fact_check diagnostics.")
        if not bundle.get("provenance_audit"):
            errors.append("Bundle is missing provenance_audit diagnostics.")
        if not bundle.get("source_health"):
            errors.append("Bundle is missing source_health diagnostics.")
        if not bundle.get("performance"):
            errors.append("Bundle is missing performance diagnostics.")

        meta = bundle.get("meta", {}) or {}
        if str(meta.get("briefing_date", "")) != report_date:
            warnings.append(f"Bundle briefing_date `{meta.get('briefing_date')}` does not match requested report date `{report_date}`.")

        report_quality = bundle.get("report_quality", {}) or {}
        if report_quality.get("warnings"):
            warnings.extend(str(item) for item in (report_quality.get("warnings", []) or [])[:8])
        release_recommendation = report_quality.get("release_recommendation", {}) or {}
        if release_recommendation.get("action") == "manual_review":
            errors.append("Report quality requires manual review; automatic distribution is blocked.")

        fact_check = bundle.get("fact_check", {}) or {}
        if fact_check.get("status") in {"warning", "error"}:
            errors.append("Fact-check guardrail has unresolved warnings or errors.")

        provenance_audit = bundle.get("provenance_audit", {}) or {}
        if provenance_audit.get("status") != "ok":
            details = "; ".join(str(item) for item in (provenance_audit.get("errors", []) or [])[:4])
            errors.append(f"Source provenance validation failed{': ' + details if details else '.'}")

        source_health = bundle.get("source_health", {}) or {}
        if source_health.get("status") == "failed":
            failures = ", ".join(str(item) for item in (source_health.get("critical_failures", []) or []))
            errors.append(f"Critical source-health policy failed{': ' + failures if failures else '.'}")

        performance = bundle.get("performance", {}) or {}
        if (performance.get("methodology", {}) or {}).get("look_ahead_guard") is not True:
            errors.append("Performance diagnostics are missing the look-ahead guard.")
        if performance.get("status") == "error":
            warnings.append(f"Performance tracking failed: {performance.get('error', 'unknown error')}")

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
