"""Post-generation audit helpers for the professional morning briefing."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# Section titles, never section numbers. Numbers are positional: inserting a
# section renumbers everything after it, and a contract written against
# "### 2.3 Flow Tracker and Attribution" then reports the section as missing
# even though it is present as 2.4. Titles are stable, so the contract is
# written against those and the number is matched loosely.
REQUIRED_REPORT_SECTION_GROUPS = [
    ("Visual dashboard", ("## Visual Dashboard",)),
    ("Catalyst event radar", ("![Catalyst & Event Radar]",)),
    ("Global asset dashboard", ("Global Asset Price Dashboard",)),
    (
        "Hong Kong quick check",
        (
            # Legitimate variants that switch with the trading-day mode.
            "Hong Kong Key Data Quick Check",
            "Hong Kong Weekly Tape Quick Check",
            "Hong Kong Last Cash-Tape Quick Check (Reference)",
        ),
    ),
    ("Flow tracker", ("Flow Tracker and Attribution",)),
    ("Stock Connect active names", ("**Stock Connect Southbound Active Names**",)),
    ("A/H premium dispersion", ("**AH Premium Dispersion**",)),
    ("Daily one chart", ("Daily One Chart",)),
    ("Report quality", ("Report Quality and Validation",)),
]

# Headings become entries in the generated table of contents
# (_structure_report_html) and are hidden by the print stylesheet, so more of
# them improves navigation on the phone without costing anything on paper. The
# cap only guards against genuine fragmentation.
MAX_REPORT_HEADINGS = 34


def _section_present(report_text: str, marker: str) -> bool:
    """Whether a required section marker appears in the report.

    Markers carrying their own markup (``## Visual Dashboard``, ``**Stock
    Connect...**``, ``![Catalyst...]``) are matched literally. A bare title is
    matched as a heading with any numbering, so inserting a section and shifting
    ``2.3`` to ``2.4`` cannot report a present section as missing — while a
    passing mention of the title in prose still does not satisfy the contract.
    """
    if marker.startswith(("#", "*", "!", "[")):
        return marker in report_text
    pattern = rf"^#{{2,4}}\s+(?:[\d.]+\s+)?{re.escape(marker)}\s*$"
    return re.search(pattern, report_text, re.MULTILINE) is not None


FORBIDDEN_PHRASES = [
    "Pending adapter",
    "not wired in this sprint",
    "waiting for a locked public historical source",
    "No stable historical public flow endpoint was selected in this sprint",
]

NON_ENGLISH_SCRIPT_RE = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]+")
CLIPPED_CELL_RE = re.compile(r"(\.\.\.|…|\[trimmed\])\s*(?:\||$)", re.IGNORECASE)
# Sized for a 20-30 minute commute read at roughly 200 wpm for dense analytical
# prose. The previous 3000-4500 band was set when nearly a third of the report
# was pipeline self-assessment; with that moved to audit/*.json the same budget
# should now buy market content.
REPORT_TARGET_WORDS = (4200, 6000)
REPORT_HARD_MAX_WORDS = 7000
WECOM_SAFE_MARKDOWN_BYTE_LIMIT = 3800


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
    require_wecom_preview: bool = False,
    quality_policy: str = "strict",
) -> Dict[str, Any]:
    if quality_policy not in {"strict", "commute"}:
        raise ValueError("quality_policy must be either 'strict' or 'commute'.")

    root = Path(output_dir)
    report_path = root / f"{report_date}_morning_briefing.md"
    bundle_path = root / "raw" / f"{report_date}_bundle.json"
    dashboard_path = root / "charts" / f"dashboard_{report_date}.png"
    catalyst_radar_path = root / "charts" / f"catalyst_radar_{report_date}.png"
    daily_chart_path = root / "charts" / f"daily_one_chart_{report_date}.png"
    email_preview_path = root / f"{report_date}_email_preview.html"
    wecom_preview_path = root / f"{report_date}_wecom_preview.md"
    wecom_html_path = root / f"{report_date}_morning_briefing.html"

    errors: List[str] = []
    warnings: List[str] = []

    for label, path in [
        ("report", report_path),
        ("bundle", bundle_path),
        ("dashboard", dashboard_path),
        ("catalyst radar", catalyst_radar_path),
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

    for label, path in (("WeCom summary preview", wecom_preview_path), ("WeCom HTML preview", wecom_html_path)):
        if require_wecom_preview and not path.exists():
            errors.append(f"Missing required {label} file: {path}")
        elif path.exists() and path.stat().st_size == 0:
            errors.append(f"{label} file is empty: {path}")

    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    bundle = _load_json(bundle_path) if bundle_path.exists() else {}
    word_count = len(re.findall(r"\b[\w'-]+\b", re.sub(r"https?://\S+", "", report_text)))
    heading_count = sum(1 for line in report_text.splitlines() if re.match(r"^#{1,6}\s+", line))
    if word_count > REPORT_HARD_MAX_WORDS:
        errors.append(
            f"Report is too long for the commute edition: {word_count} words exceeds the {REPORT_HARD_MAX_WORDS}-word hard limit."
        )
    elif word_count > REPORT_TARGET_WORDS[1]:
        warnings.append(
            f"Report is {word_count} words; the commute-edition target is {REPORT_TARGET_WORDS[0]}-{REPORT_TARGET_WORDS[1]}."
        )
    elif report_text and word_count < REPORT_TARGET_WORDS[0]:
        warnings.append(
            f"Report is {word_count} words; verify that the deep-read layer is sufficient for a one-hour commute."
        )
    if heading_count > MAX_REPORT_HEADINGS:
        warnings.append(
            f"Report has {heading_count} headings; reduce navigation fragmentation below "
            f"{MAX_REPORT_HEADINGS} where practical."
        )

    for label, markers in REQUIRED_REPORT_SECTION_GROUPS:
        if not any(_section_present(report_text, marker) for marker in markers):
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
            message = "Report quality requires manual review."
            if quality_policy == "strict":
                errors.append(f"{message} Automatic distribution is blocked under the strict policy.")
            else:
                warnings.append(
                    f"{message} Commute delivery remains enabled with the report's visible release caveat."
                )

        fact_check = bundle.get("fact_check", {}) or {}
        if fact_check.get("status") == "error" or fact_check.get("release_blocking"):
            message = "Fact-check guardrail has unresolved critical findings."
            if quality_policy == "strict":
                errors.append(f"{message} Automatic distribution is blocked under the strict policy.")
            else:
                warnings.append(
                    f"{message} Commute delivery is restricted to the caveated report and deterministic fallback copy."
                )
        elif fact_check.get("status") == "warning":
            warnings.append("Fact-check review findings were downgraded to deterministic fallback fields; inspect the audit trail.")

        provenance_audit = bundle.get("provenance_audit", {}) or {}
        if provenance_audit.get("status") != "ok":
            details = "; ".join(str(item) for item in (provenance_audit.get("errors", []) or [])[:4])
            errors.append(f"Source provenance validation failed{': ' + details if details else '.'}")

        source_health = bundle.get("source_health", {}) or {}
        if source_health.get("status") == "failed":
            failures = ", ".join(str(item) for item in (source_health.get("critical_failures", []) or []))
            message = f"Critical source-health policy failed{': ' + failures if failures else '.'}"
            if quality_policy == "strict":
                errors.append(message)
            else:
                warnings.append(
                    f"{message} Commute delivery remains enabled only because the gap is disclosed in the report."
                )

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
        for marker in ("REPORT QUALITY", "Hong Kong local checks", "DEEP READ"):
            if marker not in html:
                warnings.append(f"Email preview is missing expected marker: {marker}")

    if wecom_preview_path.exists():
        wecom_markdown = wecom_preview_path.read_text(encoding="utf-8")
        byte_count = len(wecom_markdown.encode("utf-8"))
        if byte_count > WECOM_SAFE_MARKDOWN_BYTE_LIMIT:
            errors.append(
                f"WeCom summary is {byte_count} bytes; safe delivery budget is {WECOM_SAFE_MARKDOWN_BYTE_LIMIT}."
            )
        for marker in ("5-minute scan", "## Decision frame", "**Invalidate:**", "Open full report"):
            if marker not in wecom_markdown:
                errors.append(f"WeCom summary preview is missing required marker: {marker}")

    if wecom_html_path.exists():
        wecom_html = wecom_html_path.read_text(encoding="utf-8")
        for marker in ('name="viewport"', "reading-path", "report-grid", "Morning Market Brief"):
            if marker not in wecom_html:
                errors.append(f"WeCom HTML preview is missing required marker: {marker}")

    status = "ok" if not errors else "error"
    return {
        "status": status,
        "quality_policy": quality_policy,
        "report_date": report_date,
        "errors": errors,
        "warnings": warnings,
        "reading_profile": {
            "word_count": word_count,
            "heading_count": heading_count,
            "target_words": list(REPORT_TARGET_WORDS),
            "hard_max_words": REPORT_HARD_MAX_WORDS,
            # Derived from the actual word count rather than asserted. The
            # fixed "35-50 minutes" claim was roughly double the real length.
            "estimated_read_minutes": round(word_count / 200.0, 1),
            "reading_speed_wpm": 200,
        },
        "checked_files": {
            "report": str(report_path),
            "bundle": str(bundle_path),
            "dashboard": str(dashboard_path),
            "catalyst_radar": str(catalyst_radar_path),
            "daily_chart": str(daily_chart_path),
            "email_preview": str(email_preview_path),
            "wecom_preview": str(wecom_preview_path),
            "wecom_html": str(wecom_html_path),
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
