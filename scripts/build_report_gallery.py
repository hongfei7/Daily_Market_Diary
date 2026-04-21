from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports_professional"
ARCHIVE_ROOT = REPORT_ROOT / "archive"


@dataclass(frozen=True)
class ReportEntry:
    date: str
    mode: str
    pulse: str
    quality: str
    report_path: Path
    dashboard_path: Optional[Path]
    daily_chart_path: Optional[Path]
    trend_pack_path: Optional[Path]
    raw_bundle_path: Optional[Path]


def _first_match(text: str, pattern: str) -> str:
    match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
    return match.group(1).strip() if match else ""


def _section_line(text: str, heading: str) -> str:
    marker = f"### {heading}"
    start = text.find(marker)
    if start < 0:
        return ""
    tail = text[start + len(marker) :].splitlines()
    for line in tail:
        value = line.strip()
        if value and not value.startswith("#") and not value.startswith(">"):
            return value
    return ""


def _shorten(text: str, width: int = 110) -> str:
    value = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(value) <= width:
        return value
    return value[: max(0, width - 3)].rstrip() + "..."


def _find_one(folder: Path, pattern: str) -> Optional[Path]:
    matches = sorted(path for path in folder.glob(pattern) if path.is_file())
    return matches[0] if matches else None


def _entry_from_date_dir(date_dir: Path) -> Optional[ReportEntry]:
    report_path = date_dir / "morning_briefing.md"
    if not report_path.exists():
        return None

    text = report_path.read_text(encoding="utf-8", errors="replace")
    mode = _first_match(text, r"^> Mode:\s*`([^`]+)`") or "Unknown"
    pulse = _section_line(text, "1.1 One-Line Market Pulse") or "Pulse unavailable"
    quality = _first_match(text, r"^> Report quality:\s*`([^`]+)`") or "N/A"
    charts_dir = date_dir / "charts"
    raw_dir = date_dir / "raw"
    date_value = date_dir.name

    return ReportEntry(
        date=date_value,
        mode=mode,
        pulse=_shorten(pulse, width=118),
        quality=quality,
        report_path=report_path,
        dashboard_path=_find_one(charts_dir, f"dashboard_{date_value}.png"),
        daily_chart_path=_find_one(charts_dir, f"daily_one_chart_{date_value}.png"),
        trend_pack_path=_find_one(charts_dir, f"hk_trend_pack_{date_value}.png"),
        raw_bundle_path=_find_one(raw_dir, f"{date_value}_bundle.json"),
    )


def collect_report_entries(archive_root: Path = ARCHIVE_ROOT) -> List[ReportEntry]:
    if not archive_root.exists():
        return []
    entries: List[ReportEntry] = []
    for date_dir in sorted((path for path in archive_root.iterdir() if path.is_dir()), reverse=True):
        entry = _entry_from_date_dir(date_dir)
        if entry is not None:
            entries.append(entry)
    return entries


def _rel_link(target: Optional[Path], base: Path, label: str) -> str:
    if target is None:
        return "N/A"
    rel = target.relative_to(base).as_posix()
    return f"[{label}]({rel})"


def _gallery_table(entries: Iterable[ReportEntry], base: Path) -> str:
    rows = [
        "| Date | Mode | Pulse | Quality | Report | Dashboard | One Chart | Trend Pack | Raw |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        rows.append(
            " | ".join(
                [
                    f"| {entry.date}",
                    entry.mode,
                    entry.pulse.replace("|", "/"),
                    entry.quality,
                    _rel_link(entry.report_path, base, "Report"),
                    _rel_link(entry.dashboard_path, base, "Dashboard"),
                    _rel_link(entry.daily_chart_path, base, "One Chart"),
                    _rel_link(entry.trend_pack_path, base, "Trend Pack"),
                    _rel_link(entry.raw_bundle_path, base, "Bundle"),
                ]
            )
            + " |"
        )
    return "\n".join(rows)


def _root_readme(entries: List[ReportEntry], report_root: Path) -> str:
    latest = entries[0] if entries else None
    latest_line = (
        f"- Latest report: [{latest.date} Morning Briefing](./archive/{latest.date}/morning_briefing.md)"
        if latest
        else "- Latest report: not available yet."
    )
    table = _gallery_table(entries, report_root) if entries else "_No archived reports are available yet._"
    return f"""# Professional Report Archive

This folder is the GitHub-readable archive for the professional morning research workbench.

{latest_line}

Each report date is stored as a self-contained folder:

```text
archive/YYYY-MM-DD/
|-- morning_briefing.md
|-- charts/
`-- raw/
```

Root-level generated files are runtime output. Browse the organized `archive/` folder on GitHub.

The same structure is used for daily trading reports, Sunday weekly reviews, weekend event-watch reports, and holiday reopen playbooks.

## Report Gallery

{table}
"""


def _archive_readme(entries: List[ReportEntry], archive_root: Path) -> str:
    table = _gallery_table(entries, archive_root) if entries else "_No archived reports are available yet._"
    return f"""# Report Gallery

This index is generated from archived report folders.

{table}
"""


def build_report_gallery(report_root: Path = REPORT_ROOT, archive_root: Path = ARCHIVE_ROOT) -> List[Path]:
    entries = collect_report_entries(archive_root)
    report_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)

    root_readme = report_root / "README.md"
    archive_readme = archive_root / "README.md"
    root_readme.write_text(_root_readme(entries, report_root), encoding="utf-8")
    archive_readme.write_text(_archive_readme(entries, archive_root), encoding="utf-8")
    return [root_readme, archive_readme]


def main() -> int:
    written = build_report_gallery()
    for path in written:
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
