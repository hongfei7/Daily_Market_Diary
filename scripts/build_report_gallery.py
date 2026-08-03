from __future__ import annotations

import re
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports_professional"
ARCHIVE_ROOT = REPORT_ROOT / "archive"
LATEST_ROOT = REPORT_ROOT / "latest"


@dataclass(frozen=True)
class ReportEntry:
    date: str
    mode: str
    pulse: str
    quality: str
    report_path: Path
    dashboard_path: Optional[Path]
    catalyst_radar_path: Optional[Path]
    daily_chart_path: Optional[Path]
    trend_pack_path: Optional[Path]
    raw_bundle_path: Optional[Path]
    manifest_path: Optional[Path]
    source_health: str
    performance_status: str


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


def _json_status(path: Path) -> str:
    if not path.exists():
        return "N/A"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "Invalid"
    return str(payload.get("status", "N/A")).replace("_", " ") if isinstance(payload, dict) else "Invalid"


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
    audit_dir = date_dir / "audit"
    date_value = date_dir.name

    return ReportEntry(
        date=date_value,
        mode=mode,
        pulse=_shorten(pulse, width=118),
        quality=quality,
        report_path=report_path,
        dashboard_path=_find_one(charts_dir, f"dashboard_{date_value}.png"),
        catalyst_radar_path=_find_one(charts_dir, f"catalyst_radar_{date_value}.png"),
        daily_chart_path=_find_one(charts_dir, f"daily_one_chart_{date_value}.png"),
        trend_pack_path=_find_one(charts_dir, f"hk_trend_pack_{date_value}.png"),
        raw_bundle_path=_find_one(raw_dir, f"{date_value}_bundle.json"),
        manifest_path=_find_one(date_dir, "manifest.json"),
        source_health=_json_status(audit_dir / "source_health.json"),
        performance_status=_json_status(audit_dir / "performance_summary.json"),
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
    rel = os.path.relpath(target, start=base).replace("\\", "/")
    return f"[{label}]({rel})"


def _rel_target(target: Optional[Path], base: Path) -> str:
    if target is None:
        return ""
    return os.path.relpath(target, start=base).replace("\\", "/")


def _gallery_table(entries: Iterable[ReportEntry], base: Path) -> str:
    rows = [
        "| Date | Mode | Pulse | Quality | Report | Dashboard | Event Radar | One Chart | Trend Pack | Raw |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in entries:
        rows.append(
            " | ".join(
                [
                    f"| {entry.date}",
                    entry.mode,
                    entry.pulse.replace("|", "/"),
                    entry.quality,
                    _rel_link(entry.report_path.parent / "README.md", base, "Report"),
                    _rel_link(entry.dashboard_path, base, "Dashboard"),
                    _rel_link(entry.catalyst_radar_path, base, "Event Radar"),
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
    latest_line = f"- Latest report: [Open the stable latest entry](./latest/README.md)" if latest else "- Latest report: not available yet."
    latest_archive_line = (
        f"- Latest archive folder: [{latest.date}](./archive/{latest.date}/README.md)"
        if latest
        else "- Latest archive folder: not available yet."
    )
    table = _gallery_table(entries, report_root) if entries else "_No archived reports are available yet._"
    return f"""# Professional Report Archive

This folder is the GitHub-readable archive for the professional morning research workbench.

{latest_line}
{latest_archive_line}

Each report date is stored as a self-contained folder:

```text
archive/YYYY-MM-DD/
|-- README.md
|-- morning_briefing.md
|-- charts/
|-- audit/
|-- manifest.json
`-- raw/
```

`latest/README.md` is the stable GitHub landing page for the newest published report.
Each dated folder also includes a `README.md`, so opening that folder on GitHub renders the report immediately.
Root-level generated files are runtime output. Browse `latest/` for the newest report or the organized `archive/` folder for history.

The same structure is used for daily trading reports, Sunday weekly reviews, weekend event-watch reports, and holiday reopen playbooks.

Published date payloads are immutable and carry a SHA-256 manifest. Source-health and backtest snapshots are archived separately from the optional full raw bundle.

- [Signal performance ledger](./performance/README.md)
- [Full archive integrity index](./archive/integrity_manifest.json)

## Report Gallery

{table}
"""


def _archive_readme(entries: List[ReportEntry], archive_root: Path) -> str:
    table = _gallery_table(entries, archive_root) if entries else "_No archived reports are available yet._"
    return f"""# Report Gallery

This index is generated from archived report folders.

Open a date folder directly on GitHub to read its `README.md` version of the report.
Use `../latest/README.md` when you want the newest published report without checking dates first.

{table}
"""


def _landing_asset_lines(entry: ReportEntry, base: Path) -> List[str]:
    lines = [
        f"- Dashboard: {_rel_link(entry.dashboard_path, base, 'Open image')}",
        f"- Catalyst & Event Radar: {_rel_link(entry.catalyst_radar_path, base, 'Open image')}",
        f"- Daily One Chart: {_rel_link(entry.daily_chart_path, base, 'Open image')}",
        f"- Trend Pack: {_rel_link(entry.trend_pack_path, base, 'Open image')}",
        f"- Raw bundle: {_rel_link(entry.raw_bundle_path, base, 'Open bundle')}",
    ]
    if entry.manifest_path is not None:
        lines.append(f"- Integrity manifest: {_rel_link(entry.manifest_path, base, 'Verify SHA-256 payload')}")
    if entry.source_health != "N/A":
        lines.append(f"- Source health: `{entry.source_health}`")
    if entry.performance_status != "N/A":
        lines.append(f"- Backtest status: `{entry.performance_status}`")
    return lines


def _dashboard_preview(entry: ReportEntry, base: Path) -> str:
    if entry.dashboard_path is None:
        return ""
    return f"\n## Dashboard Preview\n\n![Dashboard]({_rel_target(entry.dashboard_path, base)})\n"


def _date_readme(entry: ReportEntry) -> str:
    date_dir = entry.report_path.parent
    asset_lines = _landing_asset_lines(entry, date_dir)
    dashboard_preview = _dashboard_preview(entry, date_dir)
    return f"""# Archived Professional Report | {entry.date}

This is the GitHub landing page for the archived report dated `{entry.date}`.

- Report date: `{entry.date}`
- Report mode: `{entry.mode}`
- Quality: `{entry.quality}`
- One-line pulse: {entry.pulse}
- Latest published entry: [latest/README.md](../../latest/README.md)
- Archive gallery: [archive/README.md](../README.md)
- Direct markdown file: [morning_briefing.md](./morning_briefing.md)
{chr(10).join(asset_lines)}

{dashboard_preview}

## How to use this folder

1. Start with the dashboard preview for a quick visual read.
2. Open [morning_briefing.md](./morning_briefing.md) for the full report text.
3. Use `charts/` and `raw/` only when you need supporting assets or audit data.
"""


def _latest_readme(entries: List[ReportEntry], report_root: Path) -> str:
    if not entries:
        return """# Latest Professional Report

No archived reports are available yet.
"""

    latest = entries[0]
    latest_base = report_root / "latest"
    asset_lines = _landing_asset_lines(latest, latest_base)
    dashboard_preview = _dashboard_preview(latest, latest_base)
    return f"""# Latest Professional Report

This is the stable GitHub entry for the newest archived report.

- Report date: `{latest.date}`
- Report mode: `{latest.mode}`
- Quality: `{latest.quality}`
- One-line pulse: {latest.pulse}
- Archived folder: [archive/{latest.date}](../archive/{latest.date}/README.md)
- Direct markdown file: [morning_briefing.md](../archive/{latest.date}/morning_briefing.md)
{chr(10).join(asset_lines)}

{dashboard_preview}

## Quick start

1. Open the archived landing page for navigation and context: [archive/{latest.date}/README.md](../archive/{latest.date}/README.md)
2. Open [morning_briefing.md](../archive/{latest.date}/morning_briefing.md) if you want the full markdown report.
3. Use the chart and bundle links above when you need the production assets behind the report.
"""


def build_report_gallery(report_root: Path = REPORT_ROOT, archive_root: Path = ARCHIVE_ROOT) -> List[Path]:
    entries = collect_report_entries(archive_root)
    report_root.mkdir(parents=True, exist_ok=True)
    archive_root.mkdir(parents=True, exist_ok=True)
    latest_root = report_root / "latest"
    latest_root.mkdir(parents=True, exist_ok=True)

    root_readme = report_root / "README.md"
    archive_readme = archive_root / "README.md"
    latest_readme = latest_root / "README.md"
    root_readme.write_text(_root_readme(entries, report_root), encoding="utf-8")
    archive_readme.write_text(_archive_readme(entries, archive_root), encoding="utf-8")
    latest_readme.write_text(_latest_readme(entries, report_root), encoding="utf-8")
    written = [root_readme, archive_readme, latest_readme]
    for entry in entries:
        date_readme = entry.report_path.parent / "README.md"
        date_readme.write_text(_date_readme(entry), encoding="utf-8")
        written.append(date_readme)
    return written


def main() -> int:
    written = build_report_gallery()
    for path in written:
        print(path.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
