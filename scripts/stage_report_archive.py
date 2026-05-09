from __future__ import annotations

import argparse
import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Set

from build_report_gallery import build_report_gallery


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "reports_professional"
ARCHIVE_ROOT = ARCHIVE_DIR / "archive"
CHART_PATTERN = re.compile(r"charts/[A-Za-z0-9_.-]+")


def _run_git_add(paths: Iterable[Path]) -> None:
    normalized = [str(path.relative_to(ROOT)).replace("\\", "/") for path in paths]
    if not normalized:
        return
    subprocess.run(["git", "add", "-f", *normalized], cwd=str(ROOT), check=True)


def _runtime_report_dates() -> Set[str]:
    return {path.name.replace("_morning_briefing.md", "") for path in ARCHIVE_DIR.glob("*_morning_briefing.md")}


def _archived_report_dates() -> Set[str]:
    if not ARCHIVE_ROOT.exists():
        return set()
    return {path.name for path in ARCHIVE_ROOT.iterdir() if path.is_dir()}


def _report_dates(report_date: str | None, all_reports: bool) -> List[str]:
    if all_reports:
        return sorted(_runtime_report_dates() | _archived_report_dates())
    if not report_date:
        raise ValueError("--report-date is required unless --all is used")
    return [report_date]


def referenced_archive_files(report_path: Path) -> List[Path]:
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    content = report_path.read_text(encoding="utf-8", errors="replace")
    paths: Set[Path] = {report_path}
    readme = ARCHIVE_DIR / "README.md"
    if readme.exists():
        paths.add(readme)

    for match in CHART_PATTERN.findall(content):
        if "/test_" in match:
            continue
        chart_path = ARCHIVE_DIR / match
        if chart_path.exists() and chart_path.is_file():
            paths.add(chart_path)

    return sorted(paths)


def _copy_file(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _copy_markdown(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    lines = src.read_text(encoding="utf-8", errors="replace").splitlines()
    dst.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    return dst


def _copy_raw_files(report_date: str, destination: Path) -> List[Path]:
    copied: List[Path] = []
    raw_dir = ARCHIVE_DIR / "raw"
    if not raw_dir.exists():
        return copied

    candidates = sorted(raw_dir.glob(f"{report_date}_*.json"))
    for src in candidates:
        copied.append(_copy_file(src, destination / "raw" / src.name))
    return copied


def _existing_archive_files(report_date: str) -> List[Path]:
    date_dir = ARCHIVE_ROOT / report_date
    if not date_dir.exists():
        return []
    return sorted(path for path in date_dir.rglob("*") if path.is_file())


def build_date_archive(
    report_date: str,
    include_all_charts: bool = False,
    include_raw_bundle: bool = False,
) -> List[Path]:
    report_path = ARCHIVE_DIR / f"{report_date}_morning_briefing.md"
    archived_report_path = ARCHIVE_ROOT / report_date / "morning_briefing.md"
    if not report_path.exists():
        if archived_report_path.exists():
            return _existing_archive_files(report_date)
        raise FileNotFoundError(f"Report not found in runtime output or archive: {report_path}")

    date_dir = ARCHIVE_ROOT / report_date
    if date_dir.exists():
        shutil.rmtree(date_dir)
    date_dir.mkdir(parents=True)

    archived: Set[Path] = set()
    archived.add(_copy_markdown(report_path, date_dir / "morning_briefing.md"))

    if include_all_charts:
        chart_paths = [
            path
            for path in sorted((ARCHIVE_DIR / "charts").glob("*"))
            if path.is_file() and not path.name.startswith("test_")
        ]
    else:
        chart_paths = [
            ARCHIVE_DIR / match
            for match in CHART_PATTERN.findall(report_path.read_text(encoding="utf-8", errors="replace"))
            if "/test_" not in match
        ]
        date_charts = [
            path
            for path in sorted((ARCHIVE_DIR / "charts").glob(f"*{report_date}*"))
            if path.is_file() and not path.name.startswith("test_")
        ]
        chart_paths.extend(date_charts)

    for src in chart_paths:
        if src.exists() and src.is_file():
            archived.add(_copy_file(src, date_dir / "charts" / src.name))

    if include_raw_bundle:
        for raw_path in _copy_raw_files(report_date, date_dir):
            archived.add(raw_path)

    readme = ARCHIVE_DIR / "README.md"
    if readme.exists():
        archived.add(readme)

    return sorted(archived)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Stage GitHub-readable report archive files only.")
    parser.add_argument("--report-date", help="Briefing date in YYYY-MM-DD format.")
    parser.add_argument("--all", action="store_true", help="Stage every archived morning briefing and its referenced charts.")
    parser.add_argument(
        "--include-all-charts",
        action="store_true",
        help="Archive every non-test chart currently in reports_professional/charts. Use in a fresh CI run.",
    )
    parser.add_argument(
        "--include-raw-bundle",
        action="store_true",
        help="Also copy raw bundle JSON into the Git-tracked archive. Raw output is otherwise kept as an artifact.",
    )
    args = parser.parse_args(argv)

    staged: Set[Path] = set()
    report_dates = _report_dates(args.report_date, args.all)

    for report_date in report_dates:
        staged.add(ARCHIVE_ROOT / report_date)
        for path in build_date_archive(
            report_date,
            include_all_charts=args.include_all_charts,
            include_raw_bundle=args.include_raw_bundle,
        ):
            staged.add(path)
    for path in build_report_gallery():
        staged.add(path)

    _run_git_add(sorted(staged))
    print(f"Staged {len(staged)} report archive files")
    for path in sorted(staged):
        print(f"- {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
