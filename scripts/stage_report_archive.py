from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Set

from build_report_gallery import build_report_gallery


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_DIR = ROOT / "reports_professional"
ARCHIVE_ROOT = ARCHIVE_DIR / "archive"
CHART_PATTERN = re.compile(r"charts/[A-Za-z0-9_.-]+")
MANIFEST_SCHEMA = "report-archive-manifest-v1"
INTEGRITY_INDEX_SCHEMA = "report-archive-integrity-index-v1"


class ArchiveConflictError(RuntimeError):
    """Raised when a published date would be silently overwritten."""


def _is_safe_archive_file(path: Path) -> bool:
    """Exclude macOS metadata and hidden runtime files from tracked archives."""
    return path.is_file() and not path.name.startswith(".")


def _run_git_add(paths: Iterable[Path]) -> None:
    normalized = [
        str(path.relative_to(ROOT)).replace("\\", "/")
        for path in paths
        if _is_safe_archive_file(path)
    ]
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


def _write_archive_html(report_path: Path, date_dir: Path, report_date: str) -> Optional[Path]:
    """Archive the styled HTML alongside the markdown.

    The HTML is the deliverable that actually gets read — printed, or opened on
    a phone — while the markdown is an intermediate format. It was only ever
    generated in CI and pushed to WeCom, so a delivery failure left nothing
    readable for that day and no historical copy to go back to.

    Images are referenced rather than inlined: the charts sit in ``charts/``
    right next to this file, and inlining would add roughly 1MB per day to the
    repository for images already stored.
    """
    try:
        from send_report_wecom import _md_to_html  # noqa: PLC0415
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        try:
            from send_report_wecom import _md_to_html  # noqa: PLC0415
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[archive] HTML rendering unavailable, archiving markdown only: {exc}")
            return None

    try:
        markdown = report_path.read_text(encoding="utf-8", errors="replace")
        html = _md_to_html(
            markdown,
            date_dir,
            report_date,
            md_source_dir=report_path.parent,
            inline_images=False,
        )
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[archive] HTML rendering failed, archiving markdown only: {exc}")
        return None

    destination = date_dir / "morning_briefing.html"
    destination.write_text(html, encoding="utf-8")
    return destination


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
    return sorted(path for path in date_dir.rglob("*") if _is_safe_archive_file(path))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_files(date_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in date_dir.rglob("*")
        if _is_safe_archive_file(path) and path.name not in {"README.md", "manifest.json"}
    )


def build_archive_manifest(date_dir: Path, report_date: str) -> dict:
    files = [
        {
            "path": path.relative_to(date_dir).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _payload_files(date_dir)
    ]
    archive_id = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": MANIFEST_SCHEMA,
        "report_date": report_date,
        "archive_id": archive_id,
        "immutability": "Published payload files are append-only. A conflicting rerun must use a new report date.",
        "manifest_scope": "README.md is generated navigation and is intentionally excluded from payload hashes.",
        "files": files,
    }


def write_archive_manifest(date_dir: Path, report_date: str) -> Path:
    path = date_dir / "manifest.json"
    path.write_text(
        json.dumps(build_archive_manifest(date_dir, report_date), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def verify_archive_manifest(date_dir: Path) -> dict:
    manifest_path = date_dir / "manifest.json"
    if not manifest_path.exists():
        return {"status": "legacy_unverified", "errors": ["manifest.json is missing"]}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = build_archive_manifest(date_dir, str(manifest.get("report_date", date_dir.name)))
    errors = []
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        errors.append("unsupported manifest schema")
    if manifest.get("archive_id") != expected.get("archive_id"):
        errors.append("archive payload hash does not match manifest")
    return {
        "status": "ok" if not errors else "error",
        "archive_id": manifest.get("archive_id", ""),
        "files": len(manifest.get("files", []) or []),
        "errors": errors,
    }


def build_archive_integrity_index(archive_root: Path = ARCHIVE_ROOT) -> dict:
    entries = []
    for date_dir in sorted(path for path in archive_root.iterdir() if path.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", path.name)):
        manifest = build_archive_manifest(date_dir, date_dir.name)
        entries.append(
            {
                "report_date": date_dir.name,
                "archive_id": manifest["archive_id"],
                "files": manifest["files"],
            }
        )
    archive_id = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": INTEGRITY_INDEX_SCHEMA,
        "archive_id": archive_id,
        "dates": len(entries),
        "scope": "All dated archive payload files; generated README.md and per-date manifest.json are excluded.",
        "entries": entries,
    }


def write_archive_integrity_index(archive_root: Path = ARCHIVE_ROOT) -> Path:
    path = archive_root / "integrity_manifest.json"
    payload = build_archive_integrity_index(archive_root)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_archive_integrity_index(archive_root: Path = ARCHIVE_ROOT) -> dict:
    path = archive_root / "integrity_manifest.json"
    if not path.exists():
        return {"status": "error", "errors": ["integrity_manifest.json is missing"]}
    try:
        stored = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "errors": [f"invalid integrity index: {exc}"]}
    expected = build_archive_integrity_index(archive_root)
    errors = []
    if stored.get("schema_version") != INTEGRITY_INDEX_SCHEMA:
        errors.append("unsupported integrity index schema")
    if stored.get("archive_id") != expected.get("archive_id"):
        errors.append("archive history hash does not match integrity index")
    return {
        "status": "ok" if not errors else "error",
        "archive_id": stored.get("archive_id", ""),
        "dates": stored.get("dates", 0),
        "errors": errors,
    }


def _copy_audit_files(report_date: str, destination: Path) -> List[Path]:
    copied: List[Path] = []
    raw_dir = ARCHIVE_DIR / "raw"
    # llm_health and prose_guard are archived so recurring failure causes and
    # prose defects can be counted across days rather than inspected one run at
    # a time.
    for suffix in ("source_health", "performance_summary", "llm_health", "prose_guard"):
        src = raw_dir / f"{report_date}_{suffix}.json"
        if src.exists():
            copied.append(_copy_file(src, destination / "audit" / f"{suffix}.json"))
    return copied


def _performance_files() -> List[Path]:
    root = ARCHIVE_DIR / "performance"
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.name in {"README.md", "metric_history.json", "signal_ledger.json", "performance_summary.json", "signal_performance.png"}
    )


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

    target_dir = ARCHIVE_ROOT / report_date
    ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".archive-{report_date}-", dir=str(ARCHIVE_ROOT)) as temp_root:
        date_dir = Path(temp_root) / report_date
        date_dir.mkdir(parents=True)
        _copy_markdown(report_path, date_dir / "morning_briefing.md")

        report_text = report_path.read_text(encoding="utf-8", errors="replace")
        chart_paths = {
            ARCHIVE_DIR / match
            for match in CHART_PATTERN.findall(report_text)
            if "/test_" not in match
        }
        if include_all_charts:
            chart_paths.update(
                path
                for path in sorted((ARCHIVE_DIR / "charts").glob(f"*{report_date}*"))
                if _is_safe_archive_file(path) and not path.name.startswith("test_")
            )
        for src in sorted(chart_paths):
            if src.exists() and src.is_file():
                _copy_file(src, date_dir / "charts" / src.name)

        _write_archive_html(report_path, date_dir, report_date)

        for audit_path in _copy_audit_files(report_date, date_dir):
            _ = audit_path
        if include_raw_bundle:
            _copy_raw_files(report_date, date_dir)
        candidate_manifest = write_archive_manifest(date_dir, report_date)
        candidate_id = json.loads(candidate_manifest.read_text(encoding="utf-8"))["archive_id"]

        if target_dir.exists():
            verification = verify_archive_manifest(target_dir)
            if verification.get("status") == "ok" and verification.get("archive_id") == candidate_id:
                return _existing_archive_files(report_date)
            raise ArchiveConflictError(
                f"Archive {report_date} already exists with different or legacy-unverified content; "
                "published dates are immutable. Use a new report date instead of overwriting it."
            )

        shutil.move(str(date_dir), str(target_dir))

    verification = verify_archive_manifest(target_dir)
    if verification.get("status") != "ok":
        raise RuntimeError(f"Archive manifest verification failed: {verification.get('errors', [])}")
    archived = set(_existing_archive_files(report_date))
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
        for path in build_date_archive(
            report_date,
            include_all_charts=args.include_all_charts,
            include_raw_bundle=args.include_raw_bundle,
        ):
            staged.add(path)
    for path in build_report_gallery():
        staged.add(path)
    integrity_index = write_archive_integrity_index()
    verification = verify_archive_integrity_index()
    if verification.get("status") != "ok":
        raise RuntimeError(f"Archive integrity index verification failed: {verification.get('errors', [])}")
    staged.add(integrity_index)
    for path in _performance_files():
        staged.add(path)

    _run_git_add(sorted(staged))
    print(f"Staged {len(staged)} report archive files")
    for path in sorted(staged):
        print(f"- {path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
