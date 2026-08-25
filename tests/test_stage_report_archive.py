from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


SCRIPT_PATH = ROOT / "scripts" / "stage_report_archive.py"


def _load_stage_module():
    scripts_dir = str(SCRIPT_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("stage_report_archive", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load stage_report_archive.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_archive_manifest_is_verified_and_raw_bundle_is_opt_in() -> None:
    stage = _load_stage_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report_root = root / "reports_professional"
        chart_root = report_root / "charts"
        raw_root = report_root / "raw"
        chart_root.mkdir(parents=True)
        raw_root.mkdir(parents=True)

        report_date = "2026-04-19"
        (report_root / f"{report_date}_morning_briefing.md").write_text(
            f"![Dashboard](charts/dashboard_{report_date}.png)\n",
            encoding="utf-8",
        )
        (chart_root / f"dashboard_{report_date}.png").write_bytes(b"png")
        (raw_root / f"{report_date}_bundle.json").write_text("{}", encoding="utf-8")

        stage.ROOT = root
        stage.ARCHIVE_DIR = report_root
        stage.ARCHIVE_ROOT = report_root / "archive"

        archived = stage.build_date_archive(report_date)
        archived_rel = {path.relative_to(root).as_posix() for path in archived}
        assert f"reports_professional/archive/{report_date}/morning_briefing.md" in archived_rel
        assert f"reports_professional/archive/{report_date}/charts/dashboard_{report_date}.png" in archived_rel
        assert f"reports_professional/archive/{report_date}/raw/{report_date}_bundle.json" not in archived_rel
        assert not (stage.ARCHIVE_ROOT / report_date / "raw" / f"{report_date}_bundle.json").exists()
        assert not (stage.ARCHIVE_ROOT / report_date / "README.md").exists()
        assert (stage.ARCHIVE_ROOT / report_date / "manifest.json").exists()
        assert stage.verify_archive_manifest(stage.ARCHIVE_ROOT / report_date)["status"] == "ok"
        (stage.ARCHIVE_ROOT / report_date / "._morning_briefing.md").write_bytes(b"appledouble")
        assert stage.verify_archive_manifest(stage.ARCHIVE_ROOT / report_date)["status"] == "ok"
        assert all("._" not in item["path"] for item in stage.build_archive_manifest(stage.ARCHIVE_ROOT / report_date, report_date)["files"])
        stage.write_archive_integrity_index(stage.ARCHIVE_ROOT)
        history_audit = stage.verify_archive_integrity_index(stage.ARCHIVE_ROOT)
        assert history_audit["status"] == "ok"
        assert history_audit["dates"] == 1

        try:
            stage.build_date_archive(report_date, include_raw_bundle=True)
        except stage.ArchiveConflictError:
            pass
        else:
            raise AssertionError("A published archive must not be overwritten by a conflicting rerun")


def test_raw_bundle_can_be_included_on_first_publish() -> None:
    stage = _load_stage_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report_root = root / "reports_professional"
        chart_root = report_root / "charts"
        raw_root = report_root / "raw"
        chart_root.mkdir(parents=True)
        raw_root.mkdir(parents=True)
        report_date = "2026-04-20"
        (report_root / f"{report_date}_morning_briefing.md").write_text("# Report\n", encoding="utf-8")
        (raw_root / f"{report_date}_bundle.json").write_text("{}", encoding="utf-8")
        stage.ROOT = root
        stage.ARCHIVE_DIR = report_root
        stage.ARCHIVE_ROOT = report_root / "archive"

        archived = stage.build_date_archive(report_date, include_raw_bundle=True)
        archived_rel = {path.relative_to(root).as_posix() for path in archived}
        assert f"reports_professional/archive/{report_date}/raw/{report_date}_bundle.json" in archived_rel


def test_all_dates_include_archived_only_reports() -> None:
    stage = _load_stage_module()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        report_root = root / "reports_professional"
        archived_date = report_root / "archive" / "2026-04-18"
        archived_date.mkdir(parents=True)
        (archived_date / "morning_briefing.md").write_text("# Archived only\n", encoding="utf-8")

        stage.ROOT = root
        stage.ARCHIVE_DIR = report_root
        stage.ARCHIVE_ROOT = report_root / "archive"

        assert stage._report_dates(None, True) == ["2026-04-18"]
        archived = stage.build_date_archive("2026-04-18")
        archived_rel = {path.relative_to(root).as_posix() for path in archived}
        assert archived_rel == {"reports_professional/archive/2026-04-18/morning_briefing.md"}


def test_metric_history_is_part_of_tracked_performance_artifacts() -> None:
    stage = _load_stage_module()
    with tempfile.TemporaryDirectory() as tmp:
        report_root = Path(tmp) / "reports_professional"
        performance = report_root / "performance"
        performance.mkdir(parents=True)
        (performance / "metric_history.json").write_text("{}", encoding="utf-8")
        stage.ARCHIVE_DIR = report_root
        assert [path.name for path in stage._performance_files()] == ["metric_history.json"]


def main() -> None:
    test_archive_manifest_is_verified_and_raw_bundle_is_opt_in()
    test_raw_bundle_can_be_included_on_first_publish()
    test_all_dates_include_archived_only_reports()
    test_metric_history_is_part_of_tracked_performance_artifacts()
    print("Stage report archive test passed")


if __name__ == "__main__":
    main()
