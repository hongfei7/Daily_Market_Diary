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


def test_raw_bundle_is_opt_in_for_archive() -> None:
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

        archived_with_raw = stage.build_date_archive(report_date, include_raw_bundle=True)
        archived_with_raw_rel = {path.relative_to(root).as_posix() for path in archived_with_raw}
        assert f"reports_professional/archive/{report_date}/raw/{report_date}_bundle.json" in archived_with_raw_rel


def main() -> None:
    test_raw_bundle_is_opt_in_for_archive()
    print("Stage report archive test passed")


if __name__ == "__main__":
    main()
