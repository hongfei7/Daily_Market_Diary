from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401


SCRIPT_PATH = ROOT / "scripts" / "build_report_gallery.py"


def _load_gallery_module():
    spec = importlib.util.spec_from_file_location("build_report_gallery", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load build_report_gallery.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gallery_generation_from_archive() -> None:
    gallery = _load_gallery_module()
    with tempfile.TemporaryDirectory() as tmp:
        report_root = Path(tmp) / "reports_professional"
        date_dir = report_root / "archive" / "2026-04-19"
        (date_dir / "charts").mkdir(parents=True)
        (date_dir / "raw").mkdir(parents=True)
        (date_dir / "charts" / "dashboard_2026-04-19.png").write_bytes(b"png")
        (date_dir / "charts" / "daily_one_chart_2026-04-19.png").write_bytes(b"png")
        (date_dir / "charts" / "hk_trend_pack_2026-04-19.png").write_bytes(b"png")
        (date_dir / "raw" / "2026-04-19_bundle.json").write_text("{}", encoding="utf-8")
        (date_dir / "morning_briefing.md").write_text(
            """# Morning Research Workbench | 2026-04-19

> Mode: `Weekly Review` | Weekly note.
> Report quality: `82.4/100` | Grade `B`

### 1.1 One-Line Market Pulse
Weekly pulse with clean Hong Kong follow-through and next-week preparation.
""",
            encoding="utf-8",
        )

        written = gallery.build_report_gallery(report_root=report_root, archive_root=report_root / "archive")
        assert report_root / "README.md" in written
        assert report_root / "archive" / "README.md" in written
        assert report_root / "latest" / "README.md" in written
        assert report_root / "archive" / "2026-04-19" / "README.md" in written

        root_index = (report_root / "README.md").read_text(encoding="utf-8")
        archive_index = (report_root / "archive" / "README.md").read_text(encoding="utf-8")
        latest_index = (report_root / "latest" / "README.md").read_text(encoding="utf-8")
        date_index = (report_root / "archive" / "2026-04-19" / "README.md").read_text(encoding="utf-8")
        assert "Open the stable latest entry" in root_index
        assert "./latest/README.md" in root_index
        assert "./archive/2026-04-19/README.md" in root_index
        assert "Weekly Review" in root_index
        assert "82.4/100" in root_index
        assert "Weekly pulse with clean Hong Kong follow-through" in root_index
        assert "[Report](archive/2026-04-19/README.md)" in root_index
        assert "archive/2026-04-19/README.md" in root_index
        assert "archive/2026-04-19/charts/dashboard_2026-04-19.png" in root_index
        assert "[Report](2026-04-19/README.md)" in archive_index
        assert "2026-04-19/charts/dashboard_2026-04-19.png" in archive_index
        assert "2026-04-19/raw/2026-04-19_bundle.json" in archive_index
        assert "stable GitHub entry" in latest_index
        assert "../archive/2026-04-19/README.md" in latest_index
        assert "![Dashboard](../archive/2026-04-19/charts/dashboard_2026-04-19.png)" in latest_index
        assert "One-line pulse: Weekly pulse with clean Hong Kong follow-through and next-week preparation." in latest_index
        assert "Quick start" in latest_index
        assert "# Morning Research Workbench" not in latest_index
        assert "GitHub landing page for the archived report dated `2026-04-19`" in date_index
        assert "../../latest/README.md" in date_index
        assert "![Dashboard](charts/dashboard_2026-04-19.png)" in date_index
        assert "[morning_briefing.md](./morning_briefing.md)" in date_index
        assert "How to use this folder" in date_index
        assert "# Morning Research Workbench" not in date_index


def main() -> None:
    test_gallery_generation_from_archive()
    print("Report gallery test passed")


if __name__ == "__main__":
    main()
