"""The HTML deliverable must be archived, not just delivered.

The report that actually gets read is the styled HTML — printed, or opened on a
phone. The markdown is an intermediate format. Until now the HTML was generated
in CI, pushed to WeCom and then discarded: a delivery failure (which the quality
gates have caused more than once) left nothing readable for that day, and there
was no way to look back at any earlier day's report.

The archived copy references charts by relative path instead of inlining them.
The charts already sit in ``charts/`` next to it, and inlining costs about 1MB a
day for images that are already stored. The delivered copy still inlines so the
attachment works offline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from send_report_wecom import _md_to_html  # noqa: E402
from stage_report_archive import _write_archive_html  # noqa: E402

MARKDOWN = """# Morning Research Workbench | 2026-08-20

## Visual Dashboard

![Research Dashboard](charts/dashboard_2026-08-20.png)

### 1.1 Yesterday's Call

**Verdict: BROKEN.** The prior call did not work.

### 2.3 AI / TMT Read-Through

| Name | 1D |
| --- | --- |
| SOXX | -2.21% |
"""


@pytest.fixture
def archive_dir(tmp_path):
    source = tmp_path / "2026-08-20_morning_briefing.md"
    source.write_text(MARKDOWN, encoding="utf-8")
    (tmp_path / "charts").mkdir()
    (tmp_path / "charts" / "dashboard_2026-08-20.png").write_bytes(b"\x89PNG" + b"0" * 4000)

    date_dir = tmp_path / "archive" / "2026-08-20"
    (date_dir / "charts").mkdir(parents=True)
    (date_dir / "charts" / "dashboard_2026-08-20.png").write_bytes(b"\x89PNG" + b"0" * 4000)
    return source, date_dir


def test_archive_receives_an_html_copy(archive_dir):
    source, date_dir = archive_dir
    written = _write_archive_html(source, date_dir, "2026-08-20")
    assert written is not None
    assert written.name == "morning_briefing.html"
    assert written.exists()


def test_archived_html_references_charts_rather_than_inlining(archive_dir):
    source, date_dir = archive_dir
    html = _write_archive_html(source, date_dir, "2026-08-20").read_text(encoding="utf-8")
    assert "data:image" not in html
    assert 'src="charts/dashboard_2026-08-20.png"' in html


def test_delivered_html_still_inlines_for_offline_reading(tmp_path):
    (tmp_path / "charts").mkdir()
    (tmp_path / "charts" / "dashboard_2026-08-20.png").write_bytes(b"\x89PNG" + b"0" * 4000)
    html = _md_to_html(MARKDOWN, tmp_path, "2026-08-20", md_source_dir=tmp_path)
    assert "data:image" in html


def test_archived_copy_is_far_smaller_than_the_delivered_one(archive_dir, tmp_path):
    source, date_dir = archive_dir
    archived = _write_archive_html(source, date_dir, "2026-08-20").read_text(encoding="utf-8")
    delivered = _md_to_html(MARKDOWN, tmp_path, "2026-08-20", md_source_dir=source.parent)
    assert len(archived) < len(delivered)


def test_archived_html_keeps_the_stylesheet(archive_dir):
    """Without the CSS the archived copy is not the deliverable, just markup."""
    source, date_dir = archive_dir
    html = _write_archive_html(source, date_dir, "2026-08-20").read_text(encoding="utf-8")
    assert "@media print" in html
    assert "@media (max-width: 430px)" in html
    assert ".company-event-monitor {" in html


def test_rendering_failure_does_not_block_archiving(archive_dir, monkeypatch):
    """A broken HTML render must not cost us the markdown archive."""
    source, date_dir = archive_dir
    monkeypatch.setattr("stage_report_archive.Path.read_text", _raise)
    assert _write_archive_html(source, date_dir, "2026-08-20") is None


def _raise(*args, **kwargs):
    raise OSError("simulated failure")
