from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from _bootstrap import ROOT  # noqa: F401
from scripts.render_report_pdf import find_browser, render_report_pdf
from scripts.send_report_wecom import _md_to_html


def test_html_and_pdf_share_one_a4_rendering_source() -> None:
    try:
        find_browser()
    except RuntimeError as exc:
        pytest.skip(str(exc))

    markdown = """# Morning Research Workbench | 2026-08-25

## Executive Summary

- **Market pulse:** Hong Kong opens with a conditional risk posture.
- **Opening implication:** Confirm the signal with breadth and turnover.
- **What would invalidate it:** A weaker first hour with defensive leadership.
- **Priority catalyst:** Verify official issuer and exchange disclosures.

## Layer 1 | Scan (5 min)

### 1.1 Opening Decision Board

| Signal | Evidence | Decision read |
| --- | --- | --- |
| Breadth | Improving | Require local confirmation before adding risk. |
| Turnover | Above average | Participation makes the move more credible. |
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        html_path = root / "2026-08-25_morning_briefing.html"
        pdf_path = root / "2026-08-25_morning_briefing_core.pdf"
        html_path.write_text(_md_to_html(markdown, root, "2026-08-25"), encoding="utf-8")
        try:
            receipt = render_report_pdf(
                html_path,
                pdf_path,
                report_date="2026-08-25",
                scope="core",
                min_pages=1,
                max_pages=0,
                min_text_chars=100,
            )
        except PermissionError as exc:
            pytest.skip(f"local sandbox cannot open the Chrome debugging socket: {exc}")

        assert pdf_path.exists()
        assert receipt["page_size"] == "A4"
        assert receipt["page_count"] >= 1
        assert receipt["low_information_pages"] == []
        assert receipt["page_density"]
