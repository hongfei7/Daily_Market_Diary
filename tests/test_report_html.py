from __future__ import annotations

import tempfile
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from scripts.send_report_wecom import _md_to_html, _resolve_report_url


def test_professional_html_structure() -> None:
    markdown = """# Morning Research Workbench | 2026-08-03

## Executive Summary

- **Market pulse:** Higher yields offset lower volatility, leaving a conditional Hong Kong setup.

## Layer 1 | Scan

### 1.2 Global Asset Price Dashboard

| Signal | Last / move | Interpretation | Confirmation / invalidation |
| --- | --- | --- | --- |
| S&P 500 | 7,489 / +0.70% | US beta improved. | Confirm with breadth. |
| VIX | 15.99 / -6.44% | Stress eased. | Invalidate if breadth narrows. |
| Event date | 2026-08-03 | Dated catalyst. | Verify the source. |

### 2.5 Company Catalysts and Risk Monitor

<div class="company-event-monitor"><div class="event-monitor-summary">Decision-filtered events</div></div>
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        html = _md_to_html(markdown, Path(tmpdir), "2026-08-03")

    assert "report-shell" in html
    assert "report-toc" in html
    assert "table-shell" in html
    assert '<span class="move-positive">' in html
    assert '<span class="move-negative">' in html
    assert '<span class="move-negative">-08' not in html
    assert "Hong Kong Market Intelligence" in html
    assert "Morning Market Brief" in html
    assert "reading-path" in html
    assert "section-executive-summary" in html
    assert "company-event-monitor" in html
    assert "event-monitor-summary" in html
    assert ".event-monitor-summary { display: block" in html
    assert ".event-read-grid { display: block" in html
    assert "mobile-toc" in html
    assert 'name="viewport"' in html
    assert "overflow-x: hidden" not in html
    assert ".figure-shell" in html
    assert ".reading-path { display: block" in html
    assert "overflow-wrap: anywhere" in html
    assert "linear-gradient" not in html
    assert "background: #1677ff" not in html


def test_header_reads_current_metadata_and_wraps_figures() -> None:
    markdown = """# Morning Research Workbench | 2026-08-25

- **What changed overnight?** FXI reversed the prior Hong Kong cash direction.

*Data through: US/global `2026-08-24` | HK `2026-08-24`*

## Visual Dashboard

![Decision dashboard](charts/dashboard.png)
"""
    with tempfile.TemporaryDirectory() as tmpdir:
        html = _md_to_html(markdown, Path(tmpdir), "2026-08-25", inline_images=False)
    assert "FXI reversed the prior Hong Kong cash direction." in html
    assert "2026-08-24" in html
    assert '<div class="figure-shell"><img' in html


def test_archive_link_is_suppressed_when_publication_failed(monkeypatch) -> None:
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("DMD_REPORT_LINK_ENABLED", "false")
    assert _resolve_report_url("2026-08-25") == ""


def main() -> None:
    test_professional_html_structure()
    print("Report HTML test passed")


if __name__ == "__main__":
    main()
