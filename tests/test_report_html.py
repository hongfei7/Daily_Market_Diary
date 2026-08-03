from __future__ import annotations

import tempfile
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401
from scripts.send_report_wecom import _md_to_html


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
    assert "mobile-toc" in html
    assert "linear-gradient" not in html
    assert "background: #1677ff" not in html


def main() -> None:
    test_professional_html_structure()
    print("Report HTML test passed")


if __name__ == "__main__":
    main()
