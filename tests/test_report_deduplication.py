"""Sections 2.1 and 2.2 must not restate each other.

Both blocks rendered the same ``hk_desk_view`` fields, so every report carried
the style call twice and three word-for-word identical "Cross-market read"
bullets. Around 44% of lines were byte-identical to the previous day's report.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.report_builder import render_professional_report


def _item(price, pct, age=1):
    return {
        "Price": price,
        "Pct Change": pct,
        "Trading Freshness Days": age,
        "As Of": "2026-08-18",
        "Change Value": pct,
        "Change Unit": "pct",
        "Quality": "fresh",
    }


def _report(hstech_age: int = 1) -> str:
    summary = {
        "Equities": {
            "S&P 500": _item(7691.76, -0.69),
            "Nasdaq 100": _item(29490.96, -1.68),
            "Hang Seng Index": _item(25453.23, 1.34),
            "Hang Seng China Enterprises": _item(9000.0, 1.19),
            "Hang Seng TECH ETF": _item(4.616, -1.87, hstech_age),
            "China Large-Cap (FXI)": _item(40.0, -0.11),
        },
        "FX": {"DXY": _item(99.64, 0.0), "USD/CNH": _item(6.735, 0.05), "USD/HKD": _item(7.8431, 0.0)},
        "Vol": {"VIX": _item(15.84, 4.28)},
    }
    bundle = build_professional_bundle(
        report_date="2026-08-18",
        briefing_date="2026-08-19",
        global_market_date="2026-08-18",
        hk_data_date="2026-08-18",
        config=load_professional_config(),
        market_data={"summary": summary, "meta": {"requested_date": "2026-08-18", "effective_date": "2026-08-18"}},
        chart_features={},
        macro_data={"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
        sector_data={},
        movers_data={},
        risk_data={},
        news_headlines=[],
    )
    return render_professional_report(bundle, charts_section="")


def test_cross_market_read_is_not_duplicated():
    report = _report()
    assert report.count("- **Cross-market read:**") <= 3, "the same lines were rendered in both 2.1 and 2.2"


def test_style_call_appears_once_outside_the_summary():
    """2.1 must point at 2.2 rather than restating the desk lens."""
    report = _report()
    assert "**Desk lens.**" not in report
    assert "**Investment read.**" not in report
    assert "set out in the Hong Kong review below" in report


def test_cross_references_use_section_names_not_numbers():
    """Numbers shift when a section is inserted; names do not.

    Adding "2.3 AI / TMT Read-Through" pushed Flow Tracker to 2.4 and left three
    body references pointing at the wrong section — the same coupling that broke
    the release audit.
    """
    import re

    report = _report()
    stale = re.findall(r"(?<!### )Section \d\.\d", report)
    assert not stale, f"cross-references must name the section, not its number: {stale}"


def test_no_prose_line_repeats_verbatim():
    report = _report()
    prose = [
        line.strip()
        for line in report.splitlines()
        if line.strip().startswith("- **") and len(line.strip()) > 80
    ]
    assert len(prose) == len(set(prose)), "a long prose bullet was rendered more than once"


def test_absent_macro_source_is_described_as_missing_coverage():
    """An empty feed must not be reported as a quiet calendar."""
    report = _report()
    assert "macro calendar was light" not in report
    assert "calendar is relatively light" not in report
    assert "not evidence that the calendar was quiet" in report or "missing coverage" in report


def test_empty_forward_sections_collapse_to_one_statement():
    report = _report()
    # The three hollow sub-blocks are replaced by a single honest line.
    assert "No same-day catalysts were highlighted." not in report
    assert "No forward catalyst list was populated." not in report
