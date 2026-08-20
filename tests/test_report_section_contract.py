"""The required-section contract must be checked against a real rendered report.

CI rejected a perfectly good report because the contract in ``runtime_audit``
hard-coded section numbers:

    ("Flow tracker", ("### 2.3 Flow Tracker and Attribution",))

Inserting ``2.3 AI / TMT Read-Through`` shifted Flow Tracker to 2.4, so the
section was present but reported missing, the audit exited non-zero, and no
briefing was published that day.

The whole suite stayed green because ``tests/test_runtime_audit.py`` feeds the
audit a hand-written ``REPORT_BODY`` constant that still said 2.3. The tests
validated a fixture, not the artefact — the same failure mode this project has
been fixing elsewhere.

These tests run the contract against an actually rendered report, so any
renumbering or retitling surfaces immediately.
"""

from __future__ import annotations

import re

import _bootstrap  # noqa: F401
import pytest

from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.report_builder import render_professional_report
from professional.runtime_audit import (
    REQUIRED_REPORT_SECTION_GROUPS,
    _section_present,
)


def _item(price, pct):
    return {
        "Price": price,
        "Pct Change": pct,
        "Trading Freshness Days": 1,
        "As Of": "2026-08-18",
        "Change Value": pct,
        "Change Unit": "pct",
        "Quality": "fresh",
    }


@pytest.fixture(scope="module")
def rendered_report() -> str:
    summary = {
        "Equities": {
            "S&P 500": _item(7691.76, -0.69),
            "Nasdaq 100": _item(29490.96, -1.68),
            "Semiconductors (SOXX)": _item(280.0, -2.21),
            "Hang Seng Index": _item(25453.23, 1.34),
            "Hang Seng China Enterprises": _item(9000.0, 1.19),
            "Hang Seng TECH ETF": _item(4.65, 0.50),
            "SMIC": _item(45.0, -3.76),
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
    # The chart sections only render when the assets exist, and the contract
    # requires them, so give the renderer the paths a real run would supply.
    bundle["daily_one_chart"] = {"title": "Daily One Chart", "chart_read": "read", "why_it_matters": "why"}
    return render_professional_report(
        bundle,
        charts_section="",
        dashboard_rel_path="charts/dashboard.png",
        catalyst_radar_rel_path="charts/catalyst_radar.png",
        daily_chart_rel_path="charts/daily_one_chart.png",
    )


def test_rendered_report_satisfies_every_required_section(rendered_report):
    missing = [
        label
        for label, markers in REQUIRED_REPORT_SECTION_GROUPS
        if not any(_section_present(rendered_report, marker) for marker in markers)
    ]
    assert not missing, f"the rendered report is missing required sections: {missing}"


def test_contract_carries_no_hard_coded_section_numbers():
    """A numbered marker breaks the next time a section is inserted."""
    offenders = [
        (label, marker)
        for label, markers in REQUIRED_REPORT_SECTION_GROUPS
        for marker in markers
        if re.search(r"\d+\.\d+", marker)
    ]
    assert not offenders, f"required-section markers must not embed numbering: {offenders}"


class TestSectionMatching:
    TITLE = "Flow Tracker and Attribution"

    @pytest.mark.parametrize("heading", [
        "### 2.3 Flow Tracker and Attribution",
        "### 2.4 Flow Tracker and Attribution",
        "### 9.9 Flow Tracker and Attribution",
        "## Flow Tracker and Attribution",
    ])
    def test_any_numbering_satisfies_the_contract(self, heading):
        assert _section_present(heading, self.TITLE)

    def test_prose_mention_does_not_satisfy_the_contract(self):
        assert not _section_present("See the Flow Tracker and Attribution section below.", self.TITLE)

    def test_a_genuinely_absent_section_is_still_caught(self):
        assert not _section_present("### 2.4 Something Else Entirely", self.TITLE)

    def test_markers_with_their_own_markup_match_literally(self):
        assert _section_present("## Visual Dashboard\n", "## Visual Dashboard")
        assert _section_present("**AH Premium Dispersion**", "**AH Premium Dispersion**")
        assert _section_present("![Catalyst & Event Radar](x.png)", "![Catalyst & Event Radar]")
