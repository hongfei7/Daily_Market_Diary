"""A stale quote must never drive a conclusion.

On 2026-08-19 the shipped report opened with "HSCEI beat 3033.HK by 3.06pp
(+1.19% versus -1.87%)". The HSCEI leg was current but the 3033.HK leg was four
trading days old, so the spread compared two different dates. The same stale
value also supplied the largest negative component of the composite risk score
and therefore part of the ``Risk-off`` regime label.

Freshness was already measured in ``instruments.annotate_summary_item`` but was
dropped before it reached the analytics layer. These tests pin the gate.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from market_diary.professional.analytics_market import (
    build_hk_investor_lens,
    build_market_snapshot,
)
from market_diary.professional.attribution import build_attribution


def _summary(hstech_age: int):
    """The 2026-08-19 shape: fresh HSI/HSCEI, 3033.HK aged ``hstech_age`` days."""
    return {
        "Equities": {
            "Hang Seng Index": {"Price": 25453.23, "Pct Change": 1.34, "Trading Freshness Days": 1},
            "Hang Seng China Enterprises": {"Price": 9000.0, "Pct Change": 1.19, "Trading Freshness Days": 1},
            "Hang Seng TECH ETF": {"Price": 4.616, "Pct Change": -1.87, "Trading Freshness Days": hstech_age},
            "S&P 500": {"Price": 7691.76, "Pct Change": -0.69, "Trading Freshness Days": 1},
            "China Large-Cap (FXI)": {"Price": 40.0, "Pct Change": -0.11, "Trading Freshness Days": 1},
        },
        "FX": {
            "DXY": {"Price": 99.64, "Pct Change": 0.0, "Trading Freshness Days": 1},
            "USD/CNH": {"Price": 6.735, "Pct Change": 0.05, "Trading Freshness Days": 1},
            "USD/HKD": {"Price": 7.8431, "Pct Change": 0.0, "Trading Freshness Days": 1},
        },
        "Vol": {"VIX": {"Price": 15.84, "Pct Change": 4.28, "Trading Freshness Days": 1}},
    }


def test_snapshot_row_carries_freshness():
    rows = build_market_snapshot(_summary(hstech_age=4))
    hstech = next(row for row in rows if row["short_label"] == "3033.HK ETF")
    assert hstech["freshness_days"] == 4
    assert hstech["is_stale"] is True

    hsi = next(row for row in rows if row["short_label"] == "Hang Seng Index")
    assert hsi["is_stale"] is False


def test_stale_leg_downgrades_the_style_call():
    lens = build_hk_investor_lens(_summary(hstech_age=4), {})
    assert lens["style"] == "unconfirmed"
    assert lens["style_spread_pp"] is None
    # The reader must be told which input failed and how old it was.
    assert "3033.HK ETF" in lens["evidence"]
    assert "4 trading days" in lens["evidence"]
    assert lens["stale_inputs"]


def test_stale_leg_never_reports_a_spread():
    """The specific defect: a 3.06pp spread built from two different dates."""
    lens = build_hk_investor_lens(_summary(hstech_age=4), {})
    assert "3.06pp" not in lens["lens"]
    assert "beat" not in lens["lens"]


def test_fresh_legs_still_produce_a_style_call():
    lens = build_hk_investor_lens(_summary(hstech_age=1), {})
    assert lens["style"] == "value"
    assert lens["style_spread_pp"] is not None
    assert not lens["stale_inputs"]


def test_stale_quote_is_excluded_from_the_risk_score():
    stale = build_attribution(_summary(hstech_age=4), {}, {}, {"theme": ""})
    fresh = build_attribution(_summary(hstech_age=1), {}, {}, {"theme": ""})

    stale_labels = [item["label"] for item in stale["risk_dashboard"]["components"]]
    fresh_labels = [item["label"] for item in fresh["risk_dashboard"]["components"]]

    assert "HK growth ETF proxy" not in stale_labels
    assert "HK growth ETF proxy" in fresh_labels

    # The exclusion is disclosed rather than silent.
    assert stale["risk_dashboard"]["excluded_stale"]
    assert any("3033.HK ETF" in item for item in stale["risk_dashboard"]["excluded_stale"])

    # Removing a stale drag must actually change the score, not be cosmetic.
    assert stale["risk_dashboard"]["score"] != fresh["risk_dashboard"]["score"]


def test_stale_snapshot_line_is_labelled_for_the_reader():
    lens = build_hk_investor_lens(_summary(hstech_age=4), {})
    assert "stale 4d" in lens["lines"][0]
    assert "-1.87%" not in lens["lines"][0]
