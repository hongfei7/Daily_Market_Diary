"""AI / TMT read-through from the overnight semis leg into Hong Kong tech.

For a desk covering AI and TMT this is the primary daily transmission, and none
of it was tracked: no SOXX, TSMC, NVDA, SMIC, Hua Hong or Sunny Optical.

The chain must also refuse to overclaim. Matching direction is not the same as
being explained: on 2026-08-19 Hong Kong tech fell 4.2x harder than the
overnight leg, with Hua Hong down 11.79%, which is company-specific rather than
a cycle read.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from market_diary.professional.ai_tmt_chain import build_ai_tmt_chain


def _item(pct, age=1):
    return {"Price": 100.0, "Pct Change": pct, "Trading Freshness Days": age,
            "Change Value": pct, "Change Unit": "pct"}


def _summary(overnight, hk):
    soxx, nvda, tsm = overnight
    smic, huahong, sunny, etf = hk
    return {
        "Equities": {
            "Semiconductors (SOXX)": _item(soxx),
            "NVIDIA": _item(nvda),
            "TSMC ADR": _item(tsm),
            "SMIC": _item(smic),
            "Hua Hong Semiconductor": _item(huahong),
            "Sunny Optical": _item(sunny),
            "Hang Seng TECH ETF": _item(etf),
        }
    }


def test_risk_off_overnight_leg_sets_the_hk_expression():
    chain = build_ai_tmt_chain(_summary((-3.1, -2.4, -1.8), (-2.1, -1.5, -0.9, -1.0)))
    assert chain["verdict"] == "risk-off"
    assert "SMIC" in chain["expression"]
    assert chain["overnight_avg_pct"] < 0
    assert chain["hk_followed_overnight"] is True


def test_flat_overnight_leg_does_not_set_direction():
    chain = build_ai_tmt_chain(_summary((0.2, -0.1, 0.1), (0.3, 0.2, 0.1, 0.2)))
    assert chain["verdict"] == "neutral"
    assert "does not set the direction" in chain["headline"]


def test_opposite_direction_is_flagged_as_divergence():
    chain = build_ai_tmt_chain(_summary((-3.1, -2.4, -1.8), (1.5, 1.4, 1.6, 1.5)))
    assert chain["hk_followed_overnight"] is False
    assert "diverged" in chain["divergence_note"]


def test_same_direction_but_amplified_is_not_called_coherent():
    """The 2026-08-19 case: same sign, 4x the magnitude."""
    chain = build_ai_tmt_chain(_summary((-1.2, -1.0, -0.3), (-3.8, -11.8, -3.3, -1.0)))
    assert chain["hk_followed_overnight"] is True
    assert chain["amplification"] >= 2.0
    assert "does not explain a move that size" in chain["divergence_note"]


def test_single_name_outlier_is_called_company_specific():
    chain = build_ai_tmt_chain(_summary((-1.2, -1.0, -0.3), (-3.8, -11.8, -3.3, -1.0)))
    assert any("Hua Hong" in name for name in chain["single_name_outliers"])


def test_proportional_move_is_reported_as_coherent():
    chain = build_ai_tmt_chain(_summary((-2.0, -2.2, -1.8), (-2.1, -2.3, -1.9, -2.0)))
    assert chain["hk_followed_overnight"] is True
    assert chain["amplification"] < 2.0
    assert chain["divergence_note"] == ""
    assert chain["single_name_outliers"] == []


def test_stale_names_are_excluded_and_disclosed():
    summary = _summary((-3.1, -2.4, -1.8), (-2.1, -1.5, -0.9, -1.0))
    summary["Equities"]["SMIC"] = _item(-2.1, age=5)
    chain = build_ai_tmt_chain(summary)
    assert any("SMIC" in note for note in chain["stale_inputs"])
    smic = next(row for row in chain["hk_leg"] if row["label"] == "SMIC")
    assert smic["available"] is False
    assert "stale" in smic["display"]


def test_missing_overnight_coverage_degrades_honestly():
    chain = build_ai_tmt_chain({"Equities": {}})
    assert chain["status"] == "unavailable"
    assert "unavailable" in chain["headline"]
