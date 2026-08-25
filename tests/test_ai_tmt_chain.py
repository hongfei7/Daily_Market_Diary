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
    assert chain["hk_followed_overnight"] is None
    assert chain["comparison_posture"] == "pending_next_hk_session"
    assert "predate the US overnight leg" in chain["temporal_note"]


def test_flat_overnight_leg_does_not_set_direction():
    chain = build_ai_tmt_chain(_summary((0.2, -0.1, 0.1), (0.3, 0.2, 0.1, 0.2)))
    assert chain["verdict"] == "neutral"
    assert "does not set the direction" in chain["headline"]


def test_prior_hk_direction_is_not_compared_with_later_overnight_leg():
    chain = build_ai_tmt_chain(_summary((-3.1, -2.4, -1.8), (1.5, 1.4, 1.6, 1.5)))
    assert chain["hk_followed_overnight"] is None
    assert chain["divergence_note"] == ""
    assert "validate transmission" in chain["temporal_note"]


def test_large_prior_hk_move_is_not_attributed_to_later_us_move():
    chain = build_ai_tmt_chain(_summary((-1.2, -1.0, -0.3), (-3.8, -11.8, -3.3, -1.0)))
    assert chain["hk_followed_overnight"] is None
    assert chain["amplification"] is None
    assert chain["comparison_posture"] == "pending_next_hk_session"


def test_single_name_outlier_is_called_company_specific():
    chain = build_ai_tmt_chain(_summary((-1.2, -1.0, -0.3), (-3.8, -11.8, -3.3, -1.0)))
    assert any("Hua Hong" in name for name in chain["single_name_outliers"])


def test_proportional_prior_move_still_waits_for_next_hk_session():
    chain = build_ai_tmt_chain(_summary((-2.0, -2.2, -1.8), (-2.1, -2.3, -1.9, -2.0)))
    assert chain["hk_followed_overnight"] is None
    assert chain["amplification"] is None
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


class TestChainChart:
    """The chart shares its numbers with Section 2.3 rather than recomputing."""

    def _chain(self):
        return build_ai_tmt_chain(_summary((-2.21, -0.99, -0.32), (-3.76, -11.79, -3.31, -1.02)))

    def test_chart_is_written(self, tmp_path):
        from market_diary.professional.ai_tmt_chart import generate_ai_tmt_chain_chart

        out = generate_ai_tmt_chain_chart(self._chain(), str(tmp_path / "chain.png"))
        assert out is not None
        assert (tmp_path / "chain.png").stat().st_size > 10_000

    def test_unavailable_chain_renders_nothing(self, tmp_path):
        from market_diary.professional.ai_tmt_chart import generate_ai_tmt_chain_chart

        assert generate_ai_tmt_chain_chart({"status": "unavailable"}, str(tmp_path / "x.png")) is None
        assert generate_ai_tmt_chain_chart({}, str(tmp_path / "x.png")) is None

    def test_stale_names_are_excluded_from_the_chart(self, tmp_path):
        from market_diary.professional.ai_tmt_chart import _leg_values

        summary = _summary((-2.21, -0.99, -0.32), (-3.76, -11.79, -3.31, -1.02))
        summary["Equities"]["SMIC"] = _item(-3.76, age=5)
        chain = build_ai_tmt_chain(summary)
        plotted = [row["label"] for row in _leg_values(chain["hk_leg"])]
        assert "SMIC" not in plotted, "a stale quote must not be plotted"
