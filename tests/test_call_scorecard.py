"""Scoring the previously published call.

The ledger has recorded a directional call for every report and scored them in
aggregate as appendix hit rates, but the report never answered the question a
desk actually asks first: we said X yesterday, did it happen?

A failed call must be reported as BROKEN with the size of the miss. Softening it
into ambiguity, or counting an unscoreable call as a win, would make the
retrospective worthless.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import pytest

from market_diary.professional.call_scorecard import (
    build_call_scorecard,
    normalize_signal,
    recent_record,
)

HSI = "Hang Seng Index"
TECH = "Hang Seng TECH ETF (3033.HK)"


def _ledger(signal="Risk-On", position=1, entry=25000.0, exit_price=25500.0, with_hsi=True):
    prices_entry = {TECH: 4.60}
    prices_exit = {TECH: 4.65}
    if with_hsi:
        prices_entry[HSI] = entry
        prices_exit[HSI] = exit_price
    return {
        "signals": [
            {
                "report_date": "2026-08-19",
                "market_as_of": "2026-08-18",
                "signal": signal,
                "position": position,
                "evidence": {},
            }
        ],
        "observations": [
            {"as_of": "2026-08-18", "prices": prices_entry},
            {"as_of": "2026-08-19", "prices": prices_exit},
        ],
    }


@pytest.mark.parametrize(
    "raw,expected",
    [("Risk-off", "Risk-Off"), ("Risk-On", "Risk-On"), ("Risk-on", "Risk-On"),
     ("Mixed", "Mixed"), ("blocked", "blocked")],
)
def test_inconsistent_ledger_labels_are_normalised(raw, expected):
    """The ledger holds three spellings of two states."""
    assert normalize_signal(raw) == expected


def test_correct_call_is_confirmed():
    card = build_call_scorecard(_ledger(position=1, exit_price=25500.0), "2026-08-20")
    assert card["verdict"] == "CONFIRMED"
    assert "in the called direction" in card["headline"]


def test_wrong_call_is_reported_as_broken_with_the_miss():
    card = build_call_scorecard(_ledger(signal="Risk-Off", position=-1, exit_price=25500.0), "2026-08-20")
    assert card["verdict"] == "BROKEN"
    assert "+2.00%" in card["headline"]
    assert "against the call" in card["headline"]


def test_move_inside_the_noise_band_is_unresolved_not_a_win():
    card = build_call_scorecard(_ledger(position=1, exit_price=25025.0), "2026-08-20")
    assert card["verdict"] == "UNRESOLVED"
    assert "noise band" in card["headline"]


def test_blocked_release_is_no_call_rather_than_a_result():
    card = build_call_scorecard(_ledger(signal="blocked", position=0), "2026-08-20")
    assert card["verdict"] == "NO CALL"
    assert "blocked" in card["headline"]


def test_falls_back_to_the_growth_proxy_and_says_so():
    """A conflicting price is dropped upstream, so HSI can be missing."""
    card = build_call_scorecard(_ledger(position=-1, with_hsi=False), "2026-08-20")
    assert card["scored_on"] == TECH
    assert "no comparable Hang Seng Index close" in card["headline"]
    assert card["verdict"] == "BROKEN"


def test_today_cannot_score_its_own_call():
    ledger = _ledger()
    ledger["signals"].append(
        {"report_date": "2026-08-20", "market_as_of": "2026-08-19", "signal": "Risk-On", "position": 1}
    )
    card = build_call_scorecard(ledger, "2026-08-20")
    assert card["report_date"] == "2026-08-19"


def test_no_prior_call_degrades_honestly():
    card = build_call_scorecard({"signals": [], "observations": []}, "2026-08-20")
    assert card["status"] == "unavailable"
    assert card["verdict"] == "UNRESOLVED"


class TestRecentRecord:
    def _many(self):
        signals, observations = [], []
        # Alternate a correct and an incorrect call across ten sessions.
        for idx in range(10):
            entry_date = f"2026-07-{idx + 1:02d}"
            exit_date = f"2026-07-{idx + 2:02d}"
            correct = idx % 2 == 0
            signals.append(
                {
                    "report_date": f"2026-07-{idx + 2:02d}",
                    "market_as_of": entry_date,
                    "signal": "Risk-On",
                    "position": 1,
                }
            )
            observations.append({"as_of": entry_date, "prices": {HSI: 100.0}})
            observations.append({"as_of": exit_date, "prices": {HSI: 102.0 if correct else 98.0}})
        return {"signals": signals, "observations": observations}

    def test_counts_confirmed_and_broken(self):
        record = recent_record(self._many(), "2026-08-01")
        assert record["scored"] > 0
        assert record["confirmed"] + record["broken"] == record["scored"]
        assert 0 <= record["hit_rate_pct"] <= 100

    def test_empty_ledger_reports_nothing_scored(self):
        record = recent_record({"signals": [], "observations": []}, "2026-08-01")
        assert record["scored"] == 0
        assert record["hit_rate_pct"] is None
