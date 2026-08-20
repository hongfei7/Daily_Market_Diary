"""Direction words carry the sign in prose; the checker must read them.

"the Nasdaq 100 fell 1.68%" is correct English for a -1.68% move. The checker
captured the magnitude and discarded the verb, compared +1.68 against -1.68 and
raised a critical mismatch, which blocked automatic distribution. The guard was
penalising correct writing while still needing to catch genuinely wrong numbers.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import pytest

from market_diary.professional.fact_checker import _claim_mismatches, _claim_patterns, _signed_claim

BUNDLE = {
    "market_summary": {
        "Equities": {
            "Nasdaq 100": {"Price": 29490.96, "Pct Change": -1.68, "Change Value": -1.68, "Change Unit": "pct"},
            "S&P 500": {"Price": 7691.76, "Pct Change": -0.69, "Change Value": -0.69, "Change Unit": "pct"},
        },
        "Vol": {"VIX": {"Price": 15.84, "Pct Change": 4.28, "Change Value": 4.28, "Change Unit": "pct"}},
    }
}


def _resolve(alias: str, text: str):
    for kind, pattern in _claim_patterns(alias):
        if kind != "change_pct":
            continue
        match = pattern.search(text)
        if match:
            value = _signed_claim(match, text)
            if value is not None:
                return value
    return None


@pytest.mark.parametrize(
    "alias,text,expected",
    [
        ("Nasdaq 100", "the Nasdaq 100 fell 1.68% overnight", -1.68),
        ("Nasdaq 100", "the Nasdaq 100 rose 1.68% overnight", 1.68),
        ("Nasdaq 100", "Nasdaq 100 dropped 1.68%", -1.68),
        ("VIX", "VIX climbed 4.28%", 4.28),
        ("VIX", "VIX slipped 4.28%", -4.28),
        # An explicit sign always wins over any surrounding word.
        ("Nasdaq 100", "Nasdaq 100 -1.68%", -1.68),
        ("Nasdaq 100", "Nasdaq 100 +1.68%", 1.68),
        # Direction carried by a trailing noun across a possessive.
        ("S&P 500", "the S&P 500's 0.69% decline weighed on beta", -0.69),
        ("S&P 500", "the S&P 500 posted a 0.69% gain", 0.69),
    ],
)
def test_direction_words_set_the_sign(alias, text, expected):
    assert _resolve(alias, text) == expected


def test_correct_prose_does_not_block_release():
    """The exact sentence that blocked the 2026-08-19 report."""
    text = (
        "Overnight, the US posted a risk-off session driven by growth-style transmission rather than "
        "rates: the Nasdaq 100 fell 1.68% versus the S&P 500's 0.69% decline."
    )
    checked, mismatches = _claim_mismatches(BUNDLE, [("deep_read_setup", text)])
    assert checked >= 2, "both claims should still be inspected"
    assert mismatches == []


def test_wrong_magnitude_is_still_caught():
    checked, mismatches = _claim_mismatches(BUNDLE, [("x", "the Nasdaq 100 fell 5.20% overnight")])
    assert checked == 1
    assert len(mismatches) == 1
    assert mismatches[0]["claimed"] == -5.2
    assert mismatches[0]["expected"] == -1.68


def test_wrong_direction_is_still_caught():
    """Narrative claiming a rise when the index fell must remain a mismatch."""
    checked, mismatches = _claim_mismatches(BUNDLE, [("x", "the Nasdaq 100 rose 1.68% overnight")])
    assert checked == 1
    assert len(mismatches) == 1
    assert mismatches[0]["claimed"] == 1.68
    assert mismatches[0]["expected"] == -1.68
