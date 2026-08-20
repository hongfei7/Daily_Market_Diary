"""The executive summary answers four fixed questions, one sentence each.

The previous format put the style call, its supporting evidence, a conviction
caveat, a partial-support clause and the portfolio implication into a single
70-word bullet, and spliced clause-shaped flags after prepositions that need a
noun phrase, producing "Partial support came from FXI (+1.77%) outperformed
HSI (+0.09%)."

Fixed questions in a fixed order can be scanned in seconds and compared across
days; a free-form paragraph cannot.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from market_diary.professional.prose_guard import check_markdown
from market_diary.professional.report_blocks import _render_executive_summary

QUESTIONS = [
    "Did yesterday's call work?",
    "What changed overnight?",
    "What it means for AI/TMT",
    "What to watch today",
]


def _item(pct, age=1):
    return {"Price": 100.0, "Pct Change": pct, "Trading Freshness Days": age,
            "Change Value": pct, "Change Unit": "pct"}


def _bundle(**overrides):
    bundle = {
        "market_summary": {
            "Equities": {
                "Hang Seng Index": _item(0.09),
                "Hang Seng TECH ETF": _item(-1.02),
                "Semiconductors (SOXX)": _item(-2.21),
                "NVIDIA": _item(-0.99),
                "TSMC ADR": _item(-0.32),
                "SMIC": _item(-3.76),
            }
        },
        "call_scorecard": {"verdict": "BROKEN", "headline": "Risk-Off was published but the tape went the other way."},
        "ai_tmt_chain": {
            "overnight_leg": [
                {"label": "SOXX", "display": "-2.21%", "available": True, "change_pct": -2.21},
                {"label": "NVDA", "display": "-0.99%", "available": True, "change_pct": -0.99},
            ],
            "overnight_avg_pct": -1.6,
            "test": "Confirm if SMIC holds direction through the first hour.",
        },
        "hk_desk_view": {
            "headline": "Selective old-economy / H-share leadership",
            "style_spread_pp": -1.23,
            "implication": "Treat banks and SOE yield as the cleaner expression.",
        },
        "macro_agenda": [{"event": "China LPR (1Y / 5Y)", "status": "Upcoming", "date": "2026-08-20"}],
    }
    bundle.update(overrides)
    return bundle


def test_all_four_questions_are_present_and_ordered():
    summary = _render_executive_summary(_bundle(), "unused pulse")
    positions = [summary.find(f"**{q}**") for q in QUESTIONS]
    assert all(pos >= 0 for pos in positions), summary
    assert positions == sorted(positions), "questions must keep a stable order"


def test_each_answer_is_one_line():
    summary = _render_executive_summary(_bundle(), "")
    lines = [line for line in summary.splitlines() if line.strip()]
    assert len(lines) == len(QUESTIONS)


def test_no_answer_sprawls_into_a_paragraph():
    """The old format packed five topics into one 70-word bullet."""
    for line in _render_executive_summary(_bundle(), "").splitlines():
        assert len(line.split()) <= 60, f"answer is too long: {line}"


def test_summary_is_free_of_prose_defects():
    assert check_markdown(_render_executive_summary(_bundle(), "")) == []


def test_h_share_keeps_its_casing():
    """An earlier draft lowercased the headline and produced 'h-share'."""
    summary = _render_executive_summary(_bundle(), "")
    assert "h-share" not in summary or "H-share" in summary


def test_overnight_answer_leads_with_semis_not_crypto():
    """Bitcoin and Gold are the assets the transmission weighting demotes."""
    summary = _render_executive_summary(_bundle(), "")
    overnight = next(line for line in summary.splitlines() if "What changed overnight?" in line)
    assert "SOXX" in overnight
    assert "Bitcoin" not in overnight


def test_stale_input_replaces_the_style_call():
    bundle = _bundle(hk_desk_view={"stale_inputs": ["3033.HK ETF was stale (4 trading days old)"]})
    summary = _render_executive_summary(bundle, "")
    ai_line = next(line for line in summary.splitlines() if "What it means for AI/TMT" in line)
    assert "withheld" in ai_line
    assert "stale" in ai_line


def test_broken_call_is_surfaced_first():
    summary = _render_executive_summary(_bundle(), "")
    first = summary.splitlines()[0]
    assert "Did yesterday's call work?" in first
    assert "BROKEN" in first


def test_empty_bundle_still_answers_every_question():
    summary = _render_executive_summary({}, "")
    for question in QUESTIONS:
        assert f"**{question}**" in summary
    assert check_markdown(summary) == []
