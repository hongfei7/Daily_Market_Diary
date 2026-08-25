"""Anticipating the questions a senior will ask.

The signals needed were already in the bundle — divergences, tail readings,
coverage gaps, a call that just failed — but nothing assembled them into the
form a morning meeting takes.

Generated deterministically: the narrative overlay runs at 0-2 of 7 successful
tasks, so meeting preparation must not depend on it.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from market_diary.professional.md_questions import MAX_QUESTIONS, build_md_questions


def _item(pct, age=1):
    return {"Price": 100.0, "Pct Change": pct, "Trading Freshness Days": age,
            "Change Value": pct, "Change Unit": "pct"}


def _bundle(hsi=0.09, tech=-1.02, southbound=None, **extra):
    bundle = {
        "meta": {"hk_data_date": "2026-08-19"},
        "market_summary": {
            "Equities": {"Hang Seng Index": _item(hsi), "Hang Seng TECH ETF": _item(tech)},
        },
        "hk_local": {},
        "hk_desk_view": {},
        "call_scorecard": {},
        "ai_tmt_chain": {},
        "source_health": {},
    }
    if southbound is not None:
        bundle["hk_local"]["southbound_net_flow"] = {"value": southbound}
    bundle.update(extra)
    return bundle


def test_style_divergence_becomes_a_question():
    questions = build_md_questions(_bundle(hsi=0.09, tech=-1.02))
    assert any("what drove the split" in q["question"] for q in questions)
    text = next(q for q in questions if "split" in q["question"])
    assert "1.11pp" in text["evidence"]
    assert text["answer"]


def test_no_divergence_produces_no_style_question():
    questions = build_md_questions(_bundle(hsi=0.5, tech=0.6))
    assert not any("split" in q["question"] for q in questions)


def test_price_and_flow_disagreement_is_raised():
    questions = build_md_questions(_bundle(hsi=1.2, tech=1.1, southbound=-10_600_000_000))
    assert any("is the move real" in q["question"] for q in questions)
    item = next(q for q in questions if "is the move real" in q["question"])
    assert "-10.6bn" in item["evidence"]
    assert "unconfirmed" in item["answer"]
    assert "price move was not" in item["answer"]
    assert "fele" not in item["answer"]


def test_agreeing_price_and_flow_raises_nothing():
    questions = build_md_questions(_bundle(hsi=1.2, tech=1.1, southbound=10_600_000_000))
    assert not any("is the move real" in q["question"] for q in questions)


def test_stale_input_explains_a_missing_call():
    bundle = _bundle(hk_desk_view={"stale_inputs": ["3033.HK ETF was stale (4 trading days old)"]})
    questions = build_md_questions(bundle)
    top = questions[0]
    assert "no style call" in top["question"]
    assert "stale" in top["evidence"]


def test_broken_call_is_raised_and_ranks_high():
    bundle = _bundle(call_scorecard={"verdict": "BROKEN", "headline": "Risk-Off was published but HSI rose."})
    questions = build_md_questions(bundle)
    assert questions[0]["question"].startswith("Yesterday's call did not work")
    assert "Lead with the miss" in questions[0]["answer"]


def test_confirmed_call_raises_no_question():
    bundle = _bundle(call_scorecard={"verdict": "CONFIRMED", "headline": "worked"})
    assert not any("did not work" in q["question"] for q in build_md_questions(bundle))


def test_amplified_ai_tmt_move_is_raised():
    bundle = _bundle(ai_tmt_chain={"divergence_note": "HK tech moved 4.2x harder than the overnight leg."})
    questions = build_md_questions(bundle)
    assert any("overnight semis" in q["question"] for q in questions)


def test_output_is_capped_and_ordered_by_priority():
    bundle = _bundle(
        hsi=1.2,
        tech=-1.5,
        southbound=-10_600_000_000,
        hk_desk_view={"stale_inputs": ["3033.HK ETF was stale (4 trading days old)"]},
        call_scorecard={"verdict": "BROKEN", "headline": "missed"},
        ai_tmt_chain={"divergence_note": "amplified"},
    )
    questions = build_md_questions(bundle)
    assert len(questions) == MAX_QUESTIONS
    priorities = [q["priority"] for q in questions]
    assert priorities == sorted(priorities, reverse=True)
    # The coverage gap outranks everything: a missing conclusion is asked about first.
    assert "no style call" in questions[0]["question"]


def test_quiet_session_produces_nothing_rather_than_filler():
    assert build_md_questions(_bundle(hsi=0.2, tech=0.25)) == []


def test_every_question_carries_evidence_and_an_answer():
    bundle = _bundle(hsi=1.2, tech=-1.5, southbound=-10_600_000_000)
    for item in build_md_questions(bundle):
        assert item["question"] and item["evidence"] and item["answer"]
