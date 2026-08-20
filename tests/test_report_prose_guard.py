"""Guard the rendered report against prose defects.

The rest of the suite checks that sections and tables exist. Nothing checked
that the resulting English was publishable, which is how sentence fragments,
unbalanced brackets and internal identifiers reached the executive summary of
shipped reports. These tests close that gap.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

from market_diary.professional.prose_guard import check_markdown, summarize
from market_diary.professional.report_text import _condense_sentence


def test_detects_sentence_fragment():
    markdown = "- **Key driver:** US growth transmission was flagged as the dominant.\n"
    findings = check_markdown(markdown)
    assert any(item["rule"] == "sentence_fragment" for item in findings)


def test_detects_dangling_determiner():
    markdown = "USD stayed range-bound, keeping the funding lens benign and removing the.\n"
    findings = check_markdown(markdown)
    assert any(item["rule"] == "sentence_fragment" for item in findings)


def test_stranded_preposition_after_a_verb_is_allowed():
    """'the catalyst to prepare for.' is grammatical, not a truncation."""
    markdown = "- **Corporate / event:** China credit data is the cleanest catalyst to prepare for.\n"
    assert check_markdown(markdown) == []


def test_detects_severed_enumeration():
    markdown = "The sell-off warns that growth, platform.\n"
    findings = check_markdown(markdown)
    assert any(item["rule"] == "severed_list" for item in findings)


def test_detects_unbalanced_bracket():
    markdown = "Oil outperformed Gold (intraday spread ~3.16pp.\n"
    findings = check_markdown(markdown)
    assert any(item["rule"] == "unbalanced_bracket" for item in findings)


def test_detects_internal_identifier_leak():
    markdown = "The read is consistent with the attribution_v1 conclusion that rates did not lead.\n"
    findings = check_markdown(markdown)
    assert any(item["rule"] == "internal_identifier" for item in findings)


def test_hyphenated_compound_is_not_a_fragment():
    """'Risk-On.' ends a sentence legitimately and must not be flagged."""
    markdown = "The overnight tape reads closer to Risk-On.\n"
    assert check_markdown(markdown) == []


def test_diagnostic_field_list_is_allowed():
    """The audit appendix quotes internal field paths on purpose."""
    markdown = "- **Deterministic fallback fields:** tasks.overnight_review.paragraph\n"
    assert not any(item["rule"] == "internal_identifier" for item in check_markdown(markdown))


def test_structural_lines_are_ignored():
    markdown = "\n".join(
        [
            "| Signal | Last | Interpretation |",
            "| --- | --- | --- |",
            "| VIX | 15.84 | Higher volatility raises the |",
            "![Research Dashboard](charts/dashboard.png)",
            "## Layer 1 | Scan",
        ]
    )
    assert check_markdown(markdown) == []


def test_clean_prose_produces_no_findings():
    markdown = (
        "- **Market pulse:** The overnight tape was risk-off, with volatility higher and cyclicals weaker.\n"
        "- **Hong Kong lens:** Leadership is unconfirmed because a required input was stale.\n"
    )
    assert check_markdown(markdown) == []


def test_summarize_penalises_findings():
    clean = summarize([])
    assert clean["status"] == "ok"
    assert clean["score"] == 100.0

    dirty = summarize([{"rule": "sentence_fragment"}, {"rule": "severed_list"}])
    assert dirty["status"] == "warning"
    assert dirty["score"] < clean["score"]
    assert dirty["total"] == 2


class TestCondenseSentenceNeverEmitsFragments:
    """``_condense_sentence`` was the source of the shipped fragments."""

    CASES = [
        # Cutting at ", and " here severed "growth, platform" from the list.
        "For the HK open, the sell-off is the cleanest warning that growth, platform, and consumer-internet"
        " names will be tested first, so the right lean is to keep risk small until Southbound confirms.",
        # Cutting at width left "flagged as the dominant."
        "Growth-style drag led the US tape: Nasdaq 100 -1.68% versus S&P 500 -0.69%, with US growth and"
        " internet transmission flagged as the dominant channel into Hong Kong for the coming session.",
        # Cutting at width left an unbalanced parenthesis.
        "Oil outperformed Gold (intraday spread ~3.16pp) as the geopolitical bid persisted into the close"
        " and left the commodity complex the clearest signal of the overnight session for local desks.",
    ]

    def test_output_is_never_a_fragment(self):
        for sentence in self.CASES:
            for width in (90, 120, 150, 190):
                out = _condense_sentence(sentence, width)
                findings = check_markdown(out)
                assert findings == [], f"width={width} produced {findings} for: {out}"

    def test_shortens_at_a_real_clause_boundary(self):
        sentence = (
            "Hong Kong broad beta strengthened on the session and turnover held near its twenty-day"
            " baseline, which suggests the move had at least partial local participation behind it."
        )
        out = _condense_sentence(sentence, 120)
        assert len(out) < len(sentence)
        assert out.endswith(".")
        assert check_markdown(out) == []

    def test_keeps_sentence_whole_when_no_safe_cut_exists(self):
        sentence = (
            "The overnight tape was risk-off across US equities and credit-sensitive assets with no"
            " clause boundary available anywhere inside the configured width budget for this line."
        )
        assert _condense_sentence(sentence, 80) == sentence
