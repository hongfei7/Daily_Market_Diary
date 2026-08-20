from _bootstrap import ROOT  # noqa: F401

from professional.analytics_market import build_hk_desk_view
from professional.report_blocks import _render_executive_summary, _render_hk_review_block


def _market_summary() -> dict:
    return {
        "Equities": {
            "Hang Seng Index": {"Price": 26009.4, "Pct Change": "0.48%"},
            "Hang Seng China Enterprises": {"Price": 9250.0, "Pct Change": "0.46%"},
            "Hang Seng TECH ETF": {"Price": 4.78, "Pct Change": "1.06%"},
            "China Large-Cap (FXI)": {"Price": 38.0, "Pct Change": "-0.41%"},
        },
        "FX": {
            "USD/CNH": {"Price": 6.7285, "Pct Change": "0.00%"},
            "USD/HKD": {"Price": 7.8423, "Pct Change": "0.00%"},
        },
    }


def test_hk_lens_explains_style_evidence_and_conviction() -> None:
    hk_local = {
        "turnover_vs_20d": {"value": 0.86},
        "short_selling_ratio": {"value": 18.0},
        "southbound_net_flow": {"value": 2_600_000_000},
    }

    lens = build_hk_desk_view(_market_summary(), hk_local)

    assert lens["headline"] == "Selective growth leadership"
    assert "beat HSCEI by +0.60pp" in lens["evidence"]
    assert "turnover was only 0.86x" in lens["lens"]
    # Without trailing history the claim must be framed as an absolute
    # reference rather than asserting the reading is unusual.
    assert "short selling was 18.0%, above the 16% absolute reference" in lens["lens"]
    assert "elevated" not in lens["lens"]
    assert "Southbound recorded +HK$2.6bn net buying" in lens["lens"]
    assert "do not treat the move as broad China risk-on" in lens["implication"]
    assert "3033.HK keeps outperforming HSCEI" in lens["confirmation"]
    assert "3033.HK loses its lead" in lens["invalidation"]


def test_report_renders_full_lens_even_when_llm_returns_generic_label() -> None:
    hk_local = {
        "turnover_vs_20d": {"value": 0.86},
        "short_selling_ratio": {"value": 18.0},
        "southbound_net_flow": {"value": 2_600_000_000},
    }
    hk_desk_view = build_hk_desk_view(_market_summary(), hk_local)
    bundle = {
        "hk_desk_view": hk_desk_view,
        "llm_sections": {
            "hk_local_leadership": "Hong Kong growth / internet led.",
            "hk_follow_through": "Watch Hong Kong growth leadership.",
        },
        "flow_tracker": {},
        "today_forward": {},
        "must_watch": [],
    }

    summary = _render_executive_summary(bundle, "Overnight risk appetite improved.")
    review = _render_hk_review_block(bundle)

    # The lens carries an explicit confidence grade so a reader can tell a
    # well-evidenced call from one the pipeline could not verify.
    assert "**Hong Kong lens** (medium confidence): Selective growth leadership:" in summary
    assert "turnover was only 0.86x" in summary
    assert "**What would confirm it:** Confirm if 3033.HK" in summary
    assert "**What could break it:** Invalidate if 3033.HK" in summary
    assert "**Style call.** Selective growth leadership." in review
    assert "**Portfolio meaning:**" in review
    assert "**Failure condition.**" in review


def test_hk_lens_does_not_invent_style_when_relative_data_is_missing() -> None:
    lens = build_hk_desk_view(
        {"Equities": {"Hang Seng Index": {"Price": 26000.0, "Pct Change": "0.40%"}}},
        {},
    )

    assert lens["style"] == "unconfirmed"
    assert lens["headline"] == "Leadership unconfirmed — coverage is insufficient"
    assert "not available" in lens["evidence"]
    assert "Do not infer Hong Kong style" in lens["implication"]


if __name__ == "__main__":
    test_hk_lens_explains_style_evidence_and_conviction()
    test_report_renders_full_lens_even_when_llm_returns_generic_label()
    test_hk_lens_does_not_invent_style_when_relative_data_is_missing()
    print("Hong Kong investor lens test passed")
