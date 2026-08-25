import sys
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from professional.fact_checker import run_fact_check


def _bundle(text: str, *, risk_regime: str = "Risk-Off", us10y_change_bp: float = 9.9):
    return {
        "overview": {"risk_regime": risk_regime},
        "market_summary": {
            "Equities": {
                "S&P 500": {"Pct Change": "-0.24%", "Price": 6280},
                "Nasdaq 100": {"Pct Change": "-0.42%", "Price": 22800},
                "Hang Seng Index": {"Pct Change": "0.77%", "Price": 26000},
                "Hang Seng TECH ETF": {"Pct Change": "0.28%", "Price": 6.2},
                "China Large-Cap (FXI)": {"Pct Change": "-1.46%", "Price": 35.0},
            },
            "Rates": {
                "10Y Treasury": {
                    "Price": 4.292,
                    "Change Unit": "bp",
                    "Change Value": us10y_change_bp,
                    "Change Display": f"{us10y_change_bp:+.1f}bp",
                }
            },
            "FX": {"DXY": {"Pct Change": "0.37%", "Price": 100.5}, "USD/CNH": {"Pct Change": "0.00%", "Price": 7.12}},
            "Commodities": {"Brent Crude": {"Pct Change": "0.89%", "Price": 85.0}, "Gold": {"Pct Change": "-1.17%", "Price": 3300}},
            "Vol": {"VIX": {"Pct Change": "3.34%", "Price": 19.5}},
        },
        "hk_local": {
            "short_selling_ratio": {"value": 15.0, "display_value": "15.00%"},
            "ah_premium_index": {"value": 29.03, "display_value": "29.03%"},
        },
        "china_rates": {
            "china_10y": {"display_value": "1.79%", "change_display": "-1.2bp"},
        },
        "llm_sections": {"overnight_review": text},
    }


def test_us10y_change_and_level_are_not_confused():
    result = run_fact_check(_bundle("US 10Y yield rose +9.9bp to 4.292%, pressuring duration assets."))
    assert result["status"] == "ok"
    assert result["numeric_mismatches"] == []
    assert result["numeric_claims_checked"] >= 1


def test_us10y_wrong_change_is_flagged_as_critical():
    result = run_fact_check(_bundle("US 10Y yield rose +18.0bp to 4.292%, pressuring duration assets."))
    assert result["status"] == "warning"
    mismatch = result["numeric_mismatches"][0]
    assert mismatch["label"] == "US 10Y"
    assert mismatch["claim_type"] == "change_bp"
    assert mismatch["severity"] == "critical"
    assert result["release_blocking"] is True


def test_risk_on_pockets_do_not_conflict_with_risk_off_regime():
    result = run_fact_check(_bundle("Risk-Off backdrop remained intact, but there were risk-on pockets in energy and defensives."))
    assert result["status"] == "ok"
    assert result["logic_warnings"] == []


def test_explicit_regime_conflict_is_review_warning():
    result = run_fact_check(_bundle("Risk-on backdrop dominated the session despite weaker equities."))
    assert result["status"] == "warning"
    warning = result["logic_warnings"][0]
    assert warning["type"] == "risk_regime"
    assert warning["severity"] == "review"


def test_conditional_yield_watchpoint_is_not_a_logic_warning():
    result = run_fact_check(
        _bundle(
            "Rates impulse was supportive because US 10Y declined. "
            "Macro watchpoints: higher yields would pressure duration-sensitive HK sectors."
        )
    )
    assert result["status"] == "ok"
    assert result["logic_warnings"] == []


def test_unhedged_yield_direction_conflict_is_review_warning():
    result = run_fact_check(
        _bundle(
            "Higher yields drove the session and pressured growth.",
            risk_regime="Risk-On",
            us10y_change_bp=-9.9,
        )
    )
    assert result["status"] == "warning"
    warning = result["logic_warnings"][0]
    assert warning["type"] == "rates_logic"
    assert warning["severity"] == "review"
    assert result["release_blocking"] is False


def test_negated_lower_yield_phrase_is_not_a_logic_warning():
    result = run_fact_check(_bundle("Growth rose, but the move was not paired with lower yields."))
    assert result["logic_warnings"] == []


def test_shorthand_numeric_claim_is_checked():
    result = run_fact_check(_bundle("S&P 500 -0.24% while DXY +0.37%."))
    assert result["numeric_claims_checked"] >= 2
    assert result["numeric_mismatches"] == []


def test_company_event_without_source_is_blocking_warning():
    bundle = _bundle("S&P 500 fell -0.24% while the risk-off backdrop held.")
    bundle["provenance_audit"] = {"status": "ok"}
    bundle["company_events"] = {
        "earnings": [{"ticker": "0700.HK", "comparison": "EPS est. 4.15 HKD"}],
        "ratings": [],
    }
    result = run_fact_check(bundle)
    assert result["status"] == "warning"
    assert any(item["type"] == "missing_event_source" for item in result["source_warnings"])


def test_truncated_llm_text_is_flagged():
    bundle = _bundle("This is a sufficiently long narrative sentence that appears to end abruptly with its.")
    result = run_fact_check(bundle)
    assert result["status"] == "warning"
    assert any(item["type"] == "truncated_text" for item in result["source_warnings"])


def test_false_relative_performance_claim_blocks_release():
    result = run_fact_check(_bundle("FXI outperformed 3033.HK despite a weak overnight tape."))
    warning = next(item for item in result["logic_warnings"] if item["type"] == "relative_performance")
    assert warning["severity"] == "critical"
    assert result["release_blocking"] is True


def test_true_same_session_relative_performance_is_allowed():
    result = run_fact_check(_bundle("Nasdaq 100 underperformed S&P 500 during the US session."))
    assert not any(item["type"] == "relative_performance" for item in result["logic_warnings"])
    assert not any(item["type"] == "period_alignment" for item in result["logic_warnings"])


def test_cross_session_ranking_requires_period_alignment_review():
    result = run_fact_check(_bundle("3033.HK outperformed FXI."))
    warning = next(item for item in result["logic_warnings"] if item["type"] == "period_alignment")
    assert warning["severity"] == "review"
    assert result["release_blocking"] is False


def main() -> None:
    test_us10y_change_and_level_are_not_confused()
    test_us10y_wrong_change_is_flagged_as_critical()
    test_risk_on_pockets_do_not_conflict_with_risk_off_regime()
    test_explicit_regime_conflict_is_review_warning()
    test_conditional_yield_watchpoint_is_not_a_logic_warning()
    test_unhedged_yield_direction_conflict_is_review_warning()
    test_negated_lower_yield_phrase_is_not_a_logic_warning()
    test_shorthand_numeric_claim_is_checked()
    test_company_event_without_source_is_blocking_warning()
    test_truncated_llm_text_is_flagged()
    print("Fact checker test passed")


if __name__ == "__main__":
    main()
