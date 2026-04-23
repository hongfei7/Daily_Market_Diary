import sys
from pathlib import Path

from _bootstrap import ROOT  # noqa: F401

from professional.fact_checker import run_fact_check


def _bundle(text: str, *, risk_regime: str = "Risk-Off"):
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
            "Rates": {"10Y Treasury": {"Pct Change": "0.99%", "Price": 4.292}},
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
    result = run_fact_check(_bundle("US 10Y yield rose +0.99% to 4.292%, pressuring duration assets."))
    assert result["status"] == "ok"
    assert result["numeric_mismatches"] == []
    assert result["numeric_claims_checked"] >= 1


def test_us10y_wrong_change_is_flagged_as_critical():
    result = run_fact_check(_bundle("US 10Y yield rose +1.80% to 4.292%, pressuring duration assets."))
    assert result["status"] == "warning"
    mismatch = result["numeric_mismatches"][0]
    assert mismatch["label"] == "US 10Y"
    assert mismatch["claim_type"] == "change_pct"
    assert mismatch["severity"] == "critical"


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


def main() -> None:
    test_us10y_change_and_level_are_not_confused()
    test_us10y_wrong_change_is_flagged_as_critical()
    test_risk_on_pockets_do_not_conflict_with_risk_off_regime()
    test_explicit_regime_conflict_is_review_warning()
    print("Fact checker test passed")


if __name__ == "__main__":
    main()
