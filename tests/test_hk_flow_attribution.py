import os
import sys

from _bootstrap import ROOT  # noqa: F401

from modules.adapter_shortsell import _market_summary, _parse_stock_rows
from professional.attribution import build_attribution


def test_short_sell_parser() -> None:
    section = """
      CODE  NAME OF STOCK            (SH)             ($)             (SH)                ($)

        700 TENCENT               4,000,000       2,000,000,000      20,000,000      10,000,000,000
       9988 BABA-W                2,000,000         500,000,000      10,000,000       3,000,000,000

    (C) Short Selling of all Designated Securities
        Short Selling Turnover Total Value ($)                    : HKD      2,500,000,000
    Total market turnover                                         : HKD     20,000,000,000
    Short Selling of all Designated Securities as % total turnover             :       12.5%
    """
    rows = _parse_stock_rows(section)
    market = _market_summary(section, rows)

    assert len(rows) == 2
    assert rows[0]["ticker"] == "00700.HK"
    assert rows[0]["short_ratio_pct"] == 20.0
    assert market["short_ratio_pct"] == 12.5


def test_attribution_uses_local_flow() -> None:
    summary = {
        "Equities": {
            "S&P 500": {"Pct Change": "1.20%"},
            "Nasdaq 100": {"Pct Change": "1.60%"},
            "Hang Seng Index": {"Pct Change": "0.80%"},
            "Hang Seng TECH ETF": {"Pct Change": "1.30%"},
            "China Large-Cap (FXI)": {"Pct Change": "0.90%"},
        },
        "Rates": {"10Y Treasury": {"Pct Change": "-0.60%"}},
        "FX": {"DXY": {"Pct Change": "-0.30%"}, "USD/CNH": {"Pct Change": "-0.10%"}},
        "Vol": {"VIX": {"Pct Change": "-3.00%"}},
        "Commodities": {"Brent Crude": {"Pct Change": "0.40%"}},
    }
    hk_local = {
        "turnover_vs_20d": {"value": 1.18},
        "short_selling_ratio": {"value": 11.5},
    }
    movers_digest = {"short_sell": {"data": {"market": {"short_ratio_pct": 11.5}}}}
    overview = {"theme": "Risk-On backdrop"}

    attribution = build_attribution(summary, hk_local, movers_digest, overview)

    assert attribution["risk_dashboard"]["bucket"] == "Risk-on"
    assert any(item["name"] == "US growth-style transmission" for item in attribution["dominant_drivers"])
    assert "participation is active" in attribution["flow_summary"]


if __name__ == "__main__":
    test_short_sell_parser()
    test_attribution_uses_local_flow()
    print("HK flow attribution test passed")
