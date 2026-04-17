import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "market_diary"))

from modules.adapter_ah_premium import _calculate_premium
from modules.adapter_stockconnect import _aggregate_markets, _extract_payload, _parse_tab


def _sample_stock_connect_payload():
    return """
    tabData = [
      {
        "market": "SSE Southbound",
        "date": "2026-04-13",
        "tradingDay": true,
        "content": [
          {"table": {"schema": [["Total Turnover", "Buy Turnover", "Sell Turnover", "Total Trade Count", "ETF Turnover"]], "tr": [
            {"td": [["12,000.50"]]},
            {"td": [["7,500.25"]]},
            {"td": [["4,500.00"]]},
            {"td": [["110,000"]]},
            {"td": [["650.00"]]}
          ]}},
          {"table": {"schema": [["Rank", "Stock Code", "Stock Name", "Buy Turnover", "Sell Turnover", "Total Turnover"]], "tr": [
            {"td": [["1", "700", "TENCENT", "1,200,000,000", "800,000,000", "2,000,000,000"]]},
            {"td": [["2", "9988", "BABA-W", "500,000,000", "700,000,000", "1,200,000,000"]]}
          ]}}
        ]
      },
      {
        "market": "SZSE Southbound",
        "date": "2026-04-13",
        "tradingDay": true,
        "content": [
          {"table": {"schema": [["Total Turnover", "Buy Turnover", "Sell Turnover", "Total Trade Count", "ETF Turnover"]], "tr": [
            {"td": [["8,000.00"]]},
            {"td": [["4,000.00"]]},
            {"td": [["3,000.00"]]},
            {"td": [["80,000"]]},
            {"td": [["320.00"]]}
          ]}},
          {"table": {"schema": [["Rank", "Stock Code", "Stock Name", "Buy Turnover", "Sell Turnover", "Total Turnover"]], "tr": [
            {"td": [["1", "3690", "MEITUAN-W", "900,000,000", "400,000,000", "1,300,000,000"]]}
          ]}}
        ]
      },
      {
        "market": "SSE Northbound",
        "date": "2026-04-13",
        "tradingDay": true,
        "content": [
          {"table": {"schema": [["Total Turnover", "Total Trade Count"]], "tr": [
            {"td": [["15,000.00"]]},
            {"td": [["90,000"]]}
          ]}},
          {"table": {"schema": [["Rank", "Stock Code", "Stock Name", "Total Turnover"]], "tr": [
            {"td": [["1", "600519", "KWEICHOW MOUTAI", "2,100,000,000"]]}
          ]}}
        ]
      }
    ];
    """


def main() -> None:
    payload = _extract_payload(_sample_stock_connect_payload())
    parsed = [_parse_tab(tab) for tab in payload]
    southbound = _aggregate_markets(parsed, "Southbound")
    northbound = _aggregate_markets(parsed, "Northbound")

    assert len(parsed) == 3
    assert southbound["total_turnover"] == 20000.5
    assert southbound["buy_turnover"] == 11500.25
    assert southbound["sell_turnover"] == 7500.0
    assert round(southbound["net_buy"], 2) == 4000.25
    assert southbound["net_buy_available"] is True
    assert southbound["top_active"][0]["ticker"] == "03690.HK"
    assert southbound["top_active"][1]["ticker"] == "00700.HK"
    assert northbound["net_buy_available"] is False
    assert northbound["top_active"][0]["ticker"] == "600519"

    premium = _calculate_premium(a_price_cny=10.0, h_price_hkd=8.0, cny_hkd=1.08)
    assert round(premium, 2) == 35.0
    assert _calculate_premium(a_price_cny=10.0, h_price_hkd=0.0, cny_hkd=1.08) is None

    print("Stock Connect and AH premium test passed")


if __name__ == "__main__":
    main()
