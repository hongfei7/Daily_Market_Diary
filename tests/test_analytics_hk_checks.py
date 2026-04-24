from _bootstrap import ROOT  # noqa: F401

from professional.analytics_hk_checks import build_hk_quick_checks


def test_hk_quick_checks_combine_local_metrics_and_market_proxies() -> None:
    summary = {
        "FX": {"USD/HKD": {"Price": 7.846, "As Of": "2026-04-13"}},
        "Equities": {
            "Hang Seng Index": {"Pct Change": "0.80%", "As Of": "2026-04-13"},
            "Hang Seng TECH ETF": {"Pct Change": "1.30%", "As Of": "2026-04-13"},
        },
    }
    movers_data = {
        "etf_flows": [
            {"ticker": "2800.HK", "change_pct": 0.8, "volume_ratio": 1.6, "estimated_flow_direction": "inflow"},
            {"ticker": "QQQ", "change_pct": 1.2, "volume_ratio": 1.4, "estimated_flow_direction": "inflow"},
        ]
    }
    hk_local = {
        "main_board_turnover": {
            "display_value": "HK$207.9bn",
            "status": "live_local",
            "source": "HKEX Daily Quotations",
            "as_of": "2026-04-13",
            "note": "Participation was active.",
        },
        "turnover_vs_20d": {
            "display_value": "1.18x | +18% vs 20D",
            "status": "live_local",
            "note": "Trailing 20-session average turnover was HK$176.3bn.",
        },
        "hibor_1m": {"display_value": "2.23%", "status": "live_local", "source": "HKMA", "as_of": "2026-04-13", "note": "Funding lens."},
        "aggregate_balance": {"display_value": "HK$54.4bn", "status": "live_local", "source": "HKMA", "as_of": "2026-04-13", "note": "Liquidity lens."},
        "base_rate": {"display_value": "4.00%", "status": "live_local", "source": "HKMA", "as_of": "2026-04-13", "note": "Base-rate anchor."},
        "linked_exchange_band": {
            "display_value": "7.7500 to 7.8500",
            "status": "live_local",
            "source": "HKMA",
            "as_of": "2026-04-13",
            "note": "Official USD/HKD band.",
        },
        "short_selling_ratio": {"display_value": "11.50%", "status": "live_local", "source": "HKEX", "as_of": "2026-04-13", "note": "Short pressure."},
        "southbound_net_flow": {
            "display_value": "Net HK$4.0bn | turnover HK$20.0bn",
            "status": "live_public",
            "source": "HKEX Stock Connect",
            "as_of": "2026-04-13",
            "note": "Southbound disclosed buy/sell turnover was net positive.",
        },
        "northbound_net_flow": {
            "display_value": "Turnover RMB15.0bn | net unavailable",
            "status": "partial_public",
            "source": "HKEX Stock Connect",
            "as_of": "2026-04-13",
            "note": "Northbound net-buy unavailable.",
        },
        "ah_premium_index": {"display_value": "32.40%", "status": "live_public", "source": "Public quotes", "as_of": "2026-04-13", "note": "Average premium."},
    }

    rows = build_hk_quick_checks(summary, movers_data, {"leadership": "Growth-led"}, hk_local)
    row_map = {row["metric"]: row for row in rows}

    assert len(rows) == 11
    assert row_map["Main Board turnover vs 20D"]["value"] == "1.18x | +18% vs 20D"
    assert row_map["USD/HKD spot vs band"]["status"] == "live_hybrid"
    assert row_map["USD/HKD spot vs band"]["value"] == "7.8460 | band 7.7500 to 7.8500"
    assert "weak-side Convertibility Undertaking" in row_map["USD/HKD spot vs band"]["note"]
    assert row_map["Southbound / Northbound net flow"]["status"] == "live_public"
    assert "Southbound Net HK$4.0bn" in row_map["Southbound / Northbound net flow"]["value"]
    assert row_map["AH premium index"]["value"] == "32.40%"
    assert row_map["Hong Kong leadership"]["value"] == "Growth-led"
    assert row_map["HSI vs HSTECH"]["value"] == "HSI +0.80% | HSTECH +1.30%"
    assert "2800.HK +0.80% on 1.60x volume" in row_map["HK ETF flow proxy"]["value"]
    assert "QQQ" not in row_map["HK ETF flow proxy"]["value"]


if __name__ == "__main__":
    test_hk_quick_checks_combine_local_metrics_and_market_proxies()
    print("Analytics HK checks test passed")
