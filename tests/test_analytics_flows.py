from _bootstrap import ROOT  # noqa: F401

from professional.analytics_flows import build_flow_tracker, build_movers_and_flows


def test_movers_and_flows_summarizes_market_activity() -> None:
    movers_data = {
        "premarket_movers": {
            "gainers": [{"ticker": "NVDA", "change_pct": 3.5, "catalyst": "Earnings beat expectations"}],
            "losers": [{"ticker": "TSLA", "change_pct": -4.2, "catalyst": "Guidance cut"}],
        },
        "etf_flows": [{"ticker": "QQQ", "change_pct": 1.2, "volume_ratio": 1.4, "estimated_flow_direction": "inflow"}],
        "unusual_options": [{"ticker": "TSLA", "option_type": "Call", "volume_oi_ratio": 1.8, "sentiment": "bullish"}],
        "short_sell": {
            "data": {
                "market": {"short_ratio_pct": 15.0, "short_turnover_hkd": 31_200_000_000},
                "top_short_ratio": [{"ticker": "0700.HK", "short_ratio_pct": 22.5}],
                "top_short_value": [{"ticker": "0700.HK", "short_turnover_hkd": 2_300_000_000}],
                "watchlist_hits": [{"ticker": "0700.HK"}],
            },
            "meta": {"effective_date": "2026-04-13"},
        },
    }
    risk_data = {
        "sentiment_indicators": {
            "put_call_ratio": {"equity": 0.7, "index": 1.1, "interpretation": "equity optimism with index hedging"}
        }
    }

    digest = build_movers_and_flows(movers_data, risk_data)

    assert digest["movers"][0]["attribution"] == "Announcement / expectations"
    assert digest["movers"][1]["title"] == "TSLA -4.20%"
    assert digest["short_sell_top_ratio"][0]["ticker"] == "0700.HK"
    assert digest["short_sell_top_value"][0]["ticker"] == "0700.HK"
    assert digest["short_sell_watchlist_hits"][0]["ticker"] == "0700.HK"
    assert any("ETF flow anomalies" in bullet for bullet in digest["flow_bullets"])
    assert any("HKEX short selling" in bullet for bullet in digest["flow_bullets"])
    assert any("Put/Call structure" in bullet for bullet in digest["flow_bullets"])


def test_flow_tracker_adds_public_connect_and_premium_bullets() -> None:
    hk_quick_checks = [
        {"metric": "Main Board turnover vs 20D", "value": "1.18x"},
        {"metric": "Southbound / Northbound net flow", "value": "Southbound live"},
        {"metric": "AH premium index", "value": "32.40%"},
    ]
    movers_digest = {
        "flow_bullets": ["ETF flow anomalies were concentrated in: QQQ +1.20%."],
        "etf_flows": [{"ticker": "QQQ"}],
        "short_sell_top_ratio": [{"ticker": "0700.HK"}],
    }
    stock_connect_data = {
        "status": "ok",
        "data": {
            "southbound": {"net_buy": 4000.25, "total_turnover": 20000.5},
            "northbound": {"net_buy": None, "total_turnover": 15000.0},
        },
        "meta": {"effective_date": "2026-04-13"},
    }
    ah_premium_data = {
        "status": "ok",
        "data": {"average_premium": 32.4, "top_premium": [{"name": "CRRC", "premium_pct": 82.5}]},
    }

    tracker = build_flow_tracker(
        hk_quick_checks,
        movers_digest,
        {"flow_summary": "Participation is active."},
        stock_connect_data,
        ah_premium_data,
    )

    assert tracker["summary"] == "Participation is active."
    assert [item["metric"] for item in tracker["key_metrics"]] == [
        "Main Board turnover vs 20D",
        "Southbound / Northbound net flow",
        "AH premium index",
    ]
    assert any("Stock Connect Southbound" in bullet for bullet in tracker["flow_bullets"])
    assert any("turnover RMB15.0bn" in bullet for bullet in tracker["flow_bullets"])
    # The covered-pair average is taken over whichever names resolved today, so
    # the bullet must say the level is not comparable with prior reports.
    assert any("AH premium: covered-pair average +32.40%" in bullet for bullet in tracker["flow_bullets"])
    assert any("not comparable with prior reports" in bullet for bullet in tracker["flow_bullets"])
    assert tracker["stock_connect"] is stock_connect_data
    assert tracker["ah_premium"] is ah_premium_data


if __name__ == "__main__":
    test_movers_and_flows_summarizes_market_activity()
    test_flow_tracker_adds_public_connect_and_premium_bullets()
    print("Analytics flows test passed")
