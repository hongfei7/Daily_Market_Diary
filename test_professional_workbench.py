import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "market_diary"))

from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.daily_one_chart import generate_daily_one_chart
from professional.dashboard import generate_dashboard
from professional.report_builder import render_professional_report
from professional.trend_pack import generate_hk_trend_pack


def _minimal_fixture():
    return {
        "market_data": {
            "summary": {
                "Equities": {
                    "S&P 500": {"Price": 5000, "Pct Change": "1.20%"},
                    "Nasdaq 100": {"Price": 18000, "Pct Change": "1.60%"},
                    "Dow Jones": {"Price": 39500, "Pct Change": "0.55%"},
                    "Euro Stoxx 50": {"Price": 5100, "Pct Change": "-0.40%"},
                    "Hang Seng Index": {"Price": 17450, "Pct Change": "0.95%"},
                    "Hang Seng China Enterprises": {"Price": 6150, "Pct Change": "0.40%"},
                    "Hang Seng TECH ETF": {"Price": 4.88, "Pct Change": "1.30%"},
                    "CSI 300": {"Price": 3550, "Pct Change": "0.35%"},
                    "ChiNext Index": {"Price": 1920, "Pct Change": "-0.20%"},
                    "Nikkei 225": {"Price": 38200, "Pct Change": "0.90%"},
                    "China Large-Cap (FXI)": {"Price": 28, "Pct Change": "0.80%"},
                },
                "Rates": {"10Y Treasury": {"Price": 4.15, "Pct Change": "-0.60%"}},
                "FX": {
                    "DXY": {"Price": 104.2, "Pct Change": "-0.35%"},
                    "USD/CNH": {"Price": 7.18, "Pct Change": "0.10%"},
                    "USD/HKD": {"Price": 7.82, "Pct Change": "0.00%"},
                    "USD/JPY": {"Price": 151.8, "Pct Change": "0.20%"},
                },
                "Commodities": {
                    "Crude Oil": {"Price": 82.0, "Pct Change": "1.40%"},
                    "Brent Crude": {"Price": 85.0, "Pct Change": "1.80%"},
                    "Gold": {"Price": 2350, "Pct Change": "0.90%"},
                    "Copper": {"Price": 4.4, "Pct Change": "-1.10%"},
                },
                "Crypto": {"Bitcoin": {"Price": 70000, "Pct Change": "2.10%"}},
                "Vol": {"VIX": {"Price": 16.2, "Pct Change": "-3.50%"}},
            },
            "meta": {
                "requested_date": "2026-04-13",
                "effective_date": "2026-04-13",
                "summary_date": "2026-04-13",
                "market_quality": {
                    "available": 20,
                    "total": 20,
                    "fallback": ["Equities / FXI"],
                    "stale": [],
                    "missing": [],
                },
            },
        },
        "chart_features": {
            "fx_composite": {"available": True, "net_pp": -0.55, "range_pp": 0.88, "turning_points": []},
            "assets": {
                "Gold": {"available": True, "net_pp": 0.7, "range_pp": 1.1},
                "Oil": {"available": True, "net_pp": 1.2, "range_pp": 1.7},
                "Bitcoin": {"available": True, "net_pp": 2.0, "range_pp": 2.6},
            },
            "divergence": {"best_asset": "Bitcoin", "worst_asset": "Gold", "spread_pp": 1.3},
        },
        "macro_data": {
            "calendar": {
                "released": [{"time": "20:30", "country": "US", "indicator": "CPI MoM", "actual": "0.3%", "forecast": "0.2%", "previous": "0.4%", "impact": "high", "surprise": "beat"}],
                "upcoming": [{"time": "10:00", "country": "CN", "indicator": "PMI", "forecast": "50.2", "previous": "49.8", "impact": "high"}],
            },
            "central_bank_events": [{"time": "22:00", "bank": "Federal Reserve", "speaker": "Chair", "title": "Policy Outlook", "importance": "high", "event_type": "speech"}],
        },
        "sector_data": {
            "sector_news": {
                "Technology": [{"title": "NVIDIA signs new AI server deal", "summary": "Deal expands data-center pipeline.", "source": "reuters", "link": "https://example.com/a", "importance_score": 2.5}]
            },
            "earnings_calendar": [{"ticker": "AAPL", "company": "Apple", "time": "After close", "eps_estimate": "1.45", "revenue_estimate": "89.5B"}],
            "analyst_changes": [{"ticker": "NVDA", "firm": "Broker", "action": "Upgrade", "from_rating": "Neutral", "to_rating": "Buy", "price_target": "1200", "previous_target": "1000"}],
            "hkex_announcements": {
                "status": "ok",
                "data": {
                    "top_announcements": [
                        {
                            "grade": "A",
                            "ticker": "0700.HK",
                            "company": "Tencent",
                            "event_type": "Results announcement",
                            "title": "Annual results announcement",
                            "release_time": "2026-04-13 18:30",
                            "source": "HKEXnews",
                            "url": "https://www1.hkexnews.hk/example.pdf",
                            "score": 5.0,
                        }
                    ],
                    "watchlist_hits": [],
                },
                "meta": {"source": "HKEXnews", "available_count": 1},
            },
        },
        "movers_data": {
            "premarket_movers": {
                "gainers": [{"ticker": "NVDA", "change_pct": 3.5, "catalyst": "Earnings beat expectations"}],
                "losers": [{"ticker": "TSLA", "change_pct": -4.2, "catalyst": "Guidance cut"}],
            },
            "etf_flows": [{"ticker": "QQQ", "change_pct": 1.2, "volume_ratio": 1.4, "estimated_flow_direction": "inflow"}],
            "unusual_options": [{"ticker": "TSLA", "option_type": "Call", "volume_oi_ratio": 1.8, "sentiment": "bullish"}],
            "short_sell": {
                "status": "ok",
                "data": {
                    "market": {"short_ratio_pct": 15.0, "short_turnover_hkd": 31_200_000_000, "total_turnover_hkd": 207_900_000_000},
                    "top_short_ratio": [{"ticker": "0700.HK", "code": "00700", "name": "TENCENT", "short_ratio_pct": 22.5, "short_turnover_hkd": 2_300_000_000, "total_turnover_hkd": 10_300_000_000}],
                    "top_short_value": [{"ticker": "0700.HK", "code": "00700", "name": "TENCENT", "short_ratio_pct": 22.5, "short_turnover_hkd": 2_300_000_000, "total_turnover_hkd": 10_300_000_000}],
                    "watchlist_hits": [],
                },
                "meta": {"source": "HKEX Daily Quotations - Short Selling Turnover", "effective_date": "2026-04-13"},
            },
        },
        "risk_data": {
            "upcoming_events": [{"date": "2026-04-14", "type": "Options Expiry", "description": "Monthly expiry", "importance": "high"}],
            "sentiment_indicators": {"put_call_ratio": {"equity": 0.7, "index": 1.1, "interpretation": "equity optimism with index hedging"}},
        },
        "hk_local_data": {
            "status": "ok",
            "data": {
                "main_board_turnover": {"display_value": "HK$207.9bn", "status": "live_local", "source": "HKEX Daily Quotations", "as_of": "2026-04-13", "note": "Participation was active."},
                "turnover_vs_20d": {"display_value": "1.18x | +18% vs 20D", "status": "live_local", "source": "HKEX Daily Quotations", "as_of": "2026-04-13", "note": "Trailing 20-session average turnover was HK$176.3bn."},
                "hibor_1m": {"display_value": "2.23%", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Funding conditions were stable."},
                "aggregate_balance": {"display_value": "HK$54.4bn", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Liquidity remained ample."},
                "base_rate": {"display_value": "4.00%", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Base-rate anchor remained unchanged."},
                "linked_exchange_band": {"display_value": "7.7500 to 7.8500", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Official USD/HKD band."},
                "short_selling_ratio": {"display_value": "N/A", "status": "unavailable", "source": "HKEX Short Selling Turnover Report", "as_of": "", "note": "Only a morning-close snapshot was available."},
                "southbound_net_flow": {"display_value": "+HK$4.00bn", "status": "live_public", "source": "HKEX Stock Connect Historical Daily", "as_of": "2026-04-13", "note": "Southbound disclosed buy/sell turnover was net positive."},
                "northbound_net_flow": {"display_value": "N/A", "status": "unavailable", "source": "HKEX Stock Connect Historical Daily", "as_of": "2026-04-13", "note": "Northbound full-day net buy is not available in the current public file."},
                "ah_premium_index": {"display_value": "32.40%", "status": "live_public", "source": "Public Yahoo Finance quotes - calculated A/H premium", "as_of": "2026-04-13", "note": "Average covered A/H premium."},
            },
            "meta": {"report_date": "2026-04-13"},
        },
        "stock_connect_data": {
            "status": "ok",
            "data": {
                "southbound": {
                    "total_turnover": 20000.5,
                    "buy_turnover": 11500.25,
                    "sell_turnover": 7500.0,
                    "net_buy": 4000.25,
                    "net_buy_available": True,
                    "top_active": [
                        {"ticker": "03690.HK", "name": "MEITUAN-W", "buy_turnover": 900.0, "sell_turnover": 400.0, "net_buy": 500.0, "total_turnover": 1300.0},
                        {"ticker": "00700.HK", "name": "TENCENT", "buy_turnover": 1200.0, "sell_turnover": 800.0, "net_buy": 400.0, "total_turnover": 2000.0},
                    ],
                },
                "northbound": {"total_turnover": 15000.0, "net_buy": None, "net_buy_available": False, "top_active": []},
                "markets": [],
            },
            "meta": {"source": "HKEX Stock Connect Historical Daily", "effective_date": "2026-04-13"},
        },
        "ah_premium_data": {
            "status": "ok",
            "data": {
                "average_premium": 32.4,
                "top_premium": [
                    {"name": "CRRC", "a_ticker": "601766.SS", "h_ticker": "1766.HK", "premium_pct": 82.5},
                    {"name": "China Railway", "a_ticker": "601390.SS", "h_ticker": "0390.HK", "premium_pct": 64.2},
                ],
                "lowest_premium": [],
                "rows": [],
            },
            "meta": {"source": "Public Yahoo Finance quotes - calculated A/H premium", "effective_date": "2026-04-13", "coverage": 2, "universe": 2},
        },
        "china_rates_data": {
            "status": "ok",
            "data": {
                "china_10y": {"display_value": "1.79%", "status": "live_public", "source": "Eastmoney Treasury Yield History", "as_of": "2026-04-13", "note": "China local rates anchor.", "change_display": "-1.2bp"},
                "cn_us_10y_spread": {"display_value": "-250.7bp", "status": "live_public", "source": "Eastmoney Treasury Yield History", "as_of": "2026-04-13", "note": "Relative carry lens.", "change_display": "-3.4bp"},
            },
            "meta": {"report_date": "2026-04-13"},
        },
    }


def main():
    fixture = _minimal_fixture()
    config = load_professional_config()
    config["watchlists"] = {"core_coverage": [], "focus_pool": [], "learning_pool": []}
    config["report"].update(
        {
            "quick_items_limit": 3,
            "quick_macro_events": 2,
            "top_macro_events": 3,
            "top_news_items": 1,
            "top_movers": 1,
            "top_high_frequency_items": 2,
            "top_catalysts": 4,
            "top_source_links": 5,
            "quick_watchlist_items_per_bucket": 1,
            "watchlist_story_limit": 1,
            "catalyst_window_days": 3,
        }
    )

    bundle = build_professional_bundle(
        report_date="2026-04-13",
        config=config,
        market_data=fixture["market_data"],
        chart_features=fixture["chart_features"],
        macro_data=fixture["macro_data"],
        sector_data=fixture["sector_data"],
        movers_data=fixture["movers_data"],
        risk_data=fixture["risk_data"],
        news_headlines=[],
        stock_connect_data=fixture["stock_connect_data"],
        ah_premium_data=fixture["ah_premium_data"],
        hk_local_data=fixture["hk_local_data"],
        china_rates_data=fixture["china_rates_data"],
    )

    assert bundle["overview"]["theme"]
    assert bundle["must_watch"]
    bundle["llm_sections"] = {
        "one_line_market_pulse": "Cooling dollar pressure and firmer Hong Kong growth proxies left the overnight setup mildly constructive for the open.",
        "deep_read_setup": "The overnight tape leaned modestly constructive as softer dollar pressure and lower Treasury yields supported growth-sensitive assets. Hong Kong proxies also held up, which matters because the local open is often framed first through offshore China risk appetite rather than domestic headlines alone.",
        "selected_news": [
            {
                "headline": "NVIDIA signs new AI server deal",
                "why_it_matters": "It reinforces demand confidence for AI-related supply chains.",
                "hk_market_impact": "Helpful for Hong Kong growth and platform sentiment.",
                "importance": "A",
            }
        ],
        "overnight_drivers": ["Lower yields supported growth-sensitive assets.", "AI sentiment remained constructive."],
        "overnight_hk_implication": "Hong Kong should test whether growth leadership broadens on the open.",
        "hk_review_setup": "Hong Kong follow-through should be judged through HSTECH versus HSCEI rather than index direction alone.",
        "hk_local_leadership": "Growth is leading for now.",
        "hk_follow_through": "Watch FXI and USD/CNH for confirmation.",
        "macro_takeaway": "Macro still matters because any rates surprise can quickly reprice the opening setup.",
        "macro_watchpoints": ["US yields", "USD/CNH"],
        "company_takeaway": "Company-level flow matters because earnings and upgrades can reset short-term narratives.",
        "company_notes": [{"ticker": "AAPL", "commentary": "Expectations look manageable into results."}],
        "theme_paragraph": "The weekly theme still looks live because current signals are clustering rather than staying isolated.",
        "theme_watch_items": ["Check related catalysts.", "Monitor follow-through in leadership names."],
        "thinking_note": "Use the first 30 minutes to test whether Hong Kong growth leadership is broadening or fading back into index-heavy beta.",
        "risk_check": "A sudden reversal in yields or a stronger USD could quickly compress the overnight risk-on read.",
        "interview_answer": "The setup is constructive but not euphoric. I would frame today as a data-sensitive Hong Kong open with growth leadership worth testing, not assuming.",
    }

    dashboard_path = os.path.join("reports_professional", "charts", "test_dashboard.png")
    generate_dashboard(bundle, dashboard_path)
    assert os.path.exists(dashboard_path)

    daily_chart_path = os.path.join("reports_professional", "charts", "test_daily_one_chart.png")
    daily_meta = generate_daily_one_chart(bundle, daily_chart_path)
    bundle["daily_one_chart"] = {**daily_meta, "rel_path": "charts/test_daily_one_chart.png"}
    assert os.path.exists(daily_chart_path)
    assert os.path.basename(dashboard_path) != os.path.basename(daily_chart_path)

    trend_pack_data = {
        "southbound": [
            {"date": "2026-04-08", "net_buy_hkd_bn": 1.2, "cumulative_hkd_bn": 1.2},
            {"date": "2026-04-09", "net_buy_hkd_bn": -0.6, "cumulative_hkd_bn": 0.6},
            {"date": "2026-04-10", "net_buy_hkd_bn": 2.0, "cumulative_hkd_bn": 2.6},
            {"date": "2026-04-13", "net_buy_hkd_bn": 1.4, "cumulative_hkd_bn": 4.0},
        ],
        "liquidity": [
            {"date": "2026-04-08", "hibor_1m": 2.05, "aggregate_balance_bn": 55.1},
            {"date": "2026-04-09", "hibor_1m": 2.09, "aggregate_balance_bn": 54.8},
            {"date": "2026-04-10", "hibor_1m": 2.16, "aggregate_balance_bn": 54.5},
            {"date": "2026-04-13", "hibor_1m": 2.23, "aggregate_balance_bn": 54.4},
        ],
        "leadership": {
            "dates": ["2026-04-08", "2026-04-09", "2026-04-10", "2026-04-13"],
            "series": {
                "HSI": [100.0, 100.6, 101.1, 101.4],
                "HSCEI": [100.0, 100.4, 100.8, 100.9],
                "HSTECH": [100.0, 101.0, 101.9, 102.2],
            },
        },
        "ah_heatmap": {
            "dates": ["2026-04-08", "2026-04-09", "2026-04-10", "2026-04-13"],
            "names": ["CRRC", "China Railway"],
            "matrix": [[80.0, 81.0, 82.0, 82.5], [62.5, 63.0, 63.8, 64.2]],
        },
    }
    trend_pack_path = os.path.join("reports_professional", "charts", "test_hk_trend_pack.png")
    trend_pack_meta = generate_hk_trend_pack(bundle, trend_pack_path, trend_data=trend_pack_data)
    bundle["trend_pack"] = {**trend_pack_meta, "rel_path": "charts/test_hk_trend_pack.png"}
    assert os.path.exists(trend_pack_path)
    assert os.path.basename(trend_pack_path) != os.path.basename(daily_chart_path)

    report = render_professional_report(
        bundle,
        charts_section="Charts placeholder",
        dashboard_rel_path="charts/test_dashboard.png",
        daily_chart_rel_path="charts/test_daily_one_chart.png",
        trend_pack_rel_path="charts/test_hk_trend_pack.png",
    )
    assert "Morning Research Workbench" in report
    assert "Layer 1 | Scan" in report
    assert "One-Line Market Pulse" in report
    assert "Layer 3 | Thinking" in report
    assert "Daily One Chart" in report
    assert "Hong Kong Trend Pack" in report
    assert "Risk Dashboard" in report
    assert "Flow Tracker and Attribution" in report
    assert "HKEX Announcements" in report
    assert "Stock Connect Southbound Active Names" in report
    assert "AH Premium Dispersion" in report
    assert "MEITUAN-W" in report
    assert "CRRC" in report
    assert "![Research Dashboard](charts/test_dashboard.png)" in report
    assert "![Daily One Chart](charts/test_daily_one_chart.png)" in report
    assert "![Hong Kong Trend Pack](charts/test_hk_trend_pack.png)" in report
    assert "![Daily One Chart](charts/test_dashboard.png)" not in report
    assert "Curated overnight stories" in report
    assert "LLM Quick Takes" in report
    assert "Market data quality" in report
    assert "Live local" in report
    assert "China 10Y" in report
    assert "Pending adapter" not in report

    print("Professional workbench test passed")


if __name__ == "__main__":
    main()
