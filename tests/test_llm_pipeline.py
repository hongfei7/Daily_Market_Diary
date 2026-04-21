import os
import sys

from _bootstrap import ROOT  # noqa: F401

from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.llm_enhancer import generate_llm_sections


def _fixture():
    return {
        "market_data": {
            "summary": {
                "Equities": {
                    "S&P 500": {"Price": 5000, "Pct Change": "1.20%"},
                    "Nasdaq 100": {"Price": 18000, "Pct Change": "1.60%"},
                    "Hang Seng Index": {"Price": 17450, "Pct Change": "0.95%"},
                    "Hang Seng China Enterprises": {"Price": 6150, "Pct Change": "0.40%"},
                    "Hang Seng TECH ETF": {"Price": 4.88, "Pct Change": "1.30%"},
                    "China Large-Cap (FXI)": {"Price": 28, "Pct Change": "0.80%"},
                },
                "Rates": {"10Y Treasury": {"Price": 4.15, "Pct Change": "-0.60%"}},
                "FX": {
                    "DXY": {"Price": 104.2, "Pct Change": "-0.35%"},
                    "USD/CNH": {"Price": 7.18, "Pct Change": "0.10%"},
                    "USD/HKD": {"Price": 7.82, "Pct Change": "0.00%"},
                },
                "Commodities": {
                    "Brent Crude": {"Price": 85.0, "Pct Change": "1.80%"},
                    "Gold": {"Price": 2350, "Pct Change": "0.90%"},
                    "Copper": {"Price": 4.4, "Pct Change": "-1.10%"},
                },
                "Vol": {"VIX": {"Price": 16.2, "Pct Change": "-3.50%"}},
            },
            "meta": {
                "requested_date": "2026-04-13",
                "effective_date": "2026-04-13",
                "summary_date": "2026-04-13",
                "market_quality": {"available": 18, "total": 20, "fallback": [], "stale": [], "missing": []},
            },
        },
        "chart_features": {"fx_composite": {"available": False}, "assets": {}, "divergence": {}},
        "macro_data": {
            "calendar": {
                "released": [{"time": "20:30", "country": "US", "indicator": "CPI MoM", "actual": "0.3%", "forecast": "0.2%", "previous": "0.4%", "impact": "high", "surprise": "beat"}],
                "upcoming": [{"time": "10:00", "country": "CN", "indicator": "PMI", "forecast": "50.2", "previous": "49.8", "impact": "high"}],
            },
            "central_bank_events": [],
        },
        "sector_data": {
            "sector_news": {
                "Technology": [{"title": "NVIDIA signs new AI server deal", "summary": "Deal expands data-center pipeline.", "source": "reuters", "link": "https://example.com/a", "importance_score": 2.5}]
            },
            "earnings_calendar": [{"ticker": "AAPL", "company": "Apple", "time": "After close", "eps_estimate": "1.45", "revenue_estimate": "89.5B"}],
            "analyst_changes": [{"ticker": "NVDA", "firm": "Broker", "action": "Upgrade", "from_rating": "Neutral", "to_rating": "Buy", "price_target": "1200", "previous_target": "1000"}],
            "hkex_announcements": {"status": "ok", "data": {"top_announcements": [{"grade": "A", "ticker": "0700.HK", "company": "Tencent", "event_type": "Results announcement", "title": "Annual results announcement", "release_time": "2026-04-13 18:30", "source": "HKEXnews", "url": "https://www1.hkexnews.hk/example.pdf", "score": 5.0}], "watchlist_hits": []}, "meta": {"source": "HKEXnews"}},
        },
        "movers_data": {
            "premarket_movers": {
                "gainers": [{"ticker": "NVDA", "change_pct": 3.5, "catalyst": "Earnings beat expectations"}],
                "losers": [{"ticker": "TSLA", "change_pct": -4.2, "catalyst": "Guidance cut"}],
            },
            "etf_flows": [{"ticker": "3033.HK", "change_pct": 1.2, "volume_ratio": 1.4, "estimated_flow_direction": "inflow"}],
            "unusual_options": [{"ticker": "TSLA", "option_type": "Call", "volume_oi_ratio": 1.8, "sentiment": "bullish"}],
            "short_sell": {"status": "ok", "data": {"market": {"short_ratio_pct": 15.0, "short_turnover_hkd": 31_200_000_000, "total_turnover_hkd": 207_900_000_000}, "top_short_ratio": [], "top_short_value": [], "watchlist_hits": []}, "meta": {"source": "HKEX Daily Quotations", "effective_date": "2026-04-13"}},
        },
        "risk_data": {
            "upcoming_events": [{"date": "2026-04-14", "type": "Options Expiry", "description": "Monthly expiry", "importance": "high"}],
            "sentiment_indicators": {"put_call_ratio": {"equity": 0.7, "index": 1.1, "interpretation": "equity optimism with index hedging"}},
        },
        "stock_connect_data": {
            "status": "ok",
            "data": {
                "southbound": {
                    "net_buy": 4000.25,
                    "net_buy_available": True,
                    "top_active": [{"ticker": "00700.HK", "name": "TENCENT", "net_buy": 400.0, "total_turnover": 2000.0}],
                },
                "northbound": {"net_buy": None, "net_buy_available": False, "top_active": []},
            },
            "meta": {"source": "HKEX Stock Connect Historical Daily", "effective_date": "2026-04-13"},
        },
        "ah_premium_data": {
            "status": "ok",
            "data": {
                "average_premium": 32.4,
                "top_premium": [{"name": "CRRC", "a_ticker": "601766.SS", "h_ticker": "1766.HK", "premium_pct": 82.5}],
                "rows": [{"name": "CRRC", "premium_pct": 82.5}],
            },
            "meta": {"source": "Public Yahoo Finance quotes - calculated A/H premium", "effective_date": "2026-04-13"},
        },
    }


def main() -> None:
    data = _fixture()
    config = load_professional_config()
    config["watchlists"] = {"core_coverage": [], "focus_pool": [], "learning_pool": []}

    bundle = build_professional_bundle(
        report_date="2026-04-13",
        config=config,
        market_data=data["market_data"],
        chart_features=data["chart_features"],
        macro_data=data["macro_data"],
        sector_data=data["sector_data"],
        movers_data=data["movers_data"],
        risk_data=data["risk_data"],
        news_headlines=["- Reuters: AI demand remains firm"],
        stock_connect_data=data["stock_connect_data"],
        ah_premium_data=data["ah_premium_data"],
    )

    seen_tasks = []

    def fake_runner(task_name, context, prompt):
        seen_tasks.append(task_name)
        if task_name in {"hk_review", "final_framing"}:
            assert "flow_tracker" in context
            assert context["flow_tracker"].get("stock_connect", {}).get("status") == "ok"
            assert context["flow_tracker"].get("ah_premium", {}).get("status") == "ok"
        payloads = {
            "news_selection": {
                "summary": "Overnight headlines leaned constructive for Hong Kong growth sentiment.",
                "selected_news": [{"headline": "AI demand remains firm", "why_it_matters": "Supports growth positioning.", "hk_market_impact": "Helps HSTECH and platform sentiment.", "importance": "A"}],
            },
            "overnight_review": {
                "paragraph": "US markets rallied as lower yields and steadier growth expectations supported risk appetite.",
                "drivers": ["Lower yields supported growth.", "AI sentiment stayed firm."],
                "hk_open_implication": "Hong Kong should test whether growth leadership broadens on the open.",
            },
            "hk_review": {
                "paragraph": "Hong Kong follow-through should be judged through HSTECH versus HSCEI rather than index direction alone.",
                "local_leadership": "Growth is leading for now.",
                "follow_through": "Watch FXI and USD/CNH for confirmation.",
            },
            "macro_interpretation": {
                "paragraph": "Macro still matters because any rates surprise can quickly reprice the opening setup.",
                "watchpoints": ["US yields", "USD/CNH"],
            },
            "company_commentary": {
                "paragraph": "Company-level flow matters because earnings and upgrades can reset short-term narratives.",
                "company_notes": [{"ticker": "AAPL", "commentary": "Expectations look manageable into results."}],
            },
            "theme_deep_dive": {
                "paragraph": "The weekly theme still looks live because current signals are clustering rather than staying isolated.",
                "watch_items": ["Check related catalysts.", "Monitor follow-through in leadership names."],
            },
            "final_framing": {
                "one_line_market_pulse": "Lower yields and firm AI sentiment left the overnight setup mildly constructive for Hong Kong.",
                "thinking_note": "Treat the opening tone as promising but still conditional on local follow-through.",
                "risk_check": "A rebound in yields or a stronger dollar could quickly weaken the setup.",
                "interview_answer": "The setup is constructive but not one-way. I would watch Hong Kong growth leadership for confirmation.",
            },
        }
        return payloads[task_name], {"status": "ok", "model": "fake-model", "route": "test", "attempts": 1}

    llm_sections = generate_llm_sections(bundle=bundle, config=config, cache_dir="", runner=fake_runner)

    assert "news_selection" in seen_tasks
    assert "overnight_review" in seen_tasks
    assert "final_framing" in seen_tasks
    assert llm_sections["one_line_market_pulse"]
    assert llm_sections["deep_read_setup"]
    assert llm_sections["hk_review_setup"]
    assert llm_sections["macro_takeaway"]
    assert llm_sections["company_takeaway"]
    assert llm_sections["theme_paragraph"]
    assert llm_sections["selected_news"][0]["headline"] == "AI demand remains firm"
    assert llm_sections["task_meta"]["tasks"]["final_framing"]["status"] == "ok"

    weekly_bundle = build_professional_bundle(
        report_date="2026-04-18",
        briefing_date="2026-04-19",
        global_market_date="2026-04-18",
        hk_data_date="2026-04-17",
        config=config,
        market_data=data["market_data"],
        chart_features=data["chart_features"],
        macro_data=data["macro_data"],
        sector_data=data["sector_data"],
        movers_data=data["movers_data"],
        risk_data=data["risk_data"],
        news_headlines=["- Reuters: AI demand remains firm"],
        stock_connect_data=data["stock_connect_data"],
        ah_premium_data=data["ah_premium_data"],
    )
    weekly_bundle["weekly_review"]["trend_summary"] = {
        "status": "ok",
        "window": {"start": "2026-04-13", "end": "2026-04-17"},
        "rows": [
            {
                "signal": "Southbound flow",
                "weekly_change": "+4.0bn over 5 sessions",
                "latest": "+1.4bn",
                "read": "Southbound flow stayed net positive into the weekly close.",
            }
        ],
    }

    weekly_seen = {}

    def weekly_runner(task_name, context, prompt):
        weekly_seen[task_name] = {"context": context, "prompt": prompt}
        return fake_runner(task_name, context, prompt)

    weekly_sections = generate_llm_sections(bundle=weekly_bundle, config=config, cache_dir="", runner=weekly_runner)
    assert weekly_sections["one_line_market_pulse"]
    assert weekly_seen["overnight_review"]["context"]["weekly_review"]["trend_summary"]["rows"][0]["signal"] == "Southbound flow"
    assert weekly_seen["hk_review"]["context"]["weekly_review"]["desk_questions"]
    assert weekly_seen["final_framing"]["context"]["weekly_review"]["next_week"]
    assert "Weekly-review rule" in weekly_seen["overnight_review"]["prompt"]
    assert "Use weekly_review.trend_summary" in weekly_seen["overnight_review"]["prompt"]

    print("LLM pipeline test passed")


if __name__ == "__main__":
    main()
