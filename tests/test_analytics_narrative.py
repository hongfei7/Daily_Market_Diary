from _bootstrap import ROOT  # noqa: F401

from professional.analytics_narrative import (
    build_non_trading_focus,
    build_reflection_prompts,
    build_theme_deep_dive,
    build_today_forward,
    build_weekly_review,
)


def test_narrative_builds_theme_forward_and_non_trading_focus() -> None:
    config = {
        "thinking": {
            "rotation": [
                {
                    "weekday": 0,
                    "theme": "AI platform demand",
                    "angle": "Watch whether AI orders broaden into Hong Kong platform sentiment.",
                    "keywords": ["ai", "platform"],
                }
            ],
            "reflection_prompts": ["What would change my mind before lunch?"],
        }
    }
    sector_digest = {
        "graded_news": [
            {
                "sector": "Technology",
                "title": "AI platform orders accelerate",
                "summary": "New server demand",
                "why": "It supports growth-beta sentiment.",
                "grade": "A",
            }
        ]
    }
    watchlists = {
        "Core coverage": [
            {
                "ticker": "0700.HK",
                "name": "Tencent",
                "bucket": "Core coverage",
                "thesis": "Platform recovery",
                "note": "AI services optionality",
                "upcoming_catalyst": "Earnings",
            }
        ],
        "Learning watchlist": [
            {
                "ticker": "2800.HK",
                "name": "Tracker Fund of Hong Kong",
                "bucket": "Learning watchlist",
                "thesis": "AI platform beta proxy",
                "note": "AI platform beta proxy",
            }
        ],
    }
    high_frequency = [
        {"label": "VIX", "category": "Vol", "price": 16.2, "change_pct": -3.5, "interpretation": "Vol compression supports risk."},
        {"label": "Gold", "category": "Commodities", "price": 2350, "change_pct": 0.9, "interpretation": "Hedge demand is contained."},
    ]
    catalysts = [
        {"date": "2026-04-13", "event": "Tencent: Earnings", "category": "Core coverage", "impact": "Platform earnings reset."},
        {"date": "2026-04-14", "event": "US CPI", "category": "Upcoming", "impact": "Rates reaction."},
    ]

    theme = build_theme_deep_dive("2026-04-13", config, sector_digest, watchlists, high_frequency, catalysts)
    assert theme["theme"] == "AI platform demand"
    assert theme["news"][0]["title"] == "AI platform orders accelerate"
    assert theme["related_names"][0]["ticker"] == "0700.HK"
    assert all(item["bucket"] != "Learning watchlist" for item in theme["related_names"])
    assert any("VIX -3.50%" in line for line in theme["signals"])

    today = build_today_forward(
        "2026-04-13",
        [{"date": "2026-04-13", "event": "US CPI", "impact": "Rates reaction."}],
        catalysts,
        day_mode={"is_trading_day": False},
    )
    assert today["today_macro"][0]["event"] == "US CPI"
    assert "Non-trading review" in today["focus_lines"][0]
    assert today["next_catalysts"][0]["event"] == "Tencent: Earnings"

    non_trading = build_non_trading_focus(
        day_mode={"is_trading_day": False, "mode": "non_trading"},
        date_semantics={"review_date": "2026-04-13", "hk_cash_role": "stale", "hk_data_date": "2026-04-10"},
        overview={"theme": "Risk-on backdrop"},
        macro_agenda=[{"status": "Upcoming", "event": "US CPI", "impact": "Rates reaction."}],
        sector_digest=sector_digest,
        high_frequency=high_frequency,
        catalysts=catalysts,
        risk_data={"geopolitical_risks": [{"region": "Middle East", "event": "Oil risk", "impact": "Watch crude."}]},
    )
    assert "No fresh Hong Kong cash-market session" in non_trading["summary"]
    assert non_trading["still_moving"][0]["label"] == "VIX"
    assert non_trading["event_watch"][0]["signal"] == "VIX -3.50%"
    assert non_trading["next_open"][0].startswith("First dated catalyst")

    prompts = build_reflection_prompts(
        config,
        {"risk_regime": "Risk-On"},
        {"leadership": "Hong Kong growth / internet led"},
    )
    assert prompts[0].startswith("Does the overnight tape")
    assert prompts[-1] == "What would change my mind before lunch?"


def test_weekly_review_summarizes_cross_assets_and_hk_tape() -> None:
    summary = {
        "Equities": {
            "S&P 500": {"Price": 5000, "Pct Change": "1.20%"},
            "Nasdaq 100": {"Price": 18000, "Pct Change": "1.60%"},
            "Hang Seng Index": {"Price": 17450, "Pct Change": "0.80%"},
            "Hang Seng China Enterprises": {"Price": 6150, "Pct Change": "0.40%"},
            "Hang Seng TECH ETF": {"Price": 4.88, "Pct Change": "1.30%"},
            "China Large-Cap (FXI)": {"Price": 28, "Pct Change": "0.90%"},
        },
        "Rates": {"10Y Treasury": {"Price": 4.15, "Pct Change": "-0.60%"}},
        "FX": {"DXY": {"Price": 104.2, "Pct Change": "-0.35%"}, "USD/CNH": {"Price": 7.18, "Pct Change": "0.10%"}, "USD/HKD": {"Price": 7.82, "Pct Change": "0.00%"}},
        "Commodities": {"Gold": {"Price": 2350, "Pct Change": "0.90%"}},
        "Vol": {"VIX": {"Price": 16.2, "Pct Change": "-3.50%"}},
    }

    weekly = build_weekly_review(
        day_mode={"mode": "weekly_review", "period_start": "2026-04-06", "period_end": "2026-04-10"},
        date_semantics={"review_date": "2026-04-13"},
        overview={"theme": "Risk-on backdrop"},
        summary=summary,
        hk_desk_view={"leadership": "Hong Kong growth / internet led"},
        high_frequency=[],
        sector_digest={"graded_news": [{"grade": "A", "title": "Policy support", "why": "Helps sentiment."}]},
        macro_agenda=[{"status": "Upcoming", "event": "US CPI", "impact": "Rates reaction."}],
        catalysts=[{"date": "2026-04-14", "event": "US CPI", "impact": "Rates reaction."}],
        flow_tracker={"conclusion": "Flow evidence was supportive."},
        attribution={"dominant_drivers": [{"driver": "US growth-style transmission", "interpretation": "Growth led."}]},
    )

    assert weekly["window"]["start"] == "2026-04-06"
    assert "Risk-on backdrop" in weekly["summary"]
    assert weekly["cross_assets"][0]["asset"] == "S&P 500"
    assert any("3033.HK" in item["signal"] for item in weekly["hk_tape"])
    assert weekly["developments"][0]["item"] == "Policy support"
    assert weekly["next_week"][0]["event"] == "US CPI"
    assert weekly["flow_lines"][0] == "Flow evidence was supportive."
    assert weekly["desk_questions"]


if __name__ == "__main__":
    test_narrative_builds_theme_forward_and_non_trading_focus()
    test_weekly_review_summarizes_cross_assets_and_hk_tape()
    print("Analytics narrative test passed")
