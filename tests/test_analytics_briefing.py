from _bootstrap import ROOT  # noqa: F401

from professional.analytics_briefing import (
    build_catalyst_calendar,
    build_company_event_digest,
    build_must_watch,
    build_source_links,
)


def test_briefing_builds_catalysts_links_and_must_watch() -> None:
    macro_agenda = [
        {
            "time": "20:30",
            "event": "US CPI",
            "status": "Upcoming",
            "impact": "Rates-sensitive read",
            "attention": 5,
            "score": 90,
            "affected_industries": ["Internet"],
        },
        {"time": "08:00", "event": "Old data", "status": "Released", "score": 10},
    ]
    sector_data = {
        "earnings_calendar": [
            {
                "date": "2026-04-15",
                "ticker": "AAPL",
                "company": "Apple",
                "time": "After close",
                "eps_estimate": "1.45",
                "revenue_estimate": "89.5B",
            },
            {"date": "2026-04-30", "ticker": "LATE", "company": "LateCo"},
        ],
        "hkex_announcements": {
            "data": {
                "top_announcements": [{"ticker": "0005.HK", "title": "Fallback announcement"}],
                "watchlist_hits": [
                    {
                        "grade": "A",
                        "ticker": "0700.HK",
                        "company": "Tencent",
                        "event_type": "Results",
                        "title": "Annual results",
                        "release_time": "2026-04-13 18:30",
                        "source": "HKEXnews",
                        "url": "https://example.com/hkex.pdf",
                        "score": 5.0,
                    }
                ],
            },
            "meta": {"source": "HKEXnews"},
        },
    }
    risk_data = {"upcoming_events": [{"date": "2026-04-14", "description": "Monthly expiry", "importance": "high"}]}
    watchlists = {
        "Core coverage": [
            {
                "name": "Tencent",
                "bucket": "Core coverage",
                "thesis": "Platform recovery",
                "upcoming_catalyst": "Earnings",
                "catalyst_date": "2026-04-16",
                "recent_news": [
                    {"title": "Tencent buyback", "url": "https://example.com/tencent", "source": "Mock"},
                    {"title": "Duplicate URL", "url": "https://example.com/cpi", "source": "Mock"},
                ],
            }
        ]
    }
    config = {"report": {"catalyst_window_days": 3}}

    catalysts = build_catalyst_calendar("2026-04-13", macro_agenda, sector_data, risk_data, watchlists, config)

    assert [item["event"] for item in catalysts] == [
        "US CPI",
        "Monthly expiry",
        "Apple earnings",
        "Tencent: Earnings",
    ]
    assert "LateCo earnings" not in [item["event"] for item in catalysts]

    sector_digest = {
        "graded_news": [
            {"title": "US CPI", "url": "https://example.com/cpi", "source": "Wire", "grade": "A", "why": "Rates", "score": 4.8},
            {"title": "Duplicate", "url": "https://example.com/cpi", "source": "Wire"},
        ],
        "sell_side": [{"ticker": "NVDA", "action": "Upgrade"}],
    }
    company_events = build_company_event_digest(sector_data, sector_digest)
    source_links = build_source_links(
        sector_digest,
        watchlists,
        {"top_news_items": 1, "watchlist_story_limit": 2, "top_source_links": 3},
        company_events=company_events,
    )

    assert company_events["announcements"][0]["ticker"] == "0700.HK"
    assert company_events["ratings"][0]["ticker"] == "NVDA"
    assert [item["url"] for item in source_links] == [
        "https://example.com/cpi",
        "https://example.com/tencent",
        "https://example.com/hkex.pdf",
    ]

    must_watch = build_must_watch(
        overview={"theme": "Risk-on backdrop", "risk_regime": "risk-on"},
        macro_agenda=macro_agenda,
        sector_digest=sector_digest,
        high_frequency=[{"label": "VIX", "change_pct": -3.5, "interpretation": "Vol down", "category": "Vol"}],
        movers_digest={"movers": [{"title": "NVDA +3.50%", "attribution": "Announcement", "summary": "AI order", "score": 5.0}]},
        catalysts=catalysts,
        report_config={
            "quick_items_limit": 8,
            "top_macro_events": 1,
            "top_news_items": 1,
            "top_high_frequency_items": 1,
            "top_movers": 1,
            "top_catalysts": 2,
        },
        day_mode={"is_trading_day": True},
    )

    titles = [item["title"] for item in must_watch]
    assert "Risk-on backdrop" in titles
    assert "US CPI (Upcoming)" in titles
    assert "VIX -3.50%" in titles
    assert "NVDA +3.50%" in titles
    assert len(titles) == len(set(titles))


if __name__ == "__main__":
    test_briefing_builds_catalysts_links_and_must_watch()
    print("Analytics briefing test passed")
