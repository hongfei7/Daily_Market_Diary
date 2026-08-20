from _bootstrap import ROOT  # noqa: F401

from professional.analytics_briefing import (
    build_catalyst_calendar,
    build_company_event_digest,
    build_must_watch,
    build_source_links,
)
from professional.report_blocks import _render_company_events


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


def test_company_event_monitor_aggregates_low_signal_market_warnings() -> None:
    sector_data = {
        "earnings_calendar": [],
        "earnings_calendar_status": "unavailable",
        "analyst_changes_status": "unavailable",
        "hkex_announcements": {
            "status": "ok",
            "data": {
                "profit_warnings": [{"ticker": "00229.HK"}, {"ticker": "00474.HK"}],
                "results_announcements": [],
                "trading_halts": [],
                "watchlist_hits": [],
                "top_announcements": [
                    {
                        "grade": "C",
                        "ticker": "00229.HK",
                        "company": "Raymond Industrial",
                        "event_type": "Profit warning",
                        "title": "Profit warning / alert announcement",
                        "release_time": "2026-08-03",
                        "source": "HKEXnews",
                        "url": "https://example.com/229.pdf",
                        "score": 2.5,
                        "watchlist_match": False,
                    },
                    {
                        "grade": "C",
                        "ticker": "00474.HK",
                        "company": "Market Company",
                        "event_type": "Profit warning",
                        "title": "Profit warning / alert announcement",
                        "release_time": "2026-08-03",
                        "source": "HKEXnews",
                        "url": "https://example.com/474.pdf",
                        "score": 2.5,
                        "watchlist_match": False,
                    },
                ],
            },
            "meta": {"source": "HKEXnews", "available_count": 2},
        },
    }

    company_events = build_company_event_digest(sector_data, {"sell_side": []})
    rendered = _render_company_events({"company_events": company_events, "llm_sections": {}})

    assert company_events["event_summary"]["official_filings"] == 2
    assert company_events["event_summary"]["watchlist_hits"] == 0
    assert "No immediate portfolio catalyst" in rendered
    assert "2 profit warnings" in rendered
    assert "No event cleared the portfolio decision filter" in rendered
    assert "00229.HK" not in rendered
    assert "IPO and grey-market monitoring is not yet" not in rendered


def test_company_event_monitor_expands_portfolio_filing() -> None:
    filing = {
        "grade": "A",
        "ticker": "0700.HK",
        "company": "Tencent",
        "event_type": "Profit warning",
        "title": "Profit warning / alert announcement",
        "release_time": "2026-08-03",
        "source": "HKEXnews",
        "url": "https://example.com/700.pdf",
        "score": 5.5,
        "watchlist_match": True,
        "filing_detail_status": "parsed",
        "filing_extract": "The Group is expected to record a loss of HK$100 million.",
        "filing_drivers": "The change was mainly attributable to a one-off impairment.",
        "next_disclosure": "Final results are expected to be published on 20 August 2026.",
    }
    sector_data = {
        "earnings_calendar": [],
        "earnings_calendar_status": "unavailable",
        "analyst_changes_status": "unavailable",
        "hkex_announcements": {
            "status": "ok",
            "data": {
                "profit_warnings": [filing],
                "results_announcements": [],
                "trading_halts": [],
                "watchlist_hits": [filing],
                "top_announcements": [filing],
            },
            "meta": {"source": "HKEXnews", "available_count": 1},
        },
    }

    company_events = build_company_event_digest(sector_data, {"sell_side": []})
    rendered = _render_company_events({"company_events": company_events, "llm_sections": {}})

    assert "Portfolio attention required" in rendered
    assert "Tencent · 0700.HK" in rendered
    assert "expected to record a loss" in rendered
    assert "one-off impairment" in rendered
    assert "Investor read" in rendered
    assert "Next check" in rendered


if __name__ == "__main__":
    test_briefing_builds_catalysts_links_and_must_watch()
    print("Analytics briefing test passed")


def test_same_release_from_two_feeds_appears_once() -> None:
    """The macro calendar and the risk feed share a release schedule.

    The same event arrived as "China LPR (1Y / 5Y) (Upcoming)" from the macro
    agenda and "CN China LPR (1Y / 5Y)" from the catalyst feed, so it took two
    of the four checklist slots.
    """
    from professional.analytics_briefing import _dedupe_key

    assert _dedupe_key("China LPR (1Y / 5Y) (Upcoming)") == _dedupe_key("CN China LPR (1Y / 5Y)")
    assert _dedupe_key("US CPI (Released)") == _dedupe_key("US CPI")

    # A graded news story is analysis, not the calendar entry, and stays distinct.
    assert _dedupe_key("[A] US CPI") != _dedupe_key("US CPI (Upcoming)")

    # Different events must not collapse.
    assert _dedupe_key("China LPR (1Y / 5Y)") != _dedupe_key("Hong Kong CPI")


def test_radar_collapses_the_same_event_from_different_feeds() -> None:
    """The macro calendar and risk feed share a release schedule.

    The radar aggregates six sources and keyed dedupe on the raw event string,
    so "China LPR (1Y / 5Y)" and "CN China LPR (1Y / 5Y)" both survived and each
    release took two of the five queue slots.
    """
    from professional.catalyst_radar import _dedupe_rows

    rows = [
        {"event": "China LPR (1Y / 5Y)", "date": "2026-08-20", "entity": "CN"},
        {"event": "CN China LPR (1Y / 5Y)", "date": "2026-08-20", "entity": ""},
        {"event": "Hong Kong CPI", "date": "2026-08-21", "entity": ""},
        {"event": "HK Hong Kong CPI", "date": "2026-08-21", "entity": "HK"},
        # A genuinely different date is a different event.
        {"event": "China LPR (1Y / 5Y)", "date": "2026-09-20", "entity": "CN"},
    ]
    out = _dedupe_rows(rows)
    assert len(out) == 3
    assert [row["date"] for row in out] == ["2026-08-20", "2026-08-21", "2026-09-20"]


def test_macro_rows_carry_the_event_date_not_the_report_date() -> None:
    """An event two days out rendered as if it were today.

    The agenda pinned every row to the report date and put the event date in
    "time", so the radar rendered "2026-08-20 2026-08-21".
    """
    from modules.macro_calendar import fetch_macro_data
    from professional.analytics_macro import build_macro_agenda
    from professional.config import load_professional_config

    agenda = build_macro_agenda("2026-08-20", fetch_macro_data("2026-08-20"), load_professional_config())
    assert agenda, "expected at least one scheduled release in the window"
    for row in agenda:
        assert row["time"] == "", "a date must never be rendered as a time"
    assert any(row["date"] != "2026-08-20" for row in agenda), "event dates should not all collapse to today"
