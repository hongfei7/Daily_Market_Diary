from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from market_diary.professional.analytics_market import _format_signed


def _clears_main_news_gate(news: Dict[str, Any]) -> bool:
    grade = news.get("grade")
    sector = str(news.get("sector", "") or "").lower()
    score = float(news.get("score", 0) or 0)
    return grade in {"A", "B"} and (sector != "other" or score >= 4.0)


def build_catalyst_calendar(
    report_date: str,
    macro_agenda: List[Dict[str, Any]],
    sector_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    watchlists: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    catalysts: List[Dict[str, Any]] = []
    report_config = (config or {}).get("report", {}) or {}
    window_days = int(report_config.get("catalyst_window_days", 7))
    base_date = datetime.strptime(report_date, "%Y-%m-%d")
    cutoff_date = base_date + timedelta(days=window_days)

    for item in macro_agenda:
        if item.get("status") in {"Upcoming", "Central bank"}:
            catalysts.append(
                {
                    "date": report_date,
                    "time": item.get("time", ""),
                    "event": item.get("event", ""),
                    "category": item.get("status", ""),
                    "impact": item.get("impact", ""),
                    "importance": item.get("attention", 3),
                    "score": item.get("score", 0),
                }
            )

    for item in (sector_data or {}).get("earnings_calendar", []) or []:
        earnings_date = item.get("date") or report_date
        catalysts.append(
            {
                "date": earnings_date,
                "time": item.get("time", ""),
                "event": f"{item.get('company', item.get('ticker', ''))} earnings",
                "category": "Earnings",
                "impact": f"EPS est. {item.get('eps_estimate')} / revenue est. {item.get('revenue_estimate')}",
                "importance": 4,
                "score": 72,
            }
        )

    for item in (risk_data or {}).get("upcoming_events", []) or []:
        catalysts.append(
            {
                "date": item.get("date", report_date),
                "time": "",
                "event": item.get("description", ""),
                "category": item.get("type", "Event"),
                "impact": "Watch whether it changes risk budgets or the theme-trading cadence",
                "importance": {"critical": 5, "high": 4, "medium": 3}.get(item.get("importance"), 2),
                "score": {"critical": 85, "high": 76, "medium": 65}.get(item.get("importance"), 55),
            }
        )

    for bucket_items in watchlists.values():
        for item in bucket_items:
            catalyst = item.get("upcoming_catalyst")
            if not catalyst:
                continue
            catalyst_date = item.get("catalyst_date") or report_date
            catalysts.append(
                {
                    "date": catalyst_date,
                    "time": "",
                    "event": f"{item.get('name')}: {catalyst}",
                    "category": item.get("bucket", "Watchlist"),
                    "impact": item.get("thesis", ""),
                    "importance": 3,
                    "score": 60,
                }
            )

    def sort_key(item: Dict[str, Any]) -> Tuple[datetime, str]:
        raw_date = item.get("date") or report_date
        try:
            parsed = datetime.strptime(raw_date, "%Y-%m-%d")
        except ValueError:
            parsed = datetime.strptime(report_date, "%Y-%m-%d")
        return parsed, str(item.get("time", ""))

    filtered: List[Dict[str, Any]] = []
    for item in catalysts:
        raw_date = item.get("date") or report_date
        try:
            parsed = datetime.strptime(raw_date, "%Y-%m-%d")
        except ValueError:
            parsed = base_date
        if base_date <= parsed <= cutoff_date:
            filtered.append(item)

    filtered.sort(key=lambda item: (sort_key(item)[0], sort_key(item)[1], -float(item.get("score", 0))))
    return filtered


def build_source_links(
    sector_digest: Dict[str, Any],
    watchlists: Dict[str, List[Dict[str, Any]]],
    report_config: Dict[str, Any],
    company_events: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    seen = set()
    news_limit = int((report_config or {}).get("top_news_items", 8))
    story_limit = int((report_config or {}).get("watchlist_story_limit", 2))
    total_limit = int((report_config or {}).get("top_source_links", 15))

    for news in [item for item in sector_digest.get("graded_news", []) if _clears_main_news_gate(item)][: max(news_limit, 1) + 4]:
        url = news.get("url")
        if url and url not in seen:
            seen.add(url)
            links.append({"label": news.get("title", ""), "url": url, "source": news.get("source", "")})

    for bucket_items in watchlists.values():
        for item in bucket_items:
            for news in item.get("recent_news", [])[:story_limit]:
                url = news.get("url")
                if url and url not in seen:
                    seen.add(url)
                    links.append({"label": news.get("title", ""), "url": url, "source": news.get("source", "")})

    for item in ((company_events or {}).get("announcements", []) or [])[:8]:
        url = item.get("url")
        if url and url not in seen:
            seen.add(url)
            links.append(
                {
                    "label": f"{item.get('ticker', '')} {item.get('title', '')}",
                    "url": url,
                    "source": item.get("source", "HKEXnews"),
                }
            )

    return links[:total_limit]


def build_must_watch(
    overview: Dict[str, Any],
    macro_agenda: List[Dict[str, Any]],
    sector_digest: Dict[str, Any],
    high_frequency: List[Dict[str, Any]],
    movers_digest: Dict[str, Any],
    catalysts: List[Dict[str, Any]],
    report_config: Dict[str, Any],
    day_mode: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    quick_limit = int((report_config or {}).get("quick_items_limit", 10))
    top_macro = int((report_config or {}).get("top_macro_events", 4))
    top_news = int((report_config or {}).get("top_news_items", 4))
    top_trackers = int((report_config or {}).get("top_high_frequency_items", 3))
    top_movers = int((report_config or {}).get("top_movers", 2))
    top_catalysts = int((report_config or {}).get("top_catalysts", 3))
    is_trading_day = bool((day_mode or {}).get("is_trading_day", True))

    items: List[Dict[str, Any]] = [
        {
            "bucket": "Overnight regime" if is_trading_day else "Still-moving markets",
            "title": overview.get("theme", ""),
            "summary": (
                f"Start by deciding whether the day is about {overview.get('risk_regime')} conditions or a style pivot."
                if is_trading_day
                else "No fresh Hong Kong cash session is assumed; use global active assets and policy/event flow to prepare the next open."
            ),
            "score": 95,
        }
    ]

    for event in macro_agenda[:top_macro]:
        items.append(
            {
                "bucket": "Macro / policy",
                "title": f"{event.get('event')} ({event.get('status')})",
                "summary": f"{event.get('impact')} | Industries: {', '.join(event.get('affected_industries', []))}",
                "score": event.get("score", 0),
            }
        )

    for news in [item for item in sector_digest.get("graded_news", []) if _clears_main_news_gate(item)][:top_news]:
        items.append(
            {
                "bucket": "News / announcements",
                "title": f"[{news.get('grade')}] {news.get('title')}",
                "summary": news.get("why", ""),
                "score": int(news.get("score", 0) * 20),
            }
        )

    tracker_candidates = high_frequency
    if not is_trading_day:
        active_categories = {"FX", "Commodities", "Crypto", "Rates", "Vol"}
        tracker_candidates = [item for item in high_frequency if item.get("category") in active_categories]

    for tracker in tracker_candidates[:top_trackers]:
        items.append(
            {
                "bucket": "High-frequency data",
                "title": f"{tracker.get('label')} {_format_signed(tracker.get('change_pct'))}",
                "summary": tracker.get("interpretation", ""),
                "score": int(abs(tracker.get("change_pct") or 0) * 10) + 40,
            }
        )

    mover_candidates = movers_digest.get("movers", []) if is_trading_day else []
    for mover in mover_candidates[:top_movers]:
        items.append(
            {
                "bucket": "Mover attribution",
                "title": mover.get("title", ""),
                "summary": f"{mover.get('attribution')} | {mover.get('summary')}",
                "score": int(mover.get("score", 0) * 10) + 30,
            }
        )

    for catalyst in catalysts[:top_catalysts]:
        items.append(
            {
                "bucket": "Catalysts",
                "title": catalyst.get("event", ""),
                "summary": catalyst.get("impact", ""),
                "score": catalyst.get("score", 0),
            }
        )

    items.sort(key=lambda item: item.get("score", 0), reverse=True)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        title = item.get("title")
        if title in seen:
            continue
        seen.add(title)
        deduped.append(item)
        if len(deduped) >= quick_limit:
            break
    return deduped


def build_company_event_digest(sector_data: Dict[str, Any], sector_digest: Dict[str, Any]) -> Dict[str, Any]:
    earnings_rows: List[Dict[str, Any]] = []
    for item in (sector_data or {}).get("earnings_calendar", []) or []:
        earnings_rows.append(
            {
                "ticker": item.get("ticker", ""),
                "company": item.get("company", ""),
                "time": item.get("time", ""),
                "comparison": f"EPS est. {item.get('eps_estimate', 'N/A')} | revenue est. {item.get('revenue_estimate', 'N/A')}",
            }
        )

    hkex_announcements = ((sector_data or {}).get("hkex_announcements", {}) or {}).get("data", {}) or {}
    announcement_rows: List[Dict[str, Any]] = []
    for item in (hkex_announcements.get("watchlist_hits", []) or hkex_announcements.get("top_announcements", []) or [])[:10]:
        announcement_rows.append(
            {
                "grade": item.get("grade", ""),
                "ticker": item.get("ticker", ""),
                "company": item.get("company", ""),
                "event_type": item.get("event_type", ""),
                "title": item.get("title", ""),
                "release_time": item.get("release_time", ""),
                "source": item.get("source", "HKEXnews"),
                "url": item.get("url", ""),
                "score": item.get("score", 0),
            }
        )

    return {
        "earnings": earnings_rows,
        "ratings": (sector_digest or {}).get("sell_side", []) or [],
        "announcements": announcement_rows,
        "hkex_meta": ((sector_data or {}).get("hkex_announcements", {}) or {}).get("meta", {}) or {},
        "ipo_watch": "IPO and grey-market monitoring is not yet part of the standard production pack.",
    }
