from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict
from zoneinfo import ZoneInfo


def today_in_timezone(tz_name: str) -> str:
    try:
        return datetime.now(ZoneInfo(tz_name)).strftime("%Y-%m-%d")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d")


def previous_calendar_day(briefing_date: str) -> str:
    return (datetime.strptime(briefing_date, "%Y-%m-%d").date() - timedelta(days=1)).isoformat()


def previous_weekday(briefing_date: str) -> str:
    current = datetime.strptime(briefing_date, "%Y-%m-%d").date() - timedelta(days=1)
    for _ in range(7):
        if current.weekday() < 5:
            return current.isoformat()
        current -= timedelta(days=1)
    return previous_calendar_day(briefing_date)


def previous_hk_trading_day(briefing_date: str, config: Dict[str, Any]) -> str:
    current = datetime.strptime(briefing_date, "%Y-%m-%d").date() - timedelta(days=1)
    calendar = config.get("calendar", {}) or {}
    closed_weekdays = set(int(item) for item in (calendar.get("closed_weekdays", [5, 6]) or []))
    closed_dates = set(str(item) for item in (calendar.get("closed_dates", []) or []))

    for _ in range(14):
        if current.weekday() not in closed_weekdays and current.isoformat() not in closed_dates:
            return current.isoformat()
        current -= timedelta(days=1)

    return previous_calendar_day(briefing_date)


def resolve_report_dates(args: Any, config: Dict[str, Any]) -> Dict[str, str]:
    """Resolve briefing, review, global, and HK/China local data dates.

    Scheduled runs summarize the previous calendar day, while Hong Kong and
    China local cash-market adapters use the last completed local trading day.
    Explicit CLI dates keep their previous override behavior.
    """

    timezone = config.get("system", {}).get("timezone", "Asia/Shanghai")
    briefing_date = getattr(args, "briefing_date", "") or today_in_timezone(timezone)
    compatibility_date = getattr(args, "date", "") or ""
    review_date = getattr(args, "review_date", "") or compatibility_date or previous_calendar_day(briefing_date)
    global_market_date = getattr(args, "global_date", "") or compatibility_date or review_date
    hk_data_date = getattr(args, "hk_date", "") or compatibility_date or previous_hk_trading_day(briefing_date, config)
    return {
        "briefing_date": briefing_date,
        "review_date": review_date,
        "global_market_date": global_market_date,
        "hk_data_date": hk_data_date,
    }


def build_day_mode(report_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    day = datetime.strptime(report_date, "%Y-%m-%d")
    calendar = config.get("calendar", {}) or {}
    closed_weekdays = set(int(value) for value in (calendar.get("closed_weekdays", []) or []))
    closed_dates = set(str(value) for value in (calendar.get("closed_dates", []) or []))
    is_closed = day.weekday() in closed_weekdays or report_date in closed_dates

    if is_closed:
        return {
            "mode": "non_trading_day",
            "label": "Non-trading day",
            "is_trading_day": False,
            "note": "Treat the last available market tape as reference only; focus on policy, geopolitics, company actions, and next-session preparation.",
        }
    return {
        "mode": "trading_day",
        "label": "Trading day",
        "is_trading_day": True,
        "note": "Keep the report execution-oriented: what matters by the Hong Kong open, what can move leadership, and what needs fast follow-up.",
    }


def build_date_semantics(
    report_date: str,
    briefing_date: str,
    global_market_date: str,
    hk_data_date: str,
    market_meta: Dict[str, Any],
    day_mode: Dict[str, Any],
) -> Dict[str, Any]:
    is_trading_day = bool((day_mode or {}).get("is_trading_day", True))
    global_effective = (market_meta or {}).get("effective_date", global_market_date)
    summary_date = (market_meta or {}).get("summary_date", global_effective)
    hk_cash_role = (
        "same-session local cash tape"
        if is_trading_day and hk_data_date == report_date
        else "last completed HK/China cash-market reference tape"
    )
    global_role = (
        "completed global market session"
        if is_trading_day
        else "requested calendar day for still-moving global assets; stale cash markets remain reference-only"
    )
    lines = [
        f"Review date {report_date} is treated as `{(day_mode or {}).get('label', 'Trading day')}`.",
        f"Global request date is {global_market_date}; adapter effective date is {global_effective} and summary date is {summary_date}.",
        f"HK/China local data date is {hk_data_date}; role: {hk_cash_role}.",
    ]
    return {
        "briefing_date": briefing_date,
        "review_date": report_date,
        "global_request_date": global_market_date,
        "global_effective_date": global_effective,
        "global_summary_date": summary_date,
        "global_role": global_role,
        "hk_data_date": hk_data_date,
        "hk_cash_role": hk_cash_role,
        "is_trading_day": is_trading_day,
        "lines": lines,
    }
