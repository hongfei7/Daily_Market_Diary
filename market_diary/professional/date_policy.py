from __future__ import annotations

from datetime import date, datetime, timedelta
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


def _calendar_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return config.get("calendar", {}) or {}


def _closed_weekdays(config: Dict[str, Any]) -> set[int]:
    calendar = _calendar_config(config)
    return set(int(item) for item in (calendar.get("closed_weekdays", [5, 6]) or []))


def _closed_dates(config: Dict[str, Any]) -> set[str]:
    calendar = _calendar_config(config)
    return set(str(item) for item in (calendar.get("closed_dates", []) or []))


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def is_hk_trading_day(value: str, config: Dict[str, Any]) -> bool:
    day = _parse_date(value)
    return day.weekday() not in _closed_weekdays(config) and value not in _closed_dates(config)


def previous_hk_trading_day(briefing_date: str, config: Dict[str, Any]) -> str:
    current = _parse_date(briefing_date) - timedelta(days=1)

    for _ in range(14):
        if is_hk_trading_day(current.isoformat(), config):
            return current.isoformat()
        current -= timedelta(days=1)

    return previous_calendar_day(briefing_date)


def previous_hk_trading_day_on_or_before(value: str, config: Dict[str, Any]) -> str:
    current = _parse_date(value)
    for _ in range(14):
        if is_hk_trading_day(current.isoformat(), config):
            return current.isoformat()
        current -= timedelta(days=1)
    return value


def next_hk_trading_day_after(value: str, config: Dict[str, Any]) -> str:
    current = _parse_date(value) + timedelta(days=1)
    for _ in range(14):
        if is_hk_trading_day(current.isoformat(), config):
            return current.isoformat()
        current += timedelta(days=1)
    return value


def weekly_review_window(report_date: str, config: Dict[str, Any]) -> Dict[str, str]:
    period_end = previous_hk_trading_day_on_or_before(report_date, config)
    end_day = _parse_date(period_end)
    week_start = end_day - timedelta(days=end_day.weekday())
    current = week_start
    period_start = period_end
    while current <= end_day:
        candidate = current.isoformat()
        if is_hk_trading_day(candidate, config):
            period_start = candidate
            break
        current += timedelta(days=1)
    return {"period_start": period_start, "period_end": period_end}


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


def build_report_mode(report_date: str, config: Dict[str, Any], briefing_date: str = "") -> Dict[str, Any]:
    day = _parse_date(report_date)
    next_trading_day = next_hk_trading_day_after(report_date, config)
    last_trading_day = previous_hk_trading_day_on_or_before(report_date, config)
    base = {
        "review_date": report_date,
        "last_hk_trading_day": last_trading_day,
        "next_hk_trading_day": next_trading_day,
    }

    if is_hk_trading_day(report_date, config):
        return {
            **base,
            "mode": "trading_daily",
            "legacy_mode": "trading_day",
            "label": "Trading Daily",
            "is_trading_day": True,
            "report_horizon": "daily",
            "note": "Keep the report execution-oriented: what mattered in the completed session, what can move Hong Kong leadership, and what needs fast follow-up.",
        }

    if day.weekday() == 5:
        window = weekly_review_window(report_date, config)
        return {
            **base,
            **window,
            "mode": "weekly_review",
            "legacy_mode": "non_trading_day",
            "label": "Weekly Review",
            "is_trading_day": False,
            "report_horizon": "weekly",
            "note": "Use Saturday's non-trading review date to synthesize the completed Hong Kong week and prepare next week's work plan.",
        }

    if day.weekday() == 6:
        return {
            **base,
            "mode": "non_trading_event_watch",
            "legacy_mode": "non_trading_day",
            "label": "Weekend Event Watch",
            "is_trading_day": False,
            "report_horizon": "weekend",
            "note": "Track weekend policy, geopolitics, still-moving assets, and Monday open preparation rather than replaying stale cash-market tape.",
        }

    if briefing_date and next_trading_day == briefing_date:
        mode = "holiday_reopen_playbook"
        label = "Holiday Reopen Playbook"
        note = "Focus on what changed during the market closure and how to prepare for the next Hong Kong open."
    else:
        mode = "holiday_event_watch"
        label = "Holiday Event Watch"
        note = "Monitor policy, geopolitics, still-moving global assets, and company actions while Hong Kong cash markets are closed."

    return {
        **base,
        "mode": mode,
        "legacy_mode": "non_trading_day",
        "label": label,
        "is_trading_day": False,
        "report_horizon": "holiday",
        "note": note,
    }


def build_day_mode(report_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    return build_report_mode(report_date, config)


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
        f"Review date {report_date} is treated as `{(day_mode or {}).get('label', 'Trading Daily')}`.",
        f"Global request date is {global_market_date}; adapter effective date is {global_effective} and summary date is {summary_date}.",
        f"HK/China local data date is {hk_data_date}; role: {hk_cash_role}.",
    ]
    if (day_mode or {}).get("mode") == "weekly_review":
        lines.append(
            f"Weekly review window is {(day_mode or {}).get('period_start', '')} to {(day_mode or {}).get('period_end', '')}."
        )
    return {
        "briefing_date": briefing_date,
        "review_date": report_date,
        "report_mode": (day_mode or {}).get("mode", ""),
        "report_label": (day_mode or {}).get("label", ""),
        "report_horizon": (day_mode or {}).get("report_horizon", ""),
        "period_start": (day_mode or {}).get("period_start", ""),
        "period_end": (day_mode or {}).get("period_end", ""),
        "last_hk_trading_day": (day_mode or {}).get("last_hk_trading_day", ""),
        "next_hk_trading_day": (day_mode or {}).get("next_hk_trading_day", ""),
        "global_request_date": global_market_date,
        "global_effective_date": global_effective,
        "global_summary_date": summary_date,
        "global_role": global_role,
        "hk_data_date": hk_data_date,
        "hk_cash_role": hk_cash_role,
        "is_trading_day": is_trading_day,
        "lines": lines,
    }
