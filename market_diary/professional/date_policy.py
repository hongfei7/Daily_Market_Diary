from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict
from zoneinfo import ZoneInfo

from market_diary.professional.market_holidays import (
    load_cn_holidays,
    load_hk_holidays,
    load_us_holidays,
)


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
    dates = set(str(item) for item in (calendar.get("closed_dates", []) or []))
    # Merge the static HKEX weekday-holiday table unless the config opts out.
    if calendar.get("use_hk_holidays", True):
        holiday_dates, _ = load_hk_holidays()
        dates |= holiday_dates
    return dates


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


def _us_closed_dates(year: int) -> set[str]:
    return load_us_holidays(year)


def _cn_closed_dates(year: int) -> set[str]:
    dates, _ = load_cn_holidays(year)
    return dates


def is_us_trading_day(value: str) -> bool:
    day = _parse_date(value)
    return day.weekday() < 5 and value not in _us_closed_dates(day.year)


def is_cn_trading_day(value: str) -> bool:
    day = _parse_date(value)
    return day.weekday() < 5 and value not in _cn_closed_dates(day.year)


def previous_us_trading_day(briefing_date: str) -> str:
    current = _parse_date(briefing_date) - timedelta(days=1)
    for _ in range(14):
        if is_us_trading_day(current.isoformat()):
            return current.isoformat()
        current -= timedelta(days=1)
    return previous_weekday(briefing_date)


def previous_cn_trading_day(briefing_date: str) -> str:
    current = _parse_date(briefing_date) - timedelta(days=1)
    for _ in range(14):
        if is_cn_trading_day(current.isoformat()):
            return current.isoformat()
        current -= timedelta(days=1)
    return previous_weekday(briefing_date)


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
    # Each market resolves to its own last completed session:
    #   - global (US/Europe/FX/commodities) -> previous US trading day
    #   - Hong Kong / China local -> previous HK trading day
    #   - A-share (CSI 300 / Shanghai / ChiNext) -> previous China trading day
    # A US holiday (e.g. Thanksgiving) or a China holiday must not resolve to a
    # non-trading day. The data adapter still backs off to the actual last
    # session as a safety net.
    global_market_date = getattr(args, "global_date", "") or compatibility_date or previous_us_trading_day(briefing_date)
    hk_data_date = getattr(args, "hk_date", "") or compatibility_date or previous_hk_trading_day(briefing_date, config)
    cn_data_date = getattr(args, "cn_date", "") or compatibility_date or previous_cn_trading_day(briefing_date)
    return {
        "briefing_date": briefing_date,
        "review_date": review_date,
        "global_market_date": global_market_date,
        "hk_data_date": hk_data_date,
        "cn_data_date": cn_data_date,
    }


def build_report_mode(briefing_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Derive the report mode from the briefing day (today), not the reviewed day.

    A 05:17 HKT morning briefing's job is set by what *today* is:
      - Sunday      -> weekly review of the just-completed Mon-Fri week
      - Monday      -> week-ahead calendar, forecast, and watch list
      - Tue-Fri     -> trading-day review of the previous session
      - Saturday    -> trading-day review of Friday's completed session
      - weekday holiday -> holiday event watch / reopen playbook
    """
    day = _parse_date(briefing_date)
    last_trading_day = previous_hk_trading_day_on_or_before(
        (day - timedelta(days=1)).isoformat(), config
    )
    next_trading_day = next_hk_trading_day_after(briefing_date, config)
    target_hk_session = (
        briefing_date
        if is_hk_trading_day(briefing_date, config)
        else next_trading_day
    )
    base = {
        "briefing_date": briefing_date,
        "review_date": previous_calendar_day(briefing_date),
        "last_hk_trading_day": last_trading_day,
        "target_hk_session": target_hk_session,
        "next_hk_trading_day": next_trading_day,
    }

    if day.weekday() == 6:  # Sunday -> weekly review
        window = weekly_review_window(briefing_date, config)
        return {
            **base,
            **window,
            "mode": "weekly_review",
            "legacy_mode": "non_trading_day",
            "label": "Weekly Review",
            "is_trading_day": False,
            "report_horizon": "weekly",
            "note": "Synthesize the completed Hong Kong week into next-week preparation rather than treating Sunday as a fresh session.",
        }

    if day.weekday() == 0 and is_hk_trading_day(briefing_date, config):  # Monday trading day -> week ahead
        week_end = day + timedelta(days=4)
        return {
            **base,
            "mode": "week_ahead",
            "legacy_mode": "non_trading_day",
            "label": "Week Ahead",
            "is_trading_day": False,
            "report_horizon": "week_ahead",
            "week_start": briefing_date,
            "week_end": week_end.isoformat(),
            "note": "Use Friday's close as the baseline, lay out the week's calendar, and name the key things to watch.",
        }

    if is_hk_trading_day(briefing_date, config):  # Tue-Fri trading day
        yesterday = (day - timedelta(days=1)).isoformat()
        if not is_hk_trading_day(yesterday, config):
            return {
                **base,
                "mode": "holiday_reopen_playbook",
                "legacy_mode": "non_trading_day",
                "label": "Holiday Reopen Playbook",
                "is_trading_day": True,
                "report_horizon": "daily",
                "note": "Focus on what changed during the market closure and how to prepare for today's Hong Kong open.",
            }
        return {
            **base,
            "mode": "trading_daily",
            "legacy_mode": "trading_day",
            "label": "Trading Daily",
            "is_trading_day": True,
            "report_horizon": "daily",
            "note": "Keep the report execution-oriented: what mattered in the completed session, what can move Hong Kong leadership, and what needs fast follow-up.",
        }

    if day.weekday() == 5:  # Saturday -> review Friday's completed session
        return {
            **base,
            "mode": "trading_daily",
            "legacy_mode": "trading_day",
            "label": "Trading Daily",
            "is_trading_day": True,
            "report_horizon": "daily",
            "note": "Review Friday's completed session; treat Saturday as a non-trading prep day.",
        }

    # Weekday holiday (market closed today, reopen later)
    return {
        **base,
        "mode": "holiday_event_watch",
        "legacy_mode": "non_trading_day",
        "label": "Holiday Event Watch",
        "is_trading_day": False,
        "report_horizon": "holiday",
        "note": "Monitor policy, geopolitics, still-moving global assets, and company actions while Hong Kong cash markets are closed.",
    }


def build_day_mode(briefing_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    return build_report_mode(briefing_date, config)


def build_date_semantics(
    report_date: str,
    briefing_date: str,
    global_market_date: str,
    hk_data_date: str,
    market_meta: Dict[str, Any],
    day_mode: Dict[str, Any],
    cn_data_date: str = "",
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
        f"Global (US/Europe) data date is {global_market_date}; adapter effective date is {global_effective} and summary date is {summary_date}.",
        f"Hong Kong local data date is {hk_data_date}; role: {hk_cash_role}.",
    ]
    if cn_data_date:
        lines.append(f"A-share / mainland data date is {cn_data_date}.")
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
        "target_hk_session": (day_mode or {}).get(
            "target_hk_session",
            (day_mode or {}).get("next_hk_trading_day", ""),
        ),
        "next_hk_trading_day": (day_mode or {}).get("next_hk_trading_day", ""),
        "global_request_date": global_market_date,
        "global_effective_date": global_effective,
        "global_summary_date": summary_date,
        "global_role": global_role,
        "hk_data_date": hk_data_date,
        "hk_cash_role": hk_cash_role,
        "cn_data_date": cn_data_date,
        "is_trading_day": is_trading_day,
        "lines": lines,
    }
