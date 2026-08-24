"""Trading-holiday calendars for Hong Kong, the US, and mainland China (A-shares).

A global morning briefing must know, for each market, when a *weekday* is
actually a closure so that "previous trading day" resolves correctly. Weekends
are handled by ``calendar.closed_weekdays``; these tables supply weekday
closures per market.

- Hong Kong: hard-coded HKEX closures (lunar dates shift yearly; verify annually).
- United States: rule-based NYSE holidays (fixed rules + the Easter formula), so
  they need no annual refresh.
- Mainland China (A-shares): hard-coded SSE/SZSE closures; the Spring Festival /
  National Day windows and the "调休" (weekend-makeup) days shift yearly and
  MUST be re-verified against the SSE/SZSE annual notice.

The expiry guard turns a forgotten refresh of the hard-coded tables into a loud
warning, not a silently wrong calendar.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, Set, Tuple

# ---------------------------------------------------------------------------
# Hong Kong (HKEX)
# ---------------------------------------------------------------------------

_HK_HOLIDAYS: Dict[int, Set[str]] = {
    2025: {
        "2025-01-01",  # New Year's Day (Wed)
        "2025-01-29", "2025-01-30", "2025-01-31",  # Lunar New Year (Wed-Fri)
        "2025-04-04",  # Good Friday
        "2025-04-07",  # Easter Monday
        "2025-05-01",  # Labour Day
        "2025-05-30",  # Tuen Ng (Fri)
        "2025-07-01",  # HKSAR Establishment Day (Tue)
        "2025-10-01",  # National Day (Wed)
        "2025-12-25", "2025-12-26",  # Christmas + Boxing Day (Thu/Fri)
    },
    2026: {
        "2026-01-01",  # New Year's Day (Thu)
        "2026-02-16", "2026-02-17", "2026-02-18",  # Lunar New Year (Mon-Wed)
        "2026-04-03",  # Good Friday (Fri)
        "2026-04-06",  # Easter Monday (Mon)
        "2026-05-01",  # Labour Day (Fri)
        "2026-06-19",  # Tuen Ng / Dragon Boat (Fri)
        "2026-07-01",  # HKSAR Establishment Day (Wed)
        "2026-10-01",  # National Day (Thu)
        "2026-12-25",  # Christmas (Fri)
        # TODO(verify): Buddha's Birthday (~2026-05-24 Sun -> possible 05-25 Mon),
        #               Mid-Autumn "day following" (~2026-09-26 Sat, no weekday),
        #               Chung Yeung (~2026-10-18 Sun -> possible 10-19 Mon).
    },
}

# ---------------------------------------------------------------------------
# Mainland China (SSE/SZSE A-share) — weekday closures, best-effort for 2026.
# Spring Festival / National Day windows include the surrounding "调休" makeup
# workdays in the real calendar; re-verify against the annual SSE/SZSE notice.
# ---------------------------------------------------------------------------

_CN_HOLIDAYS: Dict[int, Set[str]] = {
    2026: {
        "2026-01-01",  # New Year's Day (Thu)
        "2026-02-16", "2026-02-17", "2026-02-18", "2026-02-19", "2026-02-20",  # Spring Festival (Mon-Fri)
        "2026-04-06",  # Qingming (Mon, observed)
        "2026-05-01", "2026-05-04", "2026-05-05",  # Labour Day (Fri, Mon, Tue)
        "2026-06-19",  # Dragon Boat (Fri)
        "2026-09-25",  # Mid-Autumn (Fri)
        "2026-10-01", "2026-10-02", "2026-10-05", "2026-10-06", "2026-10-07",  # National Day (Thu-Fri, Mon-Wed)
    },
}

# Latest year with a populated hard-coded table.
MAX_YEAR = 2026


# ---------------------------------------------------------------------------
# United States (NYSE) — rule-based, no annual refresh needed.
# ---------------------------------------------------------------------------

def _easter_sunday(year: int) -> date:
    """Meeus/Jones/Butcher Gregorian Easter algorithm."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """The ``n``-th ``weekday`` (0=Mon..6=Sun) of ``month``."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + (n - 1) * 7)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """The last ``weekday`` of ``month``."""
    if month == 12:
        last_day = date(year, 12, 31)
    else:
        last_day = date(year, month + 1, 1) - timedelta(days=1)
    return last_day - timedelta(days=(last_day.weekday() - weekday) % 7)


def _observed(year: int, month: int, day: int) -> date:
    """NYSE observed-holiday rule: Saturday -> prior Friday, Sunday -> next Monday."""
    d = date(year, month, day)
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def us_market_holidays(year: int) -> Set[str]:
    """NYSE market-holiday set for ``year`` (ISO weekday dates)."""
    out: Set[str] = set()
    out.add(_observed(year, 1, 1).isoformat())              # New Year's Day
    out.add(_nth_weekday(year, 1, 0, 3).isoformat())        # Martin Luther King Jr. Day
    out.add(_nth_weekday(year, 2, 0, 3).isoformat())        # Presidents' Day
    easter = _easter_sunday(year)
    out.add((easter - timedelta(days=2)).isoformat())       # Good Friday
    out.add(_last_weekday(year, 5, 0).isoformat())          # Memorial Day
    out.add(_observed(year, 6, 19).isoformat())             # Juneteenth
    out.add(_observed(year, 7, 4).isoformat())              # Independence Day
    out.add(_nth_weekday(year, 9, 0, 1).isoformat())        # Labor Day
    out.add(_nth_weekday(year, 11, 3, 4).isoformat())       # Thanksgiving
    out.add(_observed(year, 12, 25).isoformat())            # Christmas
    return out


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def load_hk_holidays(year: int | None = None) -> Tuple[Set[str], int]:
    target = int(year) if year is not None else MAX_YEAR
    return set(_HK_HOLIDAYS.get(target, set())), MAX_YEAR


def load_cn_holidays(year: int | None = None) -> Tuple[Set[str], int]:
    target = int(year) if year is not None else MAX_YEAR
    return set(_CN_HOLIDAYS.get(target, set())), MAX_YEAR


def load_us_holidays(year: int | None = None) -> Set[str]:
    """US holidays are computed, so any year is valid (no expiry guard needed)."""
    return us_market_holidays(int(year) if year is not None else MAX_YEAR)


def hk_holidays_available() -> Tuple[Set[str], int]:
    """Backward-compatible alias (kept for the previous import site)."""
    return load_hk_holidays()
