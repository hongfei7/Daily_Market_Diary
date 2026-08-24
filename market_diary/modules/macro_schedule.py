"""Rule-driven macro release schedule, organised by transmission channel to Hong Kong.

This is deliberately not a generic economic calendar. Most global releases never
move Hong Kong, and a generic feed produces a section that is empty on most days
while implying the calendar was quiet. Every entry here states the channel
through which it reaches Hong Kong equities:

``fed_path``       Under the peg, HKMA follows the Fed, so US data reaches Hong
                   Kong through HIBOR and equity duration.
``china_demand``   Directly sets H-share earnings; higher weight for Hong Kong
                   than US data despite lower headline attention.
``china_policy``   Rates and liquidity expectations, hence valuation.
``hk_liquidity``   Local funding conditions.

Chinese and Hong Kong statistical releases follow stable monthly rules, so the
schedule is generated rather than scraped: there is no fragile page to break, and
the rule is auditable. Dates are *scheduled* dates; a release that slips is
reported as scheduled rather than invented as released.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any, Dict, List, Optional

CHANNELS = {
    "fed_path": "Fed path -> HKMA -> HIBOR -> HK equity duration",
    "china_demand": "China demand -> H-share earnings",
    "china_policy": "China policy -> rates and valuation",
    "hk_liquidity": "Hong Kong funding conditions",
}

# Recurring monthly releases with stable publication rules.
# ``day`` is the usual calendar day of the month; ``window`` is the tolerance in
# days that the report communicates rather than pretending to exact timing.
MONTHLY_RULES: List[Dict[str, Any]] = [
    {
        "indicator": "China LPR (1Y / 5Y)",
        "source_url": "http://www.pbc.gov.cn/en/3688229/3688335/index.html",
        "country": "CN",
        "day": 20,
        "window": 0,
        "channel": "china_policy",
        "impact": "high",
        "note": "Sets the mortgage and corporate lending benchmark; the cleanest read on easing intent.",
    },
    {
        "indicator": "China NBS Manufacturing PMI",
        "source_url": "https://www.stats.gov.cn/english/PressRelease/",
        "country": "CN",
        "day": 31,
        "window": 1,
        "channel": "china_demand",
        "impact": "high",
        "note": "First hard read on the month just ended; drives cyclical and materials H-shares.",
    },
    {
        "indicator": "China Caixin Manufacturing PMI",
        "source_url": "https://www.pmi.spglobal.com/Public/Release/PressReleases",
        "country": "CN",
        "day": 1,
        "window": 2,
        "channel": "china_demand",
        "impact": "medium",
        "note": "Skews to smaller private exporters, so it can diverge from the NBS series.",
    },
    {
        "indicator": "China Trade Balance",
        "source_url": "http://english.customs.gov.cn/statics/report/monthly.html",
        "country": "CN",
        "day": 7,
        "window": 3,
        "channel": "china_demand",
        "impact": "medium",
        "note": "Export momentum feeds shipping, electronics and the industrial complex.",
    },
    {
        "indicator": "China CPI / PPI",
        "source_url": "https://www.stats.gov.cn/english/PressRelease/",
        "country": "CN",
        "day": 9,
        "window": 2,
        "channel": "china_demand",
        "impact": "medium",
        "note": "PPI is the more useful series for H-share margin direction.",
    },
    {
        "indicator": "China Aggregate Financing / New Loans",
        "source_url": "http://www.pbc.gov.cn/en/3688247/index.html",
        "country": "CN",
        "day": 12,
        "window": 4,
        "channel": "china_policy",
        "impact": "high",
        "note": "Credit impulse leads Chinese activity and, with a lag, H-share earnings revisions.",
    },
    {
        "indicator": "China Activity Data (IP / Retail Sales / FAI)",
        "source_url": "https://www.stats.gov.cn/english/PressRelease/",
        "country": "CN",
        "day": 15,
        "window": 2,
        "channel": "china_demand",
        "impact": "high",
        "note": "The broadest monthly read on domestic demand.",
    },
    {
        "indicator": "US CPI",
        "source_url": "https://www.bls.gov/schedule/news_release/cpi.htm",
        "country": "US",
        "day": 12,
        "window": 3,
        "channel": "fed_path",
        "impact": "high",
        "note": "Primary driver of the Fed path and therefore of HK equity duration under the peg.",
    },
    {
        "indicator": "US Non-Farm Payrolls",
        "source_url": "https://www.bls.gov/schedule/news_release/empsit.htm",
        "country": "US",
        "day": 5,
        "window": 3,
        "channel": "fed_path",
        "impact": "high",
        "note": "Released the first Friday; matters for Hong Kong only through the Fed path.",
    },
    {
        "indicator": "Hong Kong CPI",
        "source_url": "https://www.censtatd.gov.hk/en/press_release_list.html",
        "country": "HK",
        "day": 21,
        "window": 3,
        "channel": "hk_liquidity",
        "impact": "low",
        "note": "Local inflation has limited direct equity impact under the peg.",
    },
]


def _clamp_day(year: int, month: int, day: int) -> date:
    """Resolve a nominal day-of-month onto a real calendar date."""
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = (next_month - timedelta(days=1)).day
    return date(year, month, min(day, last_day))


def _first_friday(year: int, month: int) -> date:
    first = date(year, month, 1)
    return first + timedelta(days=(4 - first.weekday()) % 7)


def _scheduled_date(rule: Dict[str, Any], year: int, month: int) -> date:
    if rule["indicator"] == "US Non-Farm Payrolls":
        return _first_friday(year, month)
    return _clamp_day(year, month, int(rule["day"]))


def _month_range(anchor: date, back: int, forward: int) -> List[tuple]:
    """Every (year, month) that could hold a release in the window."""
    months = set()
    for offset in range(-1, 2):
        cursor = anchor + timedelta(days=offset * 31)
        months.add((cursor.year, cursor.month))
    months.add((anchor.year, anchor.month))
    return sorted(months)


def scheduled_events(
    reference: date,
    days_back: int = 4,
    days_forward: int = 5,
) -> List[Dict[str, Any]]:
    """Return releases scheduled inside the window around ``reference``.

    ``days_back`` covers what should already have printed (for the overnight
    review) and ``days_forward`` covers what is coming. The default is 4 calendar
    days so a Monday briefing still carries Friday's releases (the previous
    session's prints were dropped when the window was 1 day).
    """
    start = reference - timedelta(days=days_back)
    end = reference + timedelta(days=days_forward)

    events: List[Dict[str, Any]] = []
    for year, month in _month_range(reference, days_back, days_forward):
        for rule in MONTHLY_RULES:
            scheduled = _scheduled_date(rule, year, month)
            if not (start <= scheduled <= end):
                continue
            window = int(rule.get("window", 0))
            events.append(
                {
                    "indicator": rule["indicator"],
                    "country": rule["country"],
                    # The publisher's release page: a dated event has to be
                    # traceable to whoever actually publishes it.
                    "source_url": rule.get("source_url", ""),
                    "date": scheduled.isoformat(),
                    "channel": rule["channel"],
                    "channel_note": CHANNELS[rule["channel"]],
                    "impact": rule["impact"],
                    "note": rule["note"],
                    "timing_confidence": "exact" if window == 0 else f"+/-{window}d",
                    "status": "released" if scheduled < reference else "upcoming",
                    "basis": "rule_based_schedule",
                }
            )

    events.sort(key=lambda item: (item["date"], {"high": 0, "medium": 1, "low": 2}[item["impact"]]))
    return events


def summarize_channels(events: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for item in events:
        counts[item["channel"]] = counts.get(item["channel"], 0) + 1
    return counts
