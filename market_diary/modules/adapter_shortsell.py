"""HKEX stock-level short-selling adapter.

The daily quotation page is an official HKEX public source and contains both
stock-level short-selling turnover and matched stock turnover.  This adapter
keeps the output deterministic and structured so the LLM can only interpret
the already-normalized facts.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Set

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HKEX_DAY_QUOTATION_TEMPLATE = "https://www.hkex.com.hk/eng/stat/smstat/dayquot/d{yymmdd}e.htm"
HKEX_SHORT_SELL_SOURCE = "HKEX Daily Quotations - Short Selling Turnover"
USER_AGENT = "Daily-Market-Diary/3.1"
REQUEST_TIMEOUT = 35
LOOKBACK_DAYS = 7

SHORT_SECTION_START = '<a name = "short_selling">SHORT SELLING TURNOVER - DAILY REPORT</a>'
SHORT_SECTION_END = '<a name = "adj_short">'
SHORT_ROW_RE = re.compile(
    r"^\s*(?P<code>\d{1,5})\s+"
    r"(?P<name>.+?)\s+"
    r"(?P<short_shares>[0-9][0-9,]*)\s+"
    r"(?P<short_turnover>[0-9][0-9,]*)\s+"
    r"(?P<total_shares>[0-9][0-9,]*)\s+"
    r"(?P<total_turnover>[0-9][0-9,]*)\s*$"
)
HKEX_DATE_RE = re.compile(r"DATE:\s*([0-9]{1,2}\s+[A-Z]{3}\s+[0-9]{4})", re.IGNORECASE)
MARKET_RATIO_RE = re.compile(
    r"Short Selling of all Designated Securities as % total turnover\s*:\s*([0-9.]+)%",
    re.IGNORECASE,
)
MARKET_SHORT_VALUE_RE = re.compile(
    r"\(C\) Short Selling of all Designated Securities.*?"
    r"Short Selling Turnover Total Value \(\$\)\s*:\s*HKD\s*([0-9,]+)",
    re.IGNORECASE | re.DOTALL,
)
MARKET_TURNOVER_RE = re.compile(r"Total market turnover\s*:\s*HKD\s*([0-9,]+)", re.IGNORECASE)


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=4,
        connect=4,
        read=4,
        backoff_factor=1.0,
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _day_quotation_url(day: date) -> str:
    return HKEX_DAY_QUOTATION_TEMPLATE.format(yymmdd=day.strftime("%y%m%d"))


def _to_float(value: str) -> float:
    return float((value or "0").replace(",", "").strip() or 0)


def _normalize_code(code: Any) -> str:
    digits = re.sub(r"\D", "", str(code or ""))
    return digits.zfill(5) if digits else ""


def _watchlist_codes(watchlists: Optional[Dict[str, List[Dict[str, Any]]]]) -> Set[str]:
    codes: Set[str] = set()
    for items in (watchlists or {}).values():
        for item in items or []:
            code = _normalize_code(str(item.get("ticker", "")).split(".")[0])
            if code:
                codes.add(code)
    return codes


def _extract_section(raw_html: str) -> str:
    start = raw_html.find(SHORT_SECTION_START)
    if start < 0:
        return ""
    end = raw_html.find(SHORT_SECTION_END, start)
    if end < 0:
        end = len(raw_html)
    return html.unescape(raw_html[start:end])


def _parse_effective_date(raw_html: str, fallback: date) -> date:
    match = HKEX_DATE_RE.search(raw_html or "")
    if not match:
        return fallback
    try:
        return datetime.strptime(match.group(1).upper(), "%d %b %Y").date()
    except ValueError:
        return fallback


def _parse_stock_rows(section: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in (section or "").splitlines():
        match = SHORT_ROW_RE.match(line)
        if not match:
            continue

        short_turnover = _to_float(match.group("short_turnover"))
        total_turnover = _to_float(match.group("total_turnover"))
        ratio = (short_turnover / total_turnover * 100.0) if total_turnover > 0 else None
        code = _normalize_code(match.group("code"))
        rows.append(
            {
                "code": code,
                "ticker": f"{code}.HK" if code else "",
                "name": " ".join(match.group("name").split()),
                "short_shares": int(_to_float(match.group("short_shares"))),
                "short_turnover_hkd": short_turnover,
                "total_shares": int(_to_float(match.group("total_shares"))),
                "total_turnover_hkd": total_turnover,
                "short_ratio_pct": round(ratio, 2) if ratio is not None else None,
            }
        )
    return rows


def _market_summary(section: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    ratio_match = MARKET_RATIO_RE.search(section or "")
    value_match = MARKET_SHORT_VALUE_RE.search(section or "")
    turnover_match = MARKET_TURNOVER_RE.search(section or "")

    short_value = _to_float(value_match.group(1)) if value_match else sum(item.get("short_turnover_hkd", 0) for item in rows)
    total_turnover = _to_float(turnover_match.group(1)) if turnover_match else sum(item.get("total_turnover_hkd", 0) for item in rows)
    ratio = float(ratio_match.group(1)) if ratio_match else (short_value / total_turnover * 100.0 if total_turnover > 0 else None)

    return {
        "short_turnover_hkd": short_value,
        "total_turnover_hkd": total_turnover,
        "short_ratio_pct": round(ratio, 2) if ratio is not None else None,
    }


def _rank_rows(rows: Iterable[Dict[str, Any]], key: str, limit: int) -> List[Dict[str, Any]]:
    return sorted(
        (item for item in rows if item.get(key) is not None),
        key=lambda item: float(item.get(key, 0)),
        reverse=True,
    )[:limit]


def _fetch_for_day(day: date) -> Optional[Dict[str, Any]]:
    url = _day_quotation_url(day)
    response = _session().get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    raw_html = response.text
    section = _extract_section(raw_html)
    if not section:
        return None

    rows = _parse_stock_rows(section)
    if not rows:
        return None

    effective_date = _parse_effective_date(raw_html, day)
    return {
        "url": url,
        "effective_date": effective_date.isoformat(),
        "rows": rows,
        "market": _market_summary(section, rows),
    }


def fetch_short_sell_data(
    report_date: str,
    watchlists: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    lookback_days: int = LOOKBACK_DAYS,
    top_n: int = 10,
) -> Dict[str, Any]:
    """Fetch official HKEX stock-level short-selling data.

    The adapter walks back across a short date window because HKEX does not
    publish files for weekends and holidays.  The first parseable quotation
    page at or before the report date is treated as the effective observation.
    """

    target = _parse_date(report_date)
    last_error = ""
    for offset in range(max(lookback_days, 0) + 1):
        day = target - timedelta(days=offset)
        try:
            snapshot = _fetch_for_day(day)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue

        if not snapshot:
            continue

        rows = snapshot["rows"]
        watch_codes = _watchlist_codes(watchlists)
        watchlist_hits = [item for item in rows if item.get("code") in watch_codes]
        top_ratio = _rank_rows((item for item in rows if (item.get("total_turnover_hkd") or 0) >= 5_000_000), "short_ratio_pct", top_n)
        top_value = _rank_rows(rows, "short_turnover_hkd", top_n)

        return {
            "status": "ok",
            "data": {
                "market": snapshot["market"],
                "top_short_ratio": top_ratio,
                "top_short_value": top_value,
                "watchlist_hits": watchlist_hits[:top_n],
            },
            "meta": {
                "report_date": report_date,
                "effective_date": snapshot["effective_date"],
                "source": HKEX_SHORT_SELL_SOURCE,
                "source_url": snapshot["url"],
                "row_count": len(rows),
                "lookback_days": offset,
            },
        }

    return {
        "status": "error",
        "data": {
            "market": {},
            "top_short_ratio": [],
            "top_short_value": [],
            "watchlist_hits": [],
        },
        "meta": {
            "report_date": report_date,
            "source": HKEX_SHORT_SELL_SOURCE,
            "source_url": "",
            "error": last_error or "No parseable HKEX short-selling page was found.",
        },
    }
