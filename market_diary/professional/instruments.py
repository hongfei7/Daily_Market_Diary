"""Canonical instrument identity, unit, and display helpers.

The market adapters keep their legacy dictionary keys for compatibility, but
all reader-facing layers use this registry so that an ETF cannot silently be
presented as its underlying index and rate changes cannot be presented as
equity-style percentage returns.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Mapping


# A quote is decision-usable only if it is at most this many trading days old.
# Anything older must not silently drive a style call, a regime label, or a risk
# score: comparing a fresh leg against a stale one manufactures a spread that
# never happened on any single date.
MAX_FRESH_TRADING_DAYS = 1


INSTRUMENTS: Dict[tuple[str, str], Dict[str, str]] = {
    ("Equities", "S&P 500"): {
        "instrument_id": "index.us.spx",
        "display_name": "S&P 500",
        "security_type": "index",
        "price_unit": "index_points",
        "change_unit": "pct",
    },
    ("Equities", "Nasdaq 100"): {
        "instrument_id": "index.us.ndx",
        "display_name": "Nasdaq 100",
        "security_type": "index",
        "price_unit": "index_points",
        "change_unit": "pct",
    },
    ("Equities", "Hang Seng Index"): {
        "instrument_id": "index.hk.hsi",
        "display_name": "Hang Seng Index",
        "security_type": "index",
        "price_unit": "index_points",
        "change_unit": "pct",
    },
    ("Equities", "Hang Seng China Enterprises"): {
        "instrument_id": "index.hk.hscei",
        "display_name": "Hang Seng China Enterprises Index",
        "security_type": "index",
        "price_unit": "index_points",
        "change_unit": "pct",
    },
    ("Equities", "Hang Seng TECH ETF"): {
        "instrument_id": "etf.hk.3033",
        "display_name": "Hang Seng TECH ETF (3033.HK)",
        "security_type": "etf",
        "price_unit": "HKD_per_share",
        "change_unit": "pct",
    },
    ("Rates", "13W T-Bill"): {
        "instrument_id": "yield.us.13w",
        "display_name": "US 13W T-Bill Yield",
        "security_type": "yield",
        "price_unit": "yield_pct",
        "change_unit": "bp",
    },
    ("Rates", "5Y Treasury"): {
        "instrument_id": "yield.us.5y",
        "display_name": "US 5Y Treasury Yield",
        "security_type": "yield",
        "price_unit": "yield_pct",
        "change_unit": "bp",
    },
    ("Rates", "10Y Treasury"): {
        "instrument_id": "yield.us.10y",
        "display_name": "US 10Y Treasury Yield",
        "security_type": "yield",
        "price_unit": "yield_pct",
        "change_unit": "bp",
    },
    ("Rates", "30Y Treasury"): {
        "instrument_id": "yield.us.30y",
        "display_name": "US 30Y Treasury Yield",
        "security_type": "yield",
        "price_unit": "yield_pct",
        "change_unit": "bp",
    },
}


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value or "").replace(",", "").replace("%", "").strip()
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def instrument_spec(category: str, name: str, source_symbol: str = "") -> Dict[str, str]:
    configured = INSTRUMENTS.get((category, name), {})
    return {
        "instrument_id": configured.get("instrument_id", f"{category.lower()}.{name.lower().replace(' ', '_')}"),
        "display_name": configured.get("display_name", name),
        "security_type": configured.get("security_type", category.lower().rstrip("s")),
        "price_unit": configured.get("price_unit", "quoted_price"),
        "change_unit": configured.get("change_unit", "pct"),
        "source_symbol": source_symbol,
    }


def trading_freshness_days(category: str, as_of: Any, target_date: str) -> int | None:
    """Return missed trading weekdays, not raw calendar days.

    Crypto is treated as continuously traded. Other public market proxies use
    weekday-aware freshness; exchange-holiday precision remains a documented
    follow-up for the source-specific calendar layer.
    """
    try:
        start = datetime.strptime(str(as_of or "")[:10], "%Y-%m-%d").date()
        target = datetime.strptime(str(target_date or "")[:10], "%Y-%m-%d").date()
    except ValueError:
        return None
    if target <= start:
        return 0
    if category == "Crypto":
        return (target - start).days
    missed = 0
    cursor = start + timedelta(days=1)
    while cursor <= target:
        if cursor.weekday() < 5:
            missed += 1
        cursor += timedelta(days=1)
    return missed


def annotate_summary_item(
    category: str,
    name: str,
    source_symbol: str,
    item: Mapping[str, Any],
    target_date: str,
) -> Dict[str, Any]:
    """Attach identity, units, display change, and trading-day freshness."""
    result = dict(item)
    spec = instrument_spec(category, name, source_symbol)
    raw_change = _number(result.get("Change"))
    pct_change = _number(result.get("Pct Change"))
    if spec["change_unit"] == "bp":
        change_value = raw_change * 100.0 if raw_change is not None else None
        change_display = f"{change_value:+.1f}bp" if change_value is not None else "N/A"
    else:
        change_value = pct_change
        change_display = f"{change_value:+.2f}%" if change_value is not None else "N/A"

    freshness = trading_freshness_days(category, result.get("As Of"), target_date)
    quality = str(result.get("Quality", "fresh") or "fresh")
    if freshness is not None and freshness > MAX_FRESH_TRADING_DAYS:
        quality = "stale"

    result.update(
        {
            "Instrument ID": spec["instrument_id"],
            "Display Name": spec["display_name"],
            "Security Type": spec["security_type"],
            "Price Unit": spec["price_unit"],
            "Change Unit": spec["change_unit"],
            "Change Value": round(change_value, 4) if change_value is not None else None,
            "Change Display": change_display,
            "Source Symbol": source_symbol,
            "Trading Freshness Days": freshness,
            "Quality": quality,
        }
    )
    return result


def summary_change(item: Mapping[str, Any]) -> tuple[float | None, str]:
    value = _number(item.get("Change Value"))
    unit = str(item.get("Change Unit", "pct") or "pct")
    if value is not None:
        return value, unit
    return _number(item.get("Pct Change")), "pct"


def format_summary_change(item: Mapping[str, Any]) -> str:
    display = str(item.get("Change Display", "") or "").strip()
    if display:
        return display
    value, unit = summary_change(item)
    if value is None:
        return "N/A"
    return f"{value:+.1f}bp" if unit == "bp" else f"{value:+.2f}%"
