"""A/H premium adapter using public market quotes."""

from __future__ import annotations

import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf


AH_PREMIUM_SOURCE = "Public Yahoo Finance quotes - calculated A/H premium"
LOOKBACK_DAYS = 14
MAX_WORKERS = 6
MIN_REASONABLE_PREMIUM_PCT = -80.0
MAX_REASONABLE_PREMIUM_PCT = 250.0

AH_UNIVERSE = [
    {"name": "Ping An", "a": "601318.SS", "h": "2318.HK"},
    {"name": "ICBC", "a": "601398.SS", "h": "1398.HK"},
    {"name": "China Construction Bank", "a": "601939.SS", "h": "0939.HK"},
    {"name": "China Merchants Bank", "a": "600036.SS", "h": "3968.HK"},
    {"name": "Bank of China", "a": "601988.SS", "h": "3988.HK"},
    {"name": "Agricultural Bank of China", "a": "601288.SS", "h": "1288.HK"},
    {"name": "China Life", "a": "601628.SS", "h": "2628.HK"},
    {"name": "PetroChina", "a": "601857.SS", "h": "0857.HK"},
    {"name": "Sinopec", "a": "600028.SS", "h": "0386.HK"},
    {"name": "China Shenhua", "a": "601088.SS", "h": "1088.HK"},
    {"name": "Zijin Mining", "a": "601899.SS", "h": "2899.HK"},
    {"name": "CRRC", "a": "601766.SS", "h": "1766.HK"},
    {"name": "Chalco", "a": "601600.SS", "h": "2600.HK"},
    {"name": "China Railway", "a": "601390.SS", "h": "0390.HK"},
    {"name": "China Railway Construction", "a": "601186.SS", "h": "1186.HK"},
    {"name": "China Communications Construction", "a": "601800.SS", "h": "1800.HK"},
    {"name": "CITIC Securities", "a": "600030.SS", "h": "6030.HK"},
    {"name": "China Pacific Insurance", "a": "601601.SS", "h": "2601.HK"},
    {"name": "New China Life", "a": "601336.SS", "h": "1336.HK"},
    {"name": "Everbright Bank", "a": "601818.SS", "h": "6818.HK"},
]


def _parse_date(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def _download_history(symbol: str, target: datetime):
    start = target - timedelta(days=LOOKBACK_DAYS)
    end = target + timedelta(days=1)
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return yf.download(symbol, start=start, end=end, interval="1d", progress=False, auto_adjust=False, threads=False)


def _last_close(symbol: str, target: datetime) -> Optional[Dict[str, Any]]:
    try:
        hist = _download_history(symbol, target)
    except Exception:
        return None
    if hist is None or hist.empty or "Close" not in hist:
        return None
    close = hist["Close"].dropna()
    if close.empty:
        return None
    last = close.iloc[-1]
    if hasattr(last, "iloc"):
        last = last.iloc[0]
    date_value = close.index[-1]
    if hasattr(date_value, "date"):
        as_of = date_value.date().isoformat()
    else:
        as_of = str(date_value)[:10]
    try:
        return {"price": float(last), "as_of": as_of}
    except (TypeError, ValueError):
        return None


def _fx_cny_hkd(target: datetime) -> Optional[Dict[str, Any]]:
    direct = _last_close("CNYHKD=X", target)
    if direct and direct.get("price"):
        return {"value": direct["price"], "as_of": direct["as_of"], "basis": "CNYHKD=X"}

    usd_hkd = _last_close("HKD=X", target)
    usd_cnh = _last_close("CNH=F", target)
    if usd_hkd and usd_cnh and usd_cnh.get("price"):
        return {
            "value": float(usd_hkd["price"]) / float(usd_cnh["price"]),
            "as_of": min(str(usd_hkd["as_of"]), str(usd_cnh["as_of"])),
            "basis": "USD/HKD divided by USD/CNH",
        }
    return None


def _calculate_premium(a_price_cny: float, h_price_hkd: float, cny_hkd: float) -> Optional[float]:
    if h_price_hkd <= 0 or cny_hkd <= 0:
        return None
    a_hkd = float(a_price_cny) * float(cny_hkd)
    return ((a_hkd / float(h_price_hkd)) - 1.0) * 100.0


def _pair_row(item: Dict[str, str], target: datetime, fx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    a_quote = _last_close(item["a"], target)
    h_quote = _last_close(item["h"], target)
    if not a_quote or not h_quote or not h_quote.get("price"):
        return None

    a_price = float(a_quote["price"])
    h_price = float(h_quote["price"])
    premium = _calculate_premium(a_price, h_price, float(fx["value"]))
    if premium is None:
        return None
    if premium < MIN_REASONABLE_PREMIUM_PCT or premium > MAX_REASONABLE_PREMIUM_PCT:
        return None

    return {
        "name": item["name"],
        "a_ticker": item["a"],
        "h_ticker": item["h"],
        "a_price_cny": round(a_price, 3),
        "h_price_hkd": round(h_price, 3),
        "a_price_hkd_equiv": round(a_price * float(fx["value"]), 3),
        "premium_pct": round(premium, 2),
        "as_of": min(str(a_quote["as_of"]), str(h_quote["as_of"]), str(fx["as_of"])),
    }


def fetch_ah_premium_data(report_date: str, universe: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
    target = _parse_date(report_date)
    fx = _fx_cny_hkd(target)
    if not fx:
        return {
            "status": "error",
            "data": {"rows": [], "top_premium": [], "lowest_premium": [], "average_premium": None},
            "meta": {
                "report_date": report_date,
                "source": AH_PREMIUM_SOURCE,
                "error": "CNY/HKD conversion could not be derived from public quotes.",
            },
        }

    rows: List[Dict[str, Any]] = []
    selected_universe = universe or AH_UNIVERSE
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, max(len(selected_universe), 1))) as executor:
        futures = [executor.submit(_pair_row, item, target, fx) for item in selected_universe]
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception:
                row = None
            if row:
                rows.append(row)

    rows.sort(key=lambda row: row.get("premium_pct", 0), reverse=True)
    average = sum(row["premium_pct"] for row in rows) / len(rows) if rows else None
    status = "ok" if rows else "error"
    return {
        "status": status,
        "data": {
            "rows": rows,
            "top_premium": rows[:10],
            "lowest_premium": sorted(rows, key=lambda row: row.get("premium_pct", 0))[:10],
            "average_premium": round(average, 2) if average is not None else None,
        },
        "meta": {
            "report_date": report_date,
            "source": AH_PREMIUM_SOURCE,
            "effective_date": rows[0]["as_of"] if rows else "",
            "fx_cny_hkd": round(float(fx["value"]), 6),
            "fx_basis": fx["basis"],
            "coverage": len(rows),
            "universe": len(selected_universe),
            "premium_filter": f"{MIN_REASONABLE_PREMIUM_PCT:.0f}% to {MAX_REASONABLE_PREMIUM_PCT:.0f}%",
        },
    }
