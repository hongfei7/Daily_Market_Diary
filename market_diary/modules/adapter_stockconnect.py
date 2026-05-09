"""HKEX Stock Connect daily statistics adapter."""

from __future__ import annotations

import json
import os
import re
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


HKEX_STOCK_CONNECT_TEMPLATE = "https://www.hkex.com.hk/eng/csm/DailyStat/data_tab_daily_{yyyymmdd}e.js"
HKEX_STOCK_CONNECT_SOURCE = "HKEX Stock Connect Historical Daily"
USER_AGENT = "Daily-Market-Diary/3.1"
REQUEST_TIMEOUT = float(os.environ.get("DMD_PUBLIC_REQUEST_TIMEOUT_SECONDS", "12"))
REQUEST_RETRY_TOTAL = int(os.environ.get("DMD_PUBLIC_RETRY_TOTAL", "1"))
LOOKBACK_DAYS = 7


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=REQUEST_RETRY_TOTAL,
        connect=REQUEST_RETRY_TOTAL,
        read=REQUEST_RETRY_TOTAL,
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


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace(",", "").strip()
    if not cleaned or cleaned in {"-", "N/A"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _turnover_to_millions(value: Any) -> Optional[float]:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return parsed / 1_000_000.0


def _normalize_code(code: Any, suffix: str = "") -> str:
    digits = re.sub(r"\D", "", str(code or ""))
    if not digits:
        return ""
    if suffix == ".HK":
        return digits.zfill(5) + suffix
    return digits + suffix


def _extract_payload(text: str) -> List[Dict[str, Any]]:
    if "=" not in text:
        raise ValueError("HKEX Stock Connect response did not contain tabData assignment.")
    raw = text.split("=", 1)[1].strip().rstrip(";").strip()
    payload = json.loads(raw)
    if not isinstance(payload, list):
        raise ValueError("HKEX Stock Connect payload was not a list.")
    return payload


def _metric_rows(tab: Dict[str, Any]) -> Dict[str, Any]:
    content = (tab.get("content") or [])
    if not content:
        return {}
    table = ((content[0] or {}).get("table") or {})
    schema = ((table.get("schema") or [[]])[0] or [])
    rows = table.get("tr") or []
    values = []
    for row in rows:
        cell = (((row.get("td") or [[]])[0]) or [""])[0]
        values.append(cell)
    return {str(key): values[idx] if idx < len(values) else "" for idx, key in enumerate(schema)}


def _top_ten_rows(tab: Dict[str, Any], market: str) -> List[Dict[str, Any]]:
    content = (tab.get("content") or [])
    if len(content) < 2:
        return []
    table = ((content[1] or {}).get("table") or {})
    schema = ((table.get("schema") or [[]])[0] or [])
    output: List[Dict[str, Any]] = []
    is_southbound = "Southbound" in market
    suffix = ".HK" if is_southbound else ""
    for row in table.get("tr") or []:
        cells = ((row.get("td") or [[]])[0]) or []
        values = {str(key): cells[idx] if idx < len(cells) else "" for idx, key in enumerate(schema)}
        buy = _turnover_to_millions(values.get("Buy Turnover"))
        sell = _turnover_to_millions(values.get("Sell Turnover"))
        total = _turnover_to_millions(values.get("Total Turnover"))
        net = (buy - sell) if buy is not None and sell is not None else None
        output.append(
            {
                "rank": int(_to_float(values.get("Rank")) or 0),
                "code": str(values.get("Stock Code", "")).strip(),
                "ticker": _normalize_code(values.get("Stock Code"), suffix=suffix),
                "name": str(values.get("Stock Name", "")).strip(),
                "buy_turnover": buy,
                "sell_turnover": sell,
                "net_buy": net,
                "total_turnover": total,
                "market": market,
            }
        )
    return output


def _parse_tab(tab: Dict[str, Any]) -> Dict[str, Any]:
    market = str(tab.get("market", ""))
    metrics = _metric_rows(tab)
    total_turnover = _to_float(metrics.get("Total Turnover"))
    buy_turnover = _to_float(metrics.get("Buy Turnover"))
    sell_turnover = _to_float(metrics.get("Sell Turnover"))
    net_buy = (buy_turnover - sell_turnover) if buy_turnover is not None and sell_turnover is not None else None
    return {
        "market": market,
        "date": tab.get("date", ""),
        "trading_day": bool(tab.get("tradingDay")),
        "total_turnover": total_turnover,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "net_buy": net_buy,
        "trade_count": _to_float(metrics.get("Total Trade Count") or metrics.get("No. of Buy + Sell Trades")),
        "buy_trade_count": _to_float(metrics.get("Buy Trade Count") or metrics.get("No. of Buy Trades")),
        "sell_trade_count": _to_float(metrics.get("Sell Trade Count") or metrics.get("No. of Sell Trades")),
        "daily_quota_balance": _to_float(metrics.get("DQB") or metrics.get("Daily Quota Balance")),
        "etf_turnover": _to_float(metrics.get("ETF Turnover")),
        "top_active": _top_ten_rows(tab, market),
    }


def _sum_optional(left: Optional[float], right: Optional[float]) -> Optional[float]:
    values = [value for value in (left, right) if value is not None]
    return sum(values) if values else None


def _merge_active_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        key = row.get("ticker") or f"{row.get('market', '')}:{row.get('code', '')}"
        if key not in merged:
            merged[key] = {
                **row,
                "markets": [row.get("market", "")] if row.get("market") else [],
            }
            continue

        current = merged[key]
        current["buy_turnover"] = _sum_optional(current.get("buy_turnover"), row.get("buy_turnover"))
        current["sell_turnover"] = _sum_optional(current.get("sell_turnover"), row.get("sell_turnover"))
        current["net_buy"] = _sum_optional(current.get("net_buy"), row.get("net_buy"))
        current["total_turnover"] = _sum_optional(current.get("total_turnover"), row.get("total_turnover"))
        if row.get("market") and row.get("market") not in current["markets"]:
            current["markets"].append(row.get("market"))
        try:
            current["rank"] = min(int(current.get("rank") or 999), int(row.get("rank") or 999))
        except (TypeError, ValueError):
            pass

    output = []
    for row in merged.values():
        markets = row.pop("markets", [])
        if markets:
            row["market"] = " / ".join(markets)
        output.append(row)
    return output


def _aggregate_markets(markets: List[Dict[str, Any]], direction: str) -> Dict[str, Any]:
    rows = [item for item in markets if direction in item.get("market", "")]
    total_turnover = sum(item.get("total_turnover") or 0 for item in rows) or None
    buy_turnover_values = [item.get("buy_turnover") for item in rows if item.get("buy_turnover") is not None]
    sell_turnover_values = [item.get("sell_turnover") for item in rows if item.get("sell_turnover") is not None]
    buy_turnover = sum(buy_turnover_values) if buy_turnover_values else None
    sell_turnover = sum(sell_turnover_values) if sell_turnover_values else None
    net_buy = (buy_turnover - sell_turnover) if buy_turnover is not None and sell_turnover is not None else None

    active: List[Dict[str, Any]] = []
    for item in rows:
        active.extend(item.get("top_active", []) or [])
    active = _merge_active_rows(active)
    active.sort(key=lambda row: abs(row.get("net_buy") or row.get("total_turnover") or 0), reverse=True)

    return {
        "direction": direction,
        "markets": rows,
        "total_turnover": total_turnover,
        "buy_turnover": buy_turnover,
        "sell_turnover": sell_turnover,
        "net_buy": net_buy,
        "top_active": active[:10],
        "net_buy_available": net_buy is not None,
    }


def _fetch_for_day(day: date, session: Optional[requests.Session] = None) -> Optional[Dict[str, Any]]:
    url = HKEX_STOCK_CONNECT_TEMPLATE.format(yyyymmdd=day.strftime("%Y%m%d"))
    http = session or _session()
    response = http.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    payload = _extract_payload(response.text)
    markets = [_parse_tab(tab) for tab in payload]
    if not markets:
        return None
    effective_date = markets[0].get("date") or day.isoformat()
    return {
        "effective_date": effective_date,
        "source_url": url,
        "markets": markets,
        "southbound": _aggregate_markets(markets, "Southbound"),
        "northbound": _aggregate_markets(markets, "Northbound"),
    }


def fetch_stock_connect_data(report_date: str, lookback_days: int = LOOKBACK_DAYS) -> Dict[str, Any]:
    target = _parse_date(report_date)
    last_error = ""
    session = _session()
    for offset in range(max(lookback_days, 0) + 1):
        day = target - timedelta(days=offset)
        try:
            snapshot = _fetch_for_day(day, session=session)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
        if not snapshot:
            continue

        return {
            "status": "ok",
            "data": {
                "southbound": snapshot["southbound"],
                "northbound": snapshot["northbound"],
                "markets": snapshot["markets"],
            },
            "meta": {
                "report_date": report_date,
                "effective_date": snapshot["effective_date"],
                "source": HKEX_STOCK_CONNECT_SOURCE,
                "source_url": snapshot["source_url"],
                "lookback_days": offset,
                "notes": (
                    "HKEX public daily files provide turnover, trade count, ETF turnover, "
                    "and Southbound buy/sell turnover for top active names. Northbound net-buy "
                    "is unavailable after the current disclosure change and is kept explicit."
                ),
            },
        }

    return {
        "status": "error",
        "data": {
            "southbound": {"top_active": [], "net_buy_available": False},
            "northbound": {"top_active": [], "net_buy_available": False},
            "markets": [],
        },
        "meta": {
            "report_date": report_date,
            "source": HKEX_STOCK_CONNECT_SOURCE,
            "source_url": "",
            "error": last_error or "No parseable HKEX Stock Connect daily file was found.",
        },
    }
