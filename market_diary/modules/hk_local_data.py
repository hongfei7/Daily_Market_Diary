"""Hong Kong local-market public data adapters."""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from statistics import mean
from typing import Any, Dict, List, Optional

import requests

from market_diary.modules.local_metrics import (
    append_error_record,
    build_metric,
    format_hkd_billions,
    format_percent,
    format_ratio,
    format_rmb_billions,
    parse_target_date,
    summarize_error_records,
    unavailable_metric,
)


HKMA_LIQUIDITY_URL = (
    "https://api.hkma.gov.hk/public/market-data-and-statistics/"
    "daily-monetary-statistics/daily-figures-interbank-liquidity"
)
HKEX_DAY_QUOTATION_TEMPLATE = "https://www.hkex.com.hk/eng/stat/smstat/dayquot/d{yymmdd}e.htm"
HKEX_SHORT_SELL_URL = "https://www.hkex.com.hk/eng/stat/smstat/ssturnover/ncms/MSHTMAIN.HTM"
HKEX_SOURCE = "HKEX Daily Quotations"
HKEX_SHORT_SELL_SOURCE = "HKEX Short Selling Turnover Report"
HKMA_SOURCE = "HKMA Daily Figures - Interbank Liquidity"
USER_AGENT = "Daily-Market-Diary/3.0"
REQUEST_TIMEOUT = float(os.environ.get("DMD_PUBLIC_REQUEST_TIMEOUT_SECONDS", "12"))
TURNOVER_LOOKBACK_DAYS = 45
TURNOVER_AVERAGE_WINDOW = 20
TURNOVER_MAX_WORKERS = 6

TURNOVER_PATTERN = re.compile(r"Today's Turnover:\s*\(HK\$\):\s*([0-9,]+)", re.IGNORECASE | re.DOTALL)
HKEX_DATE_PATTERN = re.compile(r"DATE:\s*([0-9]{1,2}\s+[A-Z]{3}\s+[0-9]{4})", re.IGNORECASE)
SHORT_DATE_PATTERN = re.compile(r"TRADING DATE\s*:\s*([0-9]{1,2}\s+[A-Z]{3}\s+[0-9]{4})", re.IGNORECASE)
SHORT_TOTAL_PATTERN = re.compile(
    r"Short Selling Turnover Total Value.*?HKD\s*([0-9,]+)",
    re.IGNORECASE | re.DOTALL,
)


def _headers() -> Dict[str, str]:
    return {"User-Agent": USER_AGENT}


def _status_from_metrics(metrics: Dict[str, Dict[str, Any]]) -> str:
    statuses = [str(item.get("status", "")) for item in metrics.values() if isinstance(item, dict)]
    if not statuses:
        return "error"
    if any(status.startswith("live") or status.startswith("stale") for status in statuses):
        if all(status == "unavailable" for status in statuses):
            return "error"
        return "ok"
    return "partial"


def _stream_buffer(url: str, max_bytes: int = 24_000) -> str:
    buffer = ""
    with requests.get(url, headers=_headers(), timeout=REQUEST_TIMEOUT, stream=True) as response:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=4096, decode_unicode=True):
            if not chunk:
                continue
            buffer += chunk
            if len(buffer) >= max_bytes:
                break
    return buffer


def _day_quotation_url(day: date) -> str:
    return HKEX_DAY_QUOTATION_TEMPLATE.format(yymmdd=day.strftime("%y%m%d"))


def _fetch_turnover_for_day(day: date, errors: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    url = _day_quotation_url(day)
    try:
        buffer = _stream_buffer(url)
    except requests.RequestException as exc:
        append_error_record(errors, source=HKEX_SOURCE, message=str(exc), error_type=type(exc).__name__, context=day.isoformat())
        return None
    except Exception as exc:
        append_error_record(errors, source=HKEX_SOURCE, message=str(exc), error_type=type(exc).__name__, context=day.isoformat())
        return None

    turnover_match = TURNOVER_PATTERN.search(buffer)
    if not turnover_match:
        return None

    try:
        turnover_hkd = float(turnover_match.group(1).replace(",", ""))
    except ValueError:
        return None

    quoted_date = day
    date_match = HKEX_DATE_PATTERN.search(buffer)
    if date_match:
        try:
            quoted_date = datetime.strptime(date_match.group(1).upper(), "%d %b %Y").date()
        except ValueError:
            quoted_date = day

    return {
        "date": quoted_date,
        "turnover_hkd": turnover_hkd,
        "source": HKEX_SOURCE,
        "source_url": url,
    }


def _collect_turnover_history(target: date, errors: Optional[List[Dict[str, str]]] = None) -> List[Dict[str, Any]]:
    candidates = [target - timedelta(days=offset) for offset in range(TURNOVER_LOOKBACK_DAYS + 1)]
    snapshots: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=TURNOVER_MAX_WORKERS) as executor:
        futures = {executor.submit(_fetch_turnover_for_day, day, errors): day for day in candidates}
        for future in as_completed(futures):
            try:
                item = future.result()
            except Exception as exc:
                append_error_record(errors, source=HKEX_SOURCE, message=str(exc), error_type=type(exc).__name__, context=futures[future].isoformat())
                item = None
            if item and item.get("date") and item["date"] <= target:
                snapshots.append(item)
    snapshots.sort(key=lambda item: item["date"], reverse=True)
    return snapshots


def _fetch_hkma_record(target: date, errors: Optional[List[Dict[str, str]]] = None) -> Optional[Dict[str, Any]]:
    try:
        response = requests.get(HKMA_LIQUIDITY_URL, headers=_headers(), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        append_error_record(errors, source=HKMA_SOURCE, message=str(exc), error_type=type(exc).__name__, context=target.isoformat())
        return None
    except ValueError as exc:
        append_error_record(errors, source=HKMA_SOURCE, message=str(exc), error_type=type(exc).__name__, context=f"{target.isoformat()} invalid-json")
        return None
    except Exception as exc:
        append_error_record(errors, source=HKMA_SOURCE, message=str(exc), error_type=type(exc).__name__, context=target.isoformat())
        return None

    records = ((payload.get("result", {}) or {}).get("records", []) or [])
    # Pick the latest record at or before the target instead of the first one:
    # the HKMA API does not document a guaranteed ordering, and returning the
    # first match could publish months-stale HIBOR / Aggregate Balance as
    # "live_local".
    best: Optional[Dict[str, Any]] = None
    best_date: Optional[date] = None
    for record in records:
        try:
            end_of_date = parse_target_date(str(record.get("end_of_date", "")))
        except ValueError:
            continue
        if end_of_date <= target and (best_date is None or end_of_date > best_date):
            best = record
            best_date = end_of_date
    return best


def _fetch_short_sell_snapshot(
    target: date,
    turnover_map: Dict[date, float],
    errors: Optional[List[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    try:
        response = requests.get(HKEX_SHORT_SELL_URL, headers=_headers(), timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        html = response.text
    except requests.RequestException as exc:
        append_error_record(errors, source=HKEX_SHORT_SELL_SOURCE, message=str(exc), error_type=type(exc).__name__, context=target.isoformat())
        return unavailable_metric(
            target.isoformat(),
            HKEX_SHORT_SELL_SOURCE,
            "Short-selling report could not be retrieved from HKEX.",
        )
    except Exception as exc:
        append_error_record(errors, source=HKEX_SHORT_SELL_SOURCE, message=str(exc), error_type=type(exc).__name__, context=target.isoformat())
        return unavailable_metric(
            target.isoformat(),
            HKEX_SHORT_SELL_SOURCE,
            "Short-selling report could not be retrieved from HKEX.",
        )

    date_match = SHORT_DATE_PATTERN.search(html)
    total_match = SHORT_TOTAL_PATTERN.search(html)
    is_day_close = "Up To Day Close Today" in html

    if not date_match or not total_match:
        return unavailable_metric(
            target.isoformat(),
            HKEX_SHORT_SELL_SOURCE,
            "HKEX short-selling report did not expose a parseable total.",
        )

    try:
        record_date = datetime.strptime(date_match.group(1).upper(), "%d %b %Y").date()
        short_value_hkd = float(total_match.group(1).replace(",", ""))
    except ValueError:
        return unavailable_metric(
            target.isoformat(),
            HKEX_SHORT_SELL_SOURCE,
            "HKEX short-selling report returned an invalid value.",
        )

    if not is_day_close:
        return unavailable_metric(
            target.isoformat(),
            HKEX_SHORT_SELL_SOURCE,
            "Current HKEX short-selling page is a morning-close snapshot, so a full-day ratio is not comparable.",
        )

    total_turnover = turnover_map.get(record_date)
    if total_turnover is None or total_turnover <= 0:
        return unavailable_metric(
            target.isoformat(),
            HKEX_SHORT_SELL_SOURCE,
            "A matched full-day Main Board turnover figure was not available for the same date.",
        )

    ratio_pct = (short_value_hkd / total_turnover) * 100.0
    return build_metric(
        target_date=target.isoformat(),
        value=ratio_pct,
        display_value=format_percent(ratio_pct, digits=2),
        source=HKEX_SHORT_SELL_SOURCE,
        as_of=record_date,
        status="live_local",
        note="Short-selling turnover as a share of matched Main Board turnover.",
    )


def _short_sell_metric_from_payload(report_date: str, short_sell_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(short_sell_data, dict) or short_sell_data.get("status") not in {"ok", "partial"}:
        return None
    data = short_sell_data.get("data", {}) or {}
    meta = short_sell_data.get("meta", {}) or {}
    market = data.get("market", {}) or {}
    ratio = market.get("short_ratio_pct")
    if ratio is None:
        return None
    return build_metric(
        target_date=report_date,
        value=float(ratio),
        display_value=format_percent(float(ratio), digits=2),
        source=meta.get("source", HKEX_SHORT_SELL_SOURCE),
        as_of=meta.get("effective_date", report_date),
        status="live_local",
        note="Official HKEX stock-level short-selling table, aggregated as a share of total market turnover.",
    )


def _stock_connect_metric(
    report_date: str,
    stock_connect_data: Optional[Dict[str, Any]],
    key: str,
    label: str,
) -> Optional[Dict[str, Any]]:
    if not isinstance(stock_connect_data, dict) or stock_connect_data.get("status") not in {"ok", "partial"}:
        return None
    data = stock_connect_data.get("data", {}) or {}
    meta = stock_connect_data.get("meta", {}) or {}
    payload = data.get(key, {}) or {}
    net_buy = payload.get("net_buy")
    total_turnover = payload.get("total_turnover")
    # Southbound trades HK-listed shares (HKD); Northbound trades A-shares (RMB).
    formatter = format_rmb_billions if key == "northbound" else format_hkd_billions
    if net_buy is not None:
        display_value = f"Net {formatter(float(net_buy) * 1_000_000.0)} | turnover {formatter(float(total_turnover or 0) * 1_000_000.0)}"
        value = float(net_buy) * 1_000_000.0
        note = f"{label} net buy is calculated from HKEX disclosed buy and sell turnover."
    elif total_turnover is not None:
        # Turnover is NOT net flow: never store it as the metric value, or the
        # investor lens would report "net buying" from a figure that is
        # explicitly "net not reported".
        display_value = f"Turnover {formatter(float(total_turnover) * 1_000_000.0)} | net not reported"
        value = None
        note = f"{label} total turnover is available; net-buy is not disclosed in this public daily file."
    else:
        return None
    return build_metric(
        target_date=report_date,
        value=value,
        display_value=display_value,
        source=meta.get("source", "HKEX Stock Connect Historical Daily"),
        as_of=meta.get("effective_date", report_date),
        status="live_local",
        note=note,
    )


def _ah_premium_metric(report_date: str, ah_premium_data: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(ah_premium_data, dict) or ah_premium_data.get("status") not in {"ok", "partial"}:
        return None
    data = ah_premium_data.get("data", {}) or {}
    meta = ah_premium_data.get("meta", {}) or {}
    rows = data.get("rows", []) or []

    # Prefer the fixed basket: the covered-pair average is taken over whichever
    # names resolved today, so its day-over-day change is dominated by
    # composition rather than by the market.
    basket = data.get("fixed_basket_premium")
    if basket is not None and data.get("fixed_basket_complete"):
        headline = float(basket)
        note = (
            f"Fixed {data.get('fixed_basket_size')}-name basket, equal-weighted, both legs priced on the "
            f"same date; comparable across dates. Covered-pair average was "
            f"{float(data['average_premium']):+.2f}% across {len(rows)} pairs."
            if data.get("average_premium") is not None
            else f"Fixed {data.get('fixed_basket_size')}-name basket, equal-weighted; comparable across dates."
        )
    else:
        average = data.get("average_premium")
        if average is None:
            return None
        headline = float(average)
        missing = data.get("fixed_basket_missing", []) or []
        note = (
            f"Equal-weighted average across {len(rows)} A/H pairs that resolved today. "
            "Composition varies by day, so the level is not comparable with prior reports"
            + (f" (fixed basket incomplete: missing {', '.join(missing)})" if missing else "")
            + ". Use dispersion rather than the average alone."
        )

    return build_metric(
        target_date=report_date,
        value=headline,
        display_value=format_percent(headline, digits=2),
        source=meta.get("source", "Public AH premium model"),
        as_of=meta.get("effective_date", report_date),
        status="live_public",
        note=note,
    )


def fetch_hk_local_data(
    report_date: str,
    short_sell_data: Optional[Dict[str, Any]] = None,
    stock_connect_data: Optional[Dict[str, Any]] = None,
    ah_premium_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    target = parse_target_date(report_date)
    errors: List[Dict[str, str]] = []
    turnover_history = _collect_turnover_history(target, errors=errors)
    turnover_map = {item["date"]: float(item["turnover_hkd"]) for item in turnover_history}

    metrics: Dict[str, Dict[str, Any]] = {
        "main_board_turnover": unavailable_metric(
            report_date,
            HKEX_SOURCE,
            "No HKEX Main Board turnover page was available for the requested date window.",
        ),
        "turnover_vs_20d": unavailable_metric(
            report_date,
            HKEX_SOURCE,
            "A 20-session turnover comparison could not be calculated.",
        ),
        "short_selling_ratio": unavailable_metric(
            report_date,
            HKEX_SHORT_SELL_SOURCE,
            "A comparable full-day HKEX short-selling ratio could not be derived for the requested date.",
        ),
        "southbound_net_flow": unavailable_metric(
            report_date,
            "HKEX Stock Connect Historical Daily",
            "HKEX Stock Connect daily data could not be retrieved or parsed for the requested date window.",
        ),
        "northbound_net_flow": unavailable_metric(
            report_date,
            "HKEX Stock Connect Historical Daily",
            "Northbound full-day net-buy is unavailable in the current HKEX public daily file.",
        ),
        "ah_premium_index": unavailable_metric(
            report_date,
            "Public Yahoo Finance quotes - calculated A/H premium",
            "A/H premium could not be derived from the configured public quote window.",
        ),
        "hibor_1m": unavailable_metric(
            report_date,
            HKMA_SOURCE,
            "HKMA liquidity record was not refreshed for this date.",
        ),
        "aggregate_balance": unavailable_metric(
            report_date,
            HKMA_SOURCE,
            "HKMA liquidity record was not refreshed for this date.",
        ),
        "base_rate": unavailable_metric(
            report_date,
            HKMA_SOURCE,
            "HKMA liquidity record was not refreshed for this date.",
        ),
        "linked_exchange_band": unavailable_metric(
            report_date,
            HKMA_SOURCE,
            "HKMA liquidity record was not refreshed for this date.",
        ),
    }

    if turnover_history:
        latest = turnover_history[0]
        metrics["main_board_turnover"] = build_metric(
            target_date=report_date,
            value=latest["turnover_hkd"],
            display_value=format_hkd_billions(latest["turnover_hkd"]),
            source=HKEX_SOURCE,
            as_of=latest["date"],
            status="live_local",
            note="Main Board turnover is a fast check for whether Hong Kong participation was broad enough to trust the move.",
        )

        prior = turnover_history[1 : TURNOVER_AVERAGE_WINDOW + 1]
        if len(prior) == TURNOVER_AVERAGE_WINDOW:
            average_turnover = mean(item["turnover_hkd"] for item in prior)
            ratio = latest["turnover_hkd"] / average_turnover if average_turnover else None
            delta_pct = ((ratio - 1.0) * 100.0) if ratio is not None else None
            metrics["turnover_vs_20d"] = build_metric(
                target_date=report_date,
                value=ratio,
                display_value=f"{format_ratio(ratio)} | {delta_pct:+.0f}% vs 20D",
                source=HKEX_SOURCE,
                as_of=latest["date"],
                status="live_local",
                note=f"Trailing 20-session average turnover was {format_hkd_billions(average_turnover)}.",
            )

    metrics["short_selling_ratio"] = _short_sell_metric_from_payload(report_date, short_sell_data) or _fetch_short_sell_snapshot(
        target,
        turnover_map,
        errors=errors,
    )
    metrics["southbound_net_flow"] = _stock_connect_metric(report_date, stock_connect_data, "southbound", "Southbound") or metrics["southbound_net_flow"]
    metrics["northbound_net_flow"] = _stock_connect_metric(report_date, stock_connect_data, "northbound", "Northbound") or metrics["northbound_net_flow"]
    metrics["ah_premium_index"] = _ah_premium_metric(report_date, ah_premium_data) or metrics["ah_premium_index"]

    hkma_record = _fetch_hkma_record(target, errors=errors)
    if hkma_record:
        record_date = str(hkma_record.get("end_of_date", ""))
        hibor_1m = hkma_record.get("hibor_fixing_1m")
        aggregate_balance = hkma_record.get("closing_balance")
        base_rate = hkma_record.get("disc_win_base_rate")
        weak_side = hkma_record.get("cu_weakside")
        strong_side = hkma_record.get("cu_strongside")

        if isinstance(hibor_1m, (int, float)):
            metrics["hibor_1m"] = build_metric(
                target_date=report_date,
                value=float(hibor_1m),
                display_value=format_percent(float(hibor_1m)),
                source=HKMA_SOURCE,
                as_of=record_date,
                status="live_local",
                note="1M HIBOR is the cleanest quick read for Hong Kong funding conditions and equity-duration pressure.",
            )

        if isinstance(aggregate_balance, (int, float)):
            aggregate_hkd = float(aggregate_balance) * 1_000_000.0
            metrics["aggregate_balance"] = build_metric(
                target_date=report_date,
                value=aggregate_hkd,
                display_value=format_hkd_billions(aggregate_hkd),
                source=HKMA_SOURCE,
                as_of=record_date,
                status="live_local",
                note="Aggregate Balance helps frame whether linked-rate liquidity conditions are tight or comfortable.",
            )

        if isinstance(base_rate, (int, float)):
            metrics["base_rate"] = build_metric(
                target_date=report_date,
                value=float(base_rate),
                display_value=format_percent(float(base_rate)),
                source=HKMA_SOURCE,
                as_of=record_date,
                status="live_local",
                note="The Discount Window Base Rate anchors Hong Kong funding expectations under the linked-exchange regime.",
            )

        if isinstance(weak_side, (int, float)) and isinstance(strong_side, (int, float)):
            metrics["linked_exchange_band"] = build_metric(
                target_date=report_date,
                value={"strong_side": float(strong_side), "weak_side": float(weak_side)},
                display_value=f"{float(strong_side):.4f} to {float(weak_side):.4f}",
                source=HKMA_SOURCE,
                as_of=record_date,
                status="live_local",
                note="Official Convertibility Undertakings define the linked-rate stress boundaries for USD/HKD.",
            )

    available_count = sum(
        1
        for item in metrics.values()
        if isinstance(item, dict) and str(item.get("status", "")).startswith(("live", "stale"))
    )

    return {
        "status": _status_from_metrics(metrics),
        "data": metrics,
        "meta": {
            "report_date": report_date,
            "turnover_points": len(turnover_history),
            "available_metrics": available_count,
            "turnover_effective_date": turnover_history[0]["date"].isoformat() if turnover_history else "",
            "hkma_effective_date": str(hkma_record.get("end_of_date", "")) if hkma_record else "",
            "errors": summarize_error_records(errors, limit=12),
        },
    }
