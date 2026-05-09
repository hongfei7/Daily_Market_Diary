"""China local-rates public adapter."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from market_diary.modules.local_metrics import (
    append_error_record,
    build_metric,
    format_bp,
    format_percent,
    parse_target_date,
    summarize_error_records,
    unavailable_metric,
)


EASTMONEY_TREASURY_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
EASTMONEY_SOURCE = "Eastmoney Treasury Yield History"
EASTMONEY_PARAMS = {
    "reportName": "RPTA_WEB_TREASURYYIELD",
    "columns": "ALL",
    "sortColumns": "SOLAR_DATE",
    "sortTypes": "-1",
    "token": "894050c76af8597a853f5b408b759f5d",
    "pageNumber": "1",
    "pageSize": "120",
}
USER_AGENT = "Daily-Market-Diary/3.0"
REQUEST_TIMEOUT = float(os.environ.get("DMD_PUBLIC_REQUEST_TIMEOUT_SECONDS", "12"))


def _status_from_metrics(metrics: Dict[str, Dict[str, Any]]) -> str:
    statuses = [str(item.get("status", "")) for item in metrics.values() if isinstance(item, dict)]
    if not statuses:
        return "error"
    if all(status == "unavailable" for status in statuses):
        return "error"
    return "ok"


def _fetch_rows() -> List[Dict[str, Any]]:
    response = requests.get(
        EASTMONEY_TREASURY_URL,
        params=EASTMONEY_PARAMS,
        headers={"User-Agent": USER_AGENT},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    return ((payload.get("result", {}) or {}).get("data", []) or [])


def _parse_row_date(row: Dict[str, Any]) -> Optional[str]:
    raw = str(row.get("SOLAR_DATE", "")).strip()
    if not raw:
        return None
    return raw.split(" ")[0]


def _find_rows(target_date: str, rows: List[Dict[str, Any]]) -> tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    target = parse_target_date(target_date)
    current: Optional[Dict[str, Any]] = None
    previous: Optional[Dict[str, Any]] = None

    for row in rows:
        row_date_str = _parse_row_date(row)
        if not row_date_str:
            continue
        try:
            row_date = parse_target_date(row_date_str)
        except ValueError:
            continue
        if row_date <= target:
            current = row
            break

    if current is None:
        return None, None

    current_date = parse_target_date(_parse_row_date(current) or target_date)
    for row in rows:
        row_date_str = _parse_row_date(row)
        if not row_date_str:
            continue
        try:
            row_date = parse_target_date(row_date_str)
        except ValueError:
            continue
        if row_date < current_date:
            previous = row
            break

    return current, previous


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def fetch_china_rates_data(report_date: str) -> Dict[str, Any]:
    errors: List[Dict[str, str]] = []
    metrics: Dict[str, Dict[str, Any]] = {
        "china_10y": unavailable_metric(
            report_date,
            EASTMONEY_SOURCE,
            "China 10Y public history could not be retrieved.",
        ),
        "cn_us_10y_spread": unavailable_metric(
            report_date,
            EASTMONEY_SOURCE,
            "CN-US 10Y spread could not be calculated.",
        ),
    }

    try:
        rows = _fetch_rows()
    except requests.RequestException as exc:
        append_error_record(errors, source=EASTMONEY_SOURCE, message=str(exc), error_type=type(exc).__name__, context=report_date)
        return {
            "status": "error",
            "data": metrics,
            "meta": {"report_date": report_date, "errors": summarize_error_records(errors)},
        }
    except ValueError as exc:
        append_error_record(errors, source=EASTMONEY_SOURCE, message=str(exc), error_type=type(exc).__name__, context=f"{report_date} invalid-json")
        return {
            "status": "error",
            "data": metrics,
            "meta": {"report_date": report_date, "errors": summarize_error_records(errors)},
        }
    except Exception as exc:
        append_error_record(errors, source=EASTMONEY_SOURCE, message=str(exc), error_type=type(exc).__name__, context=report_date)
        return {
            "status": "error",
            "data": metrics,
            "meta": {"report_date": report_date, "errors": summarize_error_records(errors)},
        }

    current, previous = _find_rows(report_date, rows)
    if current is None:
        append_error_record(errors, source=EASTMONEY_SOURCE, message="no row matched the requested date", error_type="LookupError", context=report_date)
        return {
            "status": "error",
            "data": metrics,
            "meta": {"report_date": report_date, "row_count": len(rows), "errors": summarize_error_records(errors)},
        }

    current_date = _parse_row_date(current) or report_date
    china_10y = _safe_float(current.get("EMM00166466"))
    us_10y = _safe_float(current.get("EMG00001310"))

    prev_china_10y = _safe_float(previous.get("EMM00166466")) if previous else None
    prev_us_10y = _safe_float(previous.get("EMG00001310")) if previous else None

    if china_10y is not None:
        change_pct_point = (china_10y - prev_china_10y) if prev_china_10y is not None else None
        metrics["china_10y"] = build_metric(
            target_date=report_date,
            value=china_10y,
            display_value=format_percent(china_10y, digits=2),
            source=EASTMONEY_SOURCE,
            as_of=current_date,
            status="live_public",
            note="China 10Y yield from public cross-market treasury history.",
            change_value=change_pct_point,
            change_display=format_bp(change_pct_point) if change_pct_point is not None else "",
        )

    if china_10y is not None and us_10y is not None:
        spread_pct_point = china_10y - us_10y
        prev_spread = None
        if prev_china_10y is not None and prev_us_10y is not None:
            prev_spread = prev_china_10y - prev_us_10y
        change_pct_point = (spread_pct_point - prev_spread) if prev_spread is not None else None
        metrics["cn_us_10y_spread"] = build_metric(
            target_date=report_date,
            value=spread_pct_point,
            display_value=format_bp(spread_pct_point),
            source=EASTMONEY_SOURCE,
            as_of=current_date,
            status="live_public",
            note="China 10Y minus US 10Y helps frame relative carry and cross-border macro pressure.",
            change_value=change_pct_point,
            change_display=format_bp(change_pct_point) if change_pct_point is not None else "",
        )

    return {
        "status": _status_from_metrics(metrics),
        "data": metrics,
        "meta": {
            "report_date": report_date,
            "row_count": len(rows),
            "effective_date": current_date,
            "previous_date": _parse_row_date(previous) if previous else "",
            "fetched_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "errors": summarize_error_records(errors),
        },
    }
