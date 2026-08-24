from __future__ import annotations

from collections import OrderedDict
from datetime import date, datetime
from typing import Any, Dict, List, Optional


def parse_target_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def append_error_record(
    errors: Optional[List[Dict[str, str]]],
    *,
    source: str,
    message: str,
    error_type: str = "Error",
    context: str = "",
) -> None:
    if errors is None:
        return
    errors.append(
        {
            "source": source,
            "message": message,
            "error_type": error_type,
            "context": context,
        }
    )


def summarize_error_records(errors: Optional[List[Dict[str, str]]], limit: int = 20) -> List[str]:
    if not errors:
        return []

    grouped: "OrderedDict[tuple[str, str, str], List[str]]" = OrderedDict()
    for item in errors:
        source = str((item or {}).get("source", "")).strip() or "Unknown source"
        error_type = str((item or {}).get("error_type", "")).strip() or "Error"
        message = str((item or {}).get("message", "")).strip() or "No message"
        context = str((item or {}).get("context", "")).strip()
        key = (source, error_type, message)
        grouped.setdefault(key, [])
        if context:
            grouped[key].append(context)

    output: List[str] = []
    for (source, error_type, message), contexts in grouped.items():
        if not contexts:
            output.append(f"{source}: {error_type}: {message}")
            continue
        if len(contexts) == 1:
            output.append(f"{source} [{contexts[0]}]: {error_type}: {message}")
            continue

        preview = ", ".join(contexts[:3])
        remainder = len(contexts) - 3
        context_summary = preview if remainder <= 0 else f"{preview}, +{remainder} more"
        output.append(f"{source}: {error_type}: {message} (x{len(contexts)}; contexts: {context_summary})")

    return output[: max(int(limit), 1)]


def normalize_as_of(value: Any) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%d %b %Y", "%d %b %Y (%A)"):
            try:
                return datetime.strptime(cleaned, fmt).date()
            except ValueError:
                continue
    return None


def unavailable_metric(
    target_date: str,
    source: str,
    note: str = "",
    status: str = "unavailable",
) -> Dict[str, Any]:
    return {
        "value": None,
        "display_value": "N/A",
        "status": status,
        "source": source,
        "as_of": "",
        "freshness_days": None,
        "quality": "unavailable",
        "fallback_used": False,
        "note": note,
        "change_value": None,
        "change_display": "",
        "target_date": target_date,
    }


def build_metric(
    *,
    target_date: str,
    value: Any,
    display_value: str,
    source: str,
    as_of: Any,
    status: str = "live_local",
    note: str = "",
    fallback_used: bool = False,
    change_value: Optional[float] = None,
    change_display: str = "",
) -> Dict[str, Any]:
    target = parse_target_date(target_date)
    as_of_date = normalize_as_of(as_of)
    if as_of_date is None:
        return unavailable_metric(target_date, source, note or "No as-of date was available for this metric.")

    freshness_days = (target - as_of_date).days
    if freshness_days < 0:
        return unavailable_metric(
            target_date,
            source,
            note or f"Latest source observation ({as_of_date.isoformat()}) is newer than the requested report date.",
        )

    quality = "fresh" if freshness_days <= 1 else "stale"
    resolved_status = status
    if status == "live_local" and quality == "stale":
        resolved_status = "stale_local"
    if status == "live_public" and quality == "stale":
        resolved_status = "stale_public"

    return {
        "value": value,
        "display_value": display_value,
        "status": resolved_status,
        "source": source,
        "as_of": as_of_date.isoformat(),
        "freshness_days": freshness_days,
        "quality": quality,
        "fallback_used": bool(fallback_used),
        "note": note,
        "change_value": change_value,
        "change_display": change_display,
        "target_date": target_date,
    }


def format_percent(value: Optional[float], digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "N/A"
    sign = "+" if signed and value >= 0 else ""
    return f"{sign}{value:.{digits}f}%"


def format_hkd_billions(value_hkd: Optional[float], digits: int = 1) -> str:
    if value_hkd is None:
        return "N/A"
    return f"HK${value_hkd / 1_000_000_000:.{digits}f}bn"


def format_rmb_billions(value_rmb: Optional[float], digits: int = 1) -> str:
    if value_rmb is None:
        return "N/A"
    return f"RMB{value_rmb / 1_000_000_000:.{digits}f}bn"


def format_bp(value_pct_point: Optional[float], digits: int = 1) -> str:
    if value_pct_point is None:
        return "N/A"
    basis_points = value_pct_point * 100.0
    sign = "+" if basis_points >= 0 else ""
    return f"{sign}{basis_points:.{digits}f}bp"


def format_ratio(value: Optional[float], digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}x"
