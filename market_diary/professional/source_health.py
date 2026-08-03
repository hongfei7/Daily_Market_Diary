"""Deterministic source-health scoring for the daily research pipeline."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, Iterable, Mapping


HEALTHY_STATUSES = {"ok", "cached", "live_local", "live_public", "live_quote", "live_hybrid"}
DEGRADED_STATUSES = {
    "partial",
    "partial_public",
    "partial_local",
    "stale_local",
    "stale_public",
    "proxy",
    "derived",
}

DEFAULT_SOURCE_POLICIES: Dict[str, Dict[str, Any]] = {
    "market_data": {"critical": True, "max_age_days": 4},
    "hk_local": {"critical": True, "max_age_days": 4},
    "stock_connect": {"critical": False, "max_age_days": 4},
    "ah_premium": {"critical": False, "max_age_days": 4},
    "china_rates": {"critical": False, "max_age_days": 7},
    "sector_news": {"critical": False, "max_age_days": 3},
    "movers": {"critical": False, "max_age_days": 4},
    "macro_calendar": {"critical": False, "max_age_days": 14},
    "risk_feed": {"critical": False, "max_age_days": 4},
    "rss_headlines": {"critical": False, "max_age_days": 3},
}


def _parse_date(value: Any) -> date | None:
    text = str(value or "").strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _status_value(status: Any) -> float:
    normalized = str(status or "").strip().lower()
    if normalized in HEALTHY_STATUSES:
        return 1.0
    if normalized in DEGRADED_STATUSES:
        return 0.6
    return 0.0


def _source_type_value(source_type: Any) -> float:
    return {
        "official": 1.0,
        "licensed": 1.0,
        "public": 0.8,
        "derived": 0.7,
        "cached": 0.6,
        "unavailable": 0.0,
    }.get(str(source_type or "").strip().lower(), 0.0)


def _merge_policies(overrides: Mapping[str, Any] | None) -> Dict[str, Dict[str, Any]]:
    merged = {key: dict(value) for key, value in DEFAULT_SOURCE_POLICIES.items()}
    for key, value in (overrides or {}).items():
        if isinstance(value, Mapping):
            merged.setdefault(str(key), {}).update(dict(value))
    return merged


def _record_age_days(record: Mapping[str, Any], reference: date) -> int | None:
    as_of = _parse_date(record.get("as_of"))
    if as_of is None:
        return None
    return (reference - as_of).days


def _average(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0


def build_source_health(
    source_provenance: Mapping[str, Any],
    *,
    reference_date: str,
    policies: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Score source reliability without conflating availability with authority."""
    reference = _parse_date(reference_date) or date.today()
    resolved_policies = _merge_policies(policies)
    rows = []
    warnings = []

    source_keys = sorted(set(resolved_policies) | set(source_provenance or {}))
    for source_key in source_keys:
        policy = resolved_policies.get(source_key, {})
        records = source_provenance.get(source_key, []) if isinstance(source_provenance, Mapping) else []
        records = [record for record in (records or []) if isinstance(record, Mapping)]
        critical = bool(policy.get("critical", False))
        max_age_days = max(0, int(policy.get("max_age_days", 7) or 7))

        status_values = [_status_value(record.get("status")) for record in records]
        active_records = [record for record in records if _status_value(record.get("status")) > 0]
        ages = [_record_age_days(record, reference) for record in active_records]
        valid_ages = [age for age in ages if age is not None]
        freshest_age = min(valid_ages) if valid_ages else None
        future_dated = sum(1 for age in valid_ages if age < 0)
        stale = freshest_age is None or freshest_age > max_age_days

        completeness = len(active_records) / len(records) if records else 0.0
        availability = _average(status_values)
        authority = _average(_source_type_value(record.get("source_type")) for record in active_records)
        confidence = _average(float(record.get("confidence", 0.0) or 0.0) for record in active_records)
        if freshest_age is None:
            freshness = 0.0
        elif freshest_age < 0:
            freshness = 0.0
        elif freshest_age <= max_age_days:
            freshness = 1.0
        elif freshest_age <= max_age_days * 2:
            freshness = 0.5
        else:
            freshness = 0.0

        score = 100.0 * (
            0.35 * availability
            + 0.20 * completeness
            + 0.20 * authority
            + 0.15 * confidence
            + 0.10 * freshness
        )
        if not records or not active_records or future_dated:
            status = "failed" if critical else "unavailable"
        elif stale or score < 65:
            status = "failed" if critical and freshness == 0.0 else "degraded"
        else:
            status = "healthy"

        if future_dated:
            warnings.append(f"{source_key}: provenance contains a future as-of date.")
        if stale:
            warnings.append(
                f"{source_key}: freshest record is "
                f"{freshest_age if freshest_age is not None else 'unknown'} day(s) old; policy is {max_age_days}."
            )
        if status in {"failed", "unavailable"}:
            warnings.append(f"{source_key}: no decision-grade record was available.")

        rows.append(
            {
                "source": source_key,
                "critical": critical,
                "status": status,
                "score": round(score, 1),
                "records": len(records),
                "active_records": len(active_records),
                "freshest_age_days": freshest_age,
                "max_age_days": max_age_days,
                "dimensions": {
                    "availability": round(availability * 100.0, 1),
                    "completeness": round(completeness * 100.0, 1),
                    "authority": round(authority * 100.0, 1),
                    "confidence": round(confidence * 100.0, 1),
                    "freshness": round(freshness * 100.0, 1),
                },
            }
        )

    critical_failures = [row["source"] for row in rows if row["critical"] and row["status"] == "failed"]
    degraded = [row["source"] for row in rows if row["status"] == "degraded"]
    unavailable = [row["source"] for row in rows if row["status"] == "unavailable"]
    healthy = [row["source"] for row in rows if row["status"] == "healthy"]
    if critical_failures:
        status = "failed"
    elif degraded or unavailable:
        status = "degraded"
    else:
        status = "healthy"

    return {
        "schema_version": "source-health-v1",
        "status": status,
        "reference_date": reference.isoformat(),
        "coverage": {
            "healthy": len(healthy),
            "degraded": len(degraded),
            "unavailable": len(unavailable),
            "total": len(rows),
        },
        "critical_failures": critical_failures,
        "sources": rows,
        "warnings": warnings[:20],
        "methodology": {
            "dimensions": ["availability", "completeness", "authority", "confidence", "freshness"],
            "note": "A high-authority source can still fail on freshness or availability; scores do not replace provenance records.",
        },
    }
