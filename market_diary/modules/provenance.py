"""Shared provenance records and release-time validation for source payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Mapping


ACTIVE_STATUSES = {
    "ok",
    "cached",
    "partial",
    "partial_public",
    "partial_local",
    "live_local",
    "live_public",
    "live_quote",
    "live_hybrid",
    "stale_local",
    "stale_public",
    "proxy",
    "derived",
}
INVALID_PRODUCTION_STATUSES = {"placeholder", "synthetic", "sample", "mock"}
SOURCE_TYPES = {"official", "public", "licensed", "derived", "cached", "unavailable"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def provenance_record(
    *,
    source_name: str,
    source_url: str,
    as_of: str,
    source_type: str,
    status: str,
    confidence: float,
    note: str = "",
    collected_at: str = "",
) -> Dict[str, Any]:
    """Build the canonical source record used by every production adapter."""
    return {
        "source_name": str(source_name or "").strip(),
        "source_url": str(source_url or "").strip(),
        "as_of": str(as_of or "").strip(),
        "collected_at": str(collected_at or utc_now_iso()).strip(),
        "source_type": str(source_type or "unavailable").strip().lower(),
        "status": str(status or "unavailable").strip().lower(),
        "confidence": max(0.0, min(float(confidence), 1.0)),
        "note": str(note or "").strip(),
    }


def unavailable_record(source_name: str, as_of: str, note: str) -> Dict[str, Any]:
    return provenance_record(
        source_name=source_name,
        source_url="",
        as_of=as_of,
        source_type="unavailable",
        status="unavailable",
        confidence=0.0,
        note=note,
    )


def ensure_payload_provenance(
    payload: Dict[str, Any],
    *,
    source_name: str,
    source_url: str,
    as_of: str,
    source_type: str,
    status: str,
    confidence: float,
    note: str = "",
) -> Dict[str, Any]:
    """Attach aggregate provenance when a legacy adapter has not done so yet."""
    if not isinstance(payload, dict):
        payload = {"status": "unavailable", "data": payload}
    if not isinstance(payload.get("provenance"), list) or not payload.get("provenance"):
        payload["provenance"] = [
            provenance_record(
                source_name=source_name,
                source_url=source_url,
                as_of=as_of,
                source_type=source_type,
                status=status,
                confidence=confidence,
                note=note,
            )
        ]
    return payload


def collect_source_provenance(source_payloads: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        str(name): list((payload or {}).get("provenance", []) or [])
        if isinstance(payload, dict)
        else []
        for name, payload in source_payloads.items()
    }


def audit_source_provenance(source_payloads: Mapping[str, Any]) -> Dict[str, Any]:
    """Validate source identity, freshness fields, and production-safe statuses."""
    errors = []
    warnings = []
    checked = 0
    unavailable = 0

    for source_key, payload in source_payloads.items():
        records = (payload or {}).get("provenance", []) if isinstance(payload, dict) else []
        if not isinstance(records, list) or not records:
            errors.append(f"{source_key}: missing provenance records")
            continue

        for index, record in enumerate(records):
            checked += 1
            label = f"{source_key}[{index}]"
            if not isinstance(record, dict):
                errors.append(f"{label}: provenance record is not an object")
                continue

            source_name = str(record.get("source_name", "") or "").strip()
            source_url = str(record.get("source_url", "") or "").strip()
            as_of = str(record.get("as_of", "") or "").strip()
            collected_at = str(record.get("collected_at", "") or "").strip()
            source_type = str(record.get("source_type", "") or "").strip().lower()
            status = str(record.get("status", "") or "").strip().lower()

            if not source_name:
                errors.append(f"{label}: source_name is required")
            if not as_of:
                errors.append(f"{label}: as_of is required")
            if not collected_at:
                errors.append(f"{label}: collected_at is required")
            if source_type not in SOURCE_TYPES:
                errors.append(f"{label}: unsupported source_type `{source_type}`")
            if status in INVALID_PRODUCTION_STATUSES or source_type in INVALID_PRODUCTION_STATUSES:
                errors.append(f"{label}: non-production source status `{status or source_type}`")
            if status in ACTIVE_STATUSES and source_type not in {"derived", "unavailable"} and not source_url:
                errors.append(f"{label}: active external source requires source_url")
            if status == "unavailable" or source_type == "unavailable":
                unavailable += 1
                warnings.append(f"{label}: source is unavailable")

            try:
                confidence = float(record.get("confidence"))
            except (TypeError, ValueError):
                errors.append(f"{label}: confidence must be numeric")
            else:
                if not 0.0 <= confidence <= 1.0:
                    errors.append(f"{label}: confidence must be between 0 and 1")

    return {
        "status": "error" if errors else "ok",
        "checked_records": checked,
        "unavailable_records": unavailable,
        "errors": errors,
        "warnings": warnings,
    }
