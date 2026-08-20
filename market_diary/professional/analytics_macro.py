from __future__ import annotations

from typing import Any, Dict, List


def _macro_profile(indicator: str, config: Dict[str, Any]) -> Dict[str, Any]:
    indicator_upper = indicator.upper()
    for key, profile in (config.get("macro_indicator_map") or {}).items():
        if key.upper() in indicator_upper:
            return profile
    return {
        "impact": "Watch whether it changes the day's core market narrative",
        "industries": ["Market-wide"],
        "beat_direction": "If the print beats, check whether the market reprices materially",
        "miss_direction": "If the print misses, watch for a style or rates pivot",
    }


def _macro_detail(item: Dict[str, Any], released: bool) -> str:
    """Describe an event by what is actually known about it.

    Rendering "Forecast None / Prior None" advertised absent fields. When no
    forecast source is configured, the transmission channel and the timing
    confidence are the informative parts.
    """
    values = []
    if released and str(item.get("actual", "") or "").strip():
        values.append(f"Actual {item['actual']}")
    if str(item.get("forecast", "") or "").strip():
        values.append(f"Forecast {item['forecast']}")
    if str(item.get("previous", "") or "").strip():
        values.append(f"Prior {item['previous']}")
    if values:
        return " / ".join(values)

    parts = []
    if item.get("channel_note"):
        parts.append(str(item["channel_note"]))
    if item.get("note"):
        parts.append(str(item["note"]))
    timing = str(item.get("timing_confidence", "") or "")
    if timing and timing != "exact":
        parts.append(f"scheduled date {timing}")
    if not parts:
        return "No forecast or prior values are sourced for this release."
    return " | ".join(parts)


def build_macro_agenda(report_date: str, macro_data: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    agenda: List[Dict[str, Any]] = []
    calendar = (macro_data or {}).get("calendar", {}) or {}
    released = calendar.get("released", []) or []
    upcoming = calendar.get("upcoming", []) or []
    cb_events = (macro_data or {}).get("central_bank_events", []) or []

    for item in released:
        profile = _macro_profile(item.get("indicator", ""), config)
        surprise = item.get("surprise", "inline")
        direction = profile["beat_direction"] if surprise == "beat" else profile["miss_direction"] if surprise == "miss" else "The print was broadly inline; focus on the second-order market reaction"
        agenda.append(
            {
                "date": report_date,
                "time": item.get("time", ""),
                "country": item.get("country", ""),
                "event": item.get("indicator", ""),
                "status": "Released",
                "impact": profile["impact"],
                "affected_industries": profile["industries"],
                "direction": direction,
                "attention": {"high": 5, "medium": 3, "low": 1}.get(item.get("impact", "medium"), 3),
                "score": 80 + {"high": 15, "medium": 8, "low": 3}.get(item.get("impact", "medium"), 8),
                "detail": _macro_detail(item, released=True),
                "channel": item.get("channel", ""),
                "as_of": item.get("as_of", report_date),
                "source": item.get("source", ""),
                "source_url": item.get("source_url", item.get("url", "")),
            }
        )

    for item in upcoming:
        profile = _macro_profile(item.get("indicator", ""), config)
        agenda.append(
            {
                "date": report_date,
                "time": item.get("time", ""),
                "country": item.get("country", ""),
                "event": item.get("indicator", ""),
                "status": "Upcoming",
                "impact": profile["impact"],
                "affected_industries": profile["industries"],
                "direction": "The result will determine whether the current market theme continues",
                "attention": {"high": 5, "medium": 3, "low": 1}.get(item.get("impact", "medium"), 3),
                "score": 70 + {"high": 15, "medium": 8, "low": 3}.get(item.get("impact", "medium"), 8),
                "detail": _macro_detail(item, released=False),
                "channel": item.get("channel", ""),
                "as_of": item.get("as_of", report_date),
                "source": item.get("source", ""),
                "source_url": item.get("source_url", item.get("url", "")),
            }
        )

    for item in cb_events:
        agenda.append(
            {
                "date": report_date,
                "time": item.get("time", ""),
                "country": item.get("bank", ""),
                "event": f"{item.get('speaker', '')}: {item.get('title', '')}".strip(": "),
                "status": "Central bank",
                "impact": "Policy path, liquidity conditions, and cross-asset risk appetite",
                "affected_industries": ["Technology", "Financials", "Gold"],
                "direction": "Watch for any unexpectedly hawkish or dovish language",
                "attention": 5 if item.get("importance") == "high" else 3,
                "score": 78 if item.get("importance") == "high" else 68,
                "detail": item.get("event_type", "speech"),
                "as_of": item.get("as_of", report_date),
                "source": item.get("source", ""),
                "source_url": item.get("source_url", item.get("url", "")),
            }
        )

    agenda.sort(key=lambda row: row.get("score", 0), reverse=True)
    return agenda
