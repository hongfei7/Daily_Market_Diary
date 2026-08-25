"""Append-only history for Hong Kong local metrics, used for percentile context.

A level means little without a distribution behind it. The report described a
17.0% short-selling ratio as "elevated" purely because it cleared a hard-coded
16% threshold, with no indication of whether that is unusual. Hong Kong market
short-selling routinely runs in the mid-to-high teens, so the label was doing
work the data did not support.

This store follows the same append-only discipline as the signal ledger in
``performance.py``: an observation for a date is written once and never revised,
so percentiles cannot be reshaped by a rerun.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional

SCHEMA_VERSION = "metric-history-v1"

# Percentiles are unstable on tiny samples; below this the report says so
# instead of implying a distribution exists.
MIN_SAMPLE_FOR_PERCENTILE = 20
DEFAULT_WINDOW = 60

TRACKED_METRICS = ("short_selling_ratio", "turnover_vs_20d", "southbound_net_flow", "hibor_1m")

_ARCHIVE_ROW_PATTERNS = {
    "turnover_vs_20d": re.compile(r"^\| Main Board turnover vs 20D \|\s*([+-]?[0-9.]+)x\b"),
    "short_selling_ratio": re.compile(r"^\| Short-selling ratio \|\s*([+-]?[0-9.]+)%"),
    "hibor_1m": re.compile(r"^\| HIBOR 1M \|\s*([+-]?[0-9.]+)%"),
    "southbound_net_flow": re.compile(
        r"^\| Southbound / Northbound net flow \|.*?Southbound Net HK\$([+-]?[0-9.]+)bn\b"
    ),
}


def _default_path(output_dir: str) -> str:
    return os.path.join(output_dir, "performance", "metric_history.json")


def load_history(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"schema_version": SCHEMA_VERSION, "observations": {}}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {"schema_version": SCHEMA_VERSION, "observations": {}}
    if not isinstance(payload, dict):
        return {"schema_version": SCHEMA_VERSION, "observations": {}}
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("observations", {})
    return payload


def record_observations(
    history: Dict[str, Any],
    report_date: str,
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    """Append today's tracked metric values. Existing dates are never rewritten."""
    observations = history.setdefault("observations", {})
    for key in TRACKED_METRICS:
        item = metrics.get(key) if isinstance(metrics, Mapping) else None
        if not isinstance(item, Mapping):
            continue
        value = item.get("value")
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        series = observations.setdefault(key, {})
        # Append-only: the first value recorded for a date is authoritative.
        series.setdefault(report_date, numeric)
    return history


def backfill_archive_history(history: Dict[str, Any], archive_root: str | Path) -> Dict[str, Any]:
    """Seed missing observations from immutable archived report tables.

    The effective local-data date in the source column is used instead of the
    report publication date, so weekend reports do not create duplicate market
    sessions. Existing observations remain authoritative.
    """
    root = Path(archive_root)
    if not root.exists():
        return history
    observations = history.setdefault("observations", {})
    for report_path in sorted(root.glob("*/morning_briefing.md")):
        try:
            lines = report_path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            dates = re.findall(r"\b\d{4}-\d{2}-\d{2}\b", line)
            if not dates:
                continue
            effective_date = dates[-1]
            for metric, pattern in _ARCHIVE_ROW_PATTERNS.items():
                match = pattern.search(line)
                if not match:
                    continue
                value = float(match.group(1))
                if metric == "southbound_net_flow":
                    value *= 1_000_000_000.0
                observations.setdefault(metric, {}).setdefault(effective_date, value)
                break
    return history


def save_history(history: Mapping[str, Any], path: str) -> None:
    payload = dict(history)
    # A persisted file must always identify its own schema, whatever the caller
    # passed in.
    payload.setdefault("schema_version", SCHEMA_VERSION)
    payload.setdefault("observations", {})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _series_before(history: Mapping[str, Any], metric: str, report_date: str, window: int) -> List[float]:
    """Values strictly before ``report_date``, most recent ``window`` first."""
    series = (history.get("observations", {}) or {}).get(metric, {}) or {}
    dated = [(date, value) for date, value in series.items() if str(date) < str(report_date)]
    dated.sort(key=lambda pair: pair[0], reverse=True)
    return [float(value) for _, value in dated[:window]]


def percentile_context(
    history: Mapping[str, Any],
    metric: str,
    value: Optional[float],
    report_date: str,
    window: int = DEFAULT_WINDOW,
) -> Dict[str, Any]:
    """Locate ``value`` inside its own trailing distribution.

    Returns ``available=False`` when there is not enough history to make a
    percentile meaningful, so callers can fall back to an explicitly absolute
    statement rather than implying context that does not exist.
    """
    if value is None:
        return {"available": False, "reason": "no_value", "sample": 0}

    sample = _series_before(history, metric, report_date, window)
    if len(sample) < MIN_SAMPLE_FOR_PERCENTILE:
        return {
            "available": False,
            "reason": "insufficient_history",
            "sample": len(sample),
            "required": MIN_SAMPLE_FOR_PERCENTILE,
        }

    below = sum(1 for item in sample if item < value)
    ties = sum(1 for item in sample if item == value)
    # Midpoint rule keeps repeated readings from pinning the result at an extreme.
    rank = (below + 0.5 * ties) / len(sample) * 100.0

    if rank >= 90:
        band = "very high"
    elif rank >= 75:
        band = "high"
    elif rank <= 10:
        band = "very low"
    elif rank <= 25:
        band = "low"
    else:
        band = "typical"

    return {
        "available": True,
        "percentile": round(rank, 1),
        "band": band,
        "sample": len(sample),
        "window": window,
        "median": round(sorted(sample)[len(sample) // 2], 4),
    }


def describe(context: Mapping[str, Any]) -> str:
    """Render a percentile context as a short parenthetical for report prose."""
    if not context.get("available"):
        if context.get("reason") == "insufficient_history":
            return f"no percentile yet: {context.get('sample', 0)}/{context.get('required', 0)} sessions of history"
        return ""
    return f"{_ordinal(context['percentile'])} pct of the last {context['sample']} sessions, {context['band']}"


def _ordinal(value: float) -> str:
    number = int(round(value))
    if 10 <= number % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"
