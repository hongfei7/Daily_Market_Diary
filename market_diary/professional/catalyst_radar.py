from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from market_diary.professional.analytics_briefing import _dedupe_key


INK = "#13202b"
SLATE = "#5d6973"
MUTED = "#7a858e"
LINE = "#d7dde1"
PAPER = "#fbfaf7"
WHITE = "#ffffff"
NAVY = "#123a56"
BLUE = "#2274a5"
AMBER = "#b45309"
MONITOR = "#69757f"
CATALYST_RADAR_LAYOUT_VERSION = "catalyst-radar-v2"


def _text(value: Any) -> str:
    return str(value or "").strip()


def _short(value: Any, width: int) -> str:
    return textwrap.shorten(_text(value), width=width, placeholder="…")


def _event_name(item: Dict[str, Any]) -> str:
    return _text(item.get("event") or item.get("indicator") or item.get("description") or item.get("title"))


def _event_date(item: Dict[str, Any], default: str = "") -> str:
    return _text(item.get("date") or item.get("event_date") or item.get("catalyst_date") or default)


def _event_time(item: Dict[str, Any]) -> str:
    return _text(item.get("time") or item.get("release_time"))


def _importance_score(item: Dict[str, Any]) -> float:
    raw = item.get("score", item.get("importance", item.get("impact", "")))
    if isinstance(raw, (int, float)):
        return float(raw)
    normalized = _text(raw).lower()
    if normalized in {"critical", "very high"}:
        return 5.0
    if normalized == "high":
        return 4.0
    if normalized in {"medium", "moderate"}:
        return 2.5
    if normalized == "low":
        return 1.0
    return 2.0


def _timing_classification(item: Dict[str, Any], *, date: str, window: str, source: str) -> Tuple[str, str]:
    """Return presentation lane and timing confidence without promoting aggregators.

    A precise-looking date is not, by itself, confirmation. Only an explicit
    confidence flag or an issuer/exchange/official source earns CONFIRMED.
    """
    confidence = _text(
        item.get("date_confidence")
        or item.get("timing_confidence")
        or item.get("confidence")
    ).lower().replace("-", "_").replace(" ", "_")
    confirmed_values = {
        "confirmed",
        "confirmed_date",
        "issuer_confirmed",
        "exchange_confirmed",
        "official",
    }
    reported_values = {"reported", "aggregator_reported", "calendar_reported"}
    estimated_values = {"estimated", "inferred", "indicative", "window"}
    source_text = _text(source).lower()
    official_source = any(
        token in source_text
        for token in (
            "hkex",
            "issuer",
            "company filing",
            "official",
            "regulator",
            "federal reserve",
            "people's bank",
            "pboc",
            "hkma",
        )
    )

    if date and (confidence in confirmed_values or official_source):
        return "confirmed", "confirmed"
    if date and confidence in estimated_values:
        return "window", "estimated"
    if date and (confidence in reported_values or not confidence):
        return "window", "reported"
    if window:
        return "window", "estimated"
    return "monitor", "undated"


def _row(
    *,
    lane: str,
    date: str,
    time: str,
    category: str,
    event: str,
    entity: str = "",
    impact: str = "",
    source: str = "",
    score: float = 2.0,
    timing_confidence: str = "undated",
) -> Dict[str, Any]:
    return {
        "lane": lane,
        "date": date,
        "time": time,
        "category": category or "Catalyst",
        "event": event,
        "entity": entity,
        "impact": impact,
        "source": source,
        "score": score,
        "timing_confidence": timing_confidence,
    }


def _append_event_rows(
    rows: List[Dict[str, Any]],
    items: Iterable[Any],
    *,
    category: str,
    default_date: str = "",
    default_source: str = "",
) -> None:
    for raw in items:
        if not isinstance(raw, dict):
            continue
        event = _event_name(raw)
        if not event:
            continue
        date = _event_date(raw, default_date)
        window = _text(raw.get("window") or raw.get("date_window"))
        source = _text(raw.get("source") or default_source)
        lane, timing_confidence = _timing_classification(
            raw,
            date=date,
            window=window,
            source=source,
        )
        entity = _text(raw.get("country") or raw.get("bank") or raw.get("ticker") or raw.get("company"))
        rows.append(
            _row(
                lane=lane,
                date=date or window,
                time=_event_time(raw),
                category=_text(raw.get("category") or raw.get("type") or raw.get("status") or category),
                event=event,
                entity=entity,
                impact=_text(raw.get("impact") or raw.get("why_it_matters") or raw.get("importance")),
                source=source,
                score=_importance_score(raw),
                timing_confidence=timing_confidence,
            )
        )


def _dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Collapse the same event arriving from several feeds.

    The radar aggregates six sources, and the macro calendar and risk feed are
    now both driven by the same release schedule, so one release could arrive
    twice under slightly different labels. Keying on the raw event string let
    "China LPR (1Y / 5Y)" and "CN China LPR (1Y / 5Y)" both through. The entity
    is excluded from the key for the same reason: one feed carries it, the
    other does not.
    """
    result: List[Dict[str, Any]] = []
    seen = set()
    for item in rows:
        key = (_dedupe_key(_text(item.get("event"))), _text(item.get("date")))
        if not key[0] or key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def build_catalyst_radar_rows(bundle: Dict[str, Any]) -> Dict[str, Any]:
    """Build a no-fabrication catalyst queue with explicit date confidence."""

    meta = bundle.get("meta", {}) or {}
    briefing_date = _text(meta.get("briefing_date") or meta.get("report_date"))
    today_forward = bundle.get("today_forward", {}) or {}
    risk = bundle.get("risk", {}) or {}
    rows: List[Dict[str, Any]] = []

    _append_event_rows(rows, bundle.get("catalysts", []) or [], category="Catalyst")
    _append_event_rows(rows, today_forward.get("today_catalysts", []) or [], category="Catalyst", default_date=briefing_date)
    _append_event_rows(rows, today_forward.get("next_catalysts", []) or [], category="Catalyst")
    # Only forward-looking macro: a Monday radar must not lead with "Released"
    # (past) prints, which would be backward-looking rather than week-ahead.
    upcoming_macro = [
        item
        for item in (bundle.get("macro_agenda", []) or [])
        if str(item.get("status", "")).lower() in {"upcoming", "central bank"}
    ]
    _append_event_rows(rows, upcoming_macro, category="Macro", default_date=briefing_date)
    _append_event_rows(rows, today_forward.get("today_macro", []) or [], category="Macro", default_date=briefing_date)
    _append_event_rows(rows, risk.get("upcoming_events", []) or [], category="Risk event")

    for bucket, items in (bundle.get("watchlists", {}) or {}).items():
        for item in (items or []):
            if not isinstance(item, dict):
                continue
            event = _text(item.get("upcoming_catalyst") or item.get("catalyst"))
            if not event:
                continue
            date = _event_date(item)
            name = _text(item.get("name"))
            ticker = _text(item.get("ticker"))
            entity = " · ".join(part for part in (ticker, name) if part)
            source = _text(item.get("source")) or "Research watchlist"
            lane, timing_confidence = _timing_classification(
                item,
                date=date,
                window=_text(item.get("window") or item.get("date_window")),
                source=source,
            )
            rows.append(
                _row(
                    lane=lane,
                    date=date,
                    time="",
                    category=_text(bucket).replace("_", " ").title() or "Watchlist",
                    event=event,
                    entity=entity,
                    impact=_text(item.get("thesis") or item.get("why_it_matters")),
                    source=source,
                    score=3.0 if "core" in _text(bucket).lower() else 2.5,
                    timing_confidence=timing_confidence,
                )
            )

    rows = _dedupe_rows(rows)
    lane_order = {"confirmed": 0, "window": 1, "monitor": 2}
    rows.sort(
        key=lambda item: (
            lane_order.get(_text(item.get("lane")), 3),
            _text(item.get("date")) or "9999-99-99",
            -float(item.get("score", 0) or 0),
        )
    )

    confirmed = [item for item in rows if item["lane"] == "confirmed"]
    windows = [item for item in rows if item["lane"] == "window"]
    monitors = [item for item in rows if item["lane"] == "monitor"]
    selected = (confirmed[:4] + windows[:2] + monitors[: max(0, 5 - min(4, len(confirmed)) - min(2, len(windows)))])[:5]

    company_events = bundle.get("company_events", {}) or {}
    announcements = company_events.get("announcements", []) or []
    watchlist_tickers = {
        _text(item.get("ticker")).lstrip("0").lower()
        for items in (bundle.get("watchlists", {}) or {}).values()
        for item in (items or [])
        if isinstance(item, dict) and _text(item.get("ticker"))
    }
    explicit_watchlist_announcements = company_events.get("watchlist_announcements", []) or []
    matched_announcements = explicit_watchlist_announcements or [
        item
        for item in announcements
        if isinstance(item, dict) and _text(item.get("ticker")).lstrip("0").lower() in watchlist_tickers
    ]
    issuer_signals: List[Dict[str, Any]] = []
    seen_issuers = set()
    for item in sorted(
        (raw for raw in matched_announcements if isinstance(raw, dict)),
        key=lambda raw: (_text(raw.get("release_time")), _importance_score(raw)),
        reverse=True,
    ):
        entity = " · ".join(part for part in (_text(item.get("ticker")), _text(item.get("company"))) if part)
        dedupe_key = entity.lower() or _event_name(item).lower()
        if not dedupe_key or dedupe_key in seen_issuers:
            continue
        seen_issuers.add(dedupe_key)
        issuer_signals.append(
            {
                "date": _event_date(item) or _event_time(item),
                "entity": entity,
                "event": _event_name(item),
                "grade": _text(item.get("grade")),
                "score": _importance_score(item),
                "source": _text(item.get("source")) or "HKEXnews",
            }
        )
        if len(issuer_signals) >= 3:
            break

    return {
        "rows": selected,
        "all_rows": rows,
        "issuer_signals": issuer_signals,
        "issuer_review": {
            "reviewed": int((company_events.get("hkex_meta", {}) or {}).get("available_count", len(announcements)) or 0),
            "watchlist_matches": len(matched_announcements),
        },
        "counts": {
            "confirmed": len(confirmed),
            "window": len(windows),
            "monitor": len(monitors),
        },
        "next_hk_open": _text(
            (bundle.get("day_mode", {}) or {}).get("target_hk_session")
            or (bundle.get("day_mode", {}) or {}).get("next_hk_trading_day")
        ),
    }


def _lane_style(item: Dict[str, Any]) -> Tuple[str, str]:
    lane = _text(item.get("lane"))
    confidence = _text(item.get("timing_confidence"))
    if lane == "confirmed" and confidence == "confirmed":
        return "CONFIRMED", NAVY
    if lane == "window":
        return ("REPORTED", BLUE) if confidence == "reported" else ("ESTIMATED", AMBER)
    return "MONITOR", MONITOR


def _date_label(item: Dict[str, Any]) -> str:
    value = " ".join(part for part in (_text(item.get("date")), _text(item.get("time"))) if part)
    return value or "Date unconfirmed"


def _draw_queue(fig, payload: Dict[str, Any]) -> None:
    rows = payload.get("rows", []) or []
    left, right = 0.055, 0.945
    top = 0.650
    row_h = 0.079

    fig.text(left, top + 0.047, "FORWARD DECISION QUEUE", fontsize=10.2, fontweight="bold", color=NAVY)
    fig.text(right, top + 0.047, f"{len(rows)} SHOWN  ·  DATE CONFIDENCE IS EXPLICIT", fontsize=8.7, color=MUTED, ha="right")
    fig.lines.append(plt.Line2D([left, right], [top + 0.025, top + 0.025], transform=fig.transFigure, color=NAVY, linewidth=1.25))

    if not rows:
        fig.add_artist(Rectangle((left, top - row_h), right - left, row_h, transform=fig.transFigure, facecolor="#f1f3f4", edgecolor=LINE, linewidth=0.7))
        fig.text(left + 0.018, top - 0.037, "NO CONFIRMED DATE", fontsize=9.5, fontweight="bold", color=MONITOR)
        fig.text(left + 0.205, top - 0.037, "No calendar or watchlist catalyst is populated; do not infer a date.", fontsize=10.2, color=INK)
        return

    y = top
    for index, item in enumerate(rows):
        lane_label, lane_color = _lane_style(item)
        face = WHITE if index % 2 == 0 else "#f6f7f7"
        fig.add_artist(Rectangle((left, y - row_h), right - left, row_h, transform=fig.transFigure, facecolor=face, edgecolor=LINE, linewidth=0.55))
        fig.add_artist(Rectangle((left, y - row_h), 0.006, row_h, transform=fig.transFigure, facecolor=lane_color, edgecolor=lane_color, linewidth=0))
        fig.text(left + 0.016, y - 0.026, lane_label, fontsize=8.5, fontweight="bold", color=lane_color, va="center")
        fig.text(left + 0.108, y - 0.026, _short(_date_label(item), 27), fontsize=9.2, fontweight="bold", color=INK, va="center")
        identity = " · ".join(part for part in (_text(item.get("entity")), _text(item.get("category"))) if part)
        fig.text(left + 0.335, y - 0.026, _short(identity, 48), fontsize=8.8, color=SLATE, va="center")
        fig.text(left + 0.108, y - 0.059, _short(item.get("event"), 82), fontsize=10.6, fontweight="bold", color=INK, va="center")
        y -= row_h


def _draw_issuer_signals(fig, payload: Dict[str, Any]) -> None:
    signals = payload.get("issuer_signals", []) or []
    left, right = 0.055, 0.945
    y = 0.090
    fig.text(left, y + 0.108, "LATEST ISSUER READ-THROUGH", fontsize=10.2, fontweight="bold", color=NAVY)
    fig.text(left + 0.335, y + 0.108, "Backward-looking evidence, not a future event", fontsize=8.8, color=MUTED)
    fig.lines.append(plt.Line2D([left, right], [y + 0.089, y + 0.089], transform=fig.transFigure, color=LINE, linewidth=0.9))

    if not signals:
        review = payload.get("issuer_review", {}) or {}
        fig.add_artist(Rectangle((left, y), right - left, 0.055, transform=fig.transFigure, facecolor=WHITE, edgecolor=LINE, linewidth=0.7))
        fig.text(
            left + 0.014,
            y + 0.028,
            f"{review.get('reviewed', 0)} HKEX filings reviewed  ·  {review.get('watchlist_matches', 0)} configured-watchlist matches  ·  no issuer event promoted",
            fontsize=9.3,
            color=SLATE,
            va="center",
        )
        return

    width = (right - left - 0.026) / 3
    for index, item in enumerate(signals[:3]):
        x = left + index * (width + 0.013)
        fig.add_artist(Rectangle((x, y), width, 0.055, transform=fig.transFigure, facecolor=WHITE, edgecolor=LINE, linewidth=0.7))
        fig.add_artist(Rectangle((x, y), 0.005, 0.055, transform=fig.transFigure, facecolor=AMBER, edgecolor=AMBER, linewidth=0))
        fig.text(x + 0.013, y + 0.038, _short(f"{item.get('date', '')} · {item.get('entity', '')}", 38), fontsize=8.4, color=SLATE, va="center")
        fig.text(x + 0.013, y + 0.018, _short(item.get("event"), 40), fontsize=9.3, fontweight="bold", color=INK, va="center")


def generate_catalyst_radar(bundle: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    payload = build_catalyst_radar_rows(bundle)
    counts = payload["counts"]
    meta = bundle.get("meta", {}) or {}
    briefing_date = _text(meta.get("briefing_date") or meta.get("report_date"))
    next_open = payload.get("next_hk_open") or "N/A"

    plt.style.use("default")
    fig = plt.figure(figsize=(10.6, 11.8), facecolor=PAPER)
    fig.text(0.055, 0.955, f"CATALYST & EVENT RADAR  /  {briefing_date}", fontsize=9.2, fontweight="bold", color=NAVY)
    fig.text(0.055, 0.902, "What can change the Hong Kong setup next?", fontsize=22.5, fontweight="bold", color=INK)
    fig.text(
        0.055,
        0.862,
        "Issuer/exchange-confirmed dates lead; reported or estimated timing remains visibly distinct.",
        fontsize=10.8,
        color=SLATE,
    )

    metrics = [
        ("CONFIRMED", str(counts["confirmed"]), NAVY),
        ("REPORTED / EST.", str(counts["window"]), BLUE),
        ("MONITORING", str(counts["monitor"]), MONITOR),
        ("NEXT HK OPEN", next_open, INK),
    ]
    x_positions = [0.055, 0.275, 0.495, 0.715]
    for (label, value, color), x in zip(metrics, x_positions):
        fig.lines.append(plt.Line2D([x, x + 0.18], [0.815, 0.815], transform=fig.transFigure, color=LINE, linewidth=0.9))
        fig.text(x, 0.787, label, fontsize=8.4, fontweight="bold", color=MUTED)
        fig.text(x, 0.746, _short(value, 22), fontsize=15.2, fontweight="bold", color=color)

    _draw_queue(fig, payload)
    _draw_issuer_signals(fig, payload)
    fig.text(
        0.055,
        0.025,
        "Decision rule: proximity does not equal materiality. Validate each event against the thesis, expected range and invalidation trigger in the written brief.",
        fontsize=8.8,
        color=MUTED,
    )
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)

    if counts["confirmed"]:
        read = f"{counts['confirmed']} issuer/exchange-confirmed event(s) lead; {counts['window']} reported or estimated timing item(s) remain distinct."
    else:
        read = f"No issuer/exchange-confirmed event date is populated; {counts['window']} reported or estimated timing item(s) and {counts['monitor']} undated trigger(s) remain explicit."
    return {
        "path": os.path.basename(output_path),
        "title": "Catalyst & Event Radar",
        "caption": read,
        "source": "Public calendars, issuer disclosures and configured research watchlists",
        "counts": counts,
        "layout_version": CATALYST_RADAR_LAYOUT_VERSION,
    }
