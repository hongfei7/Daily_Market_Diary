from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from market_diary.professional.report_formatting import (
    _bundle_metric,
    _compact_source_as_of,
    _fmt_alert_pct,
    _fmt_pct,
    _fmt_price,
    _make_table,
    _status_label,
    _summary_pct,
    _summary_price,
    _truncate,
)


def _safe_sentence_clip(text: Any, limit: int = 160) -> str:
    sentence = " ".join(str(text or "").split()).strip()
    if len(sentence) <= limit:
        return sentence
    phrase = _truncate(sentence, limit, suffix="").strip()
    phrase = re.sub(r"\b(?:could|can|would|may|might|should|will)(?:\s+\w+){0,2}$", "", phrase, flags=re.IGNORECASE).strip()
    phrase = re.sub(
        r"\b(?:after|before|during|into|onto|above|below|around|than|via|through|against|despite)\s+(?:any|the|a|an|this|that|these|those|current|next)?$",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip()
    phrase = re.sub(r"\b(?:and|or|but|with|without|to|from|for|of|the|a|an|because|whether|if)$", "", phrase, flags=re.IGNORECASE).strip()
    if phrase and phrase[-1] not in ".!?":
        phrase = phrase.rstrip(" ,;:-|") + "."
    return phrase


def _render_top_items(items: List[Dict[str, Any]], limit: int = 6) -> str:
    if not items:
        return "1. No priority items were available."
    lines = []
    for idx, item in enumerate(items[:limit], 1):
        lines.append(f"{idx}. **{item.get('title', '')}**")
        lines.append(f"   {item.get('bucket', '')} | {_safe_sentence_clip(item.get('summary', ''), 175)}")
    return "\n".join(lines)


def _render_selected_news(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    items = llm_sections.get("selected_news", []) or []
    if not items:
        return ""
    lines = ["**Curated overnight stories**"]
    for idx, item in enumerate(items[:5], 1):
        lines.append(f"{idx}. **{_truncate(item.get('headline', ''), 90, suffix='')}**")
        lines.append(f"   Why it matters: {_safe_sentence_clip(item.get('why_it_matters', ''), 175)}")
        lines.append(f"   HK read-through: {_safe_sentence_clip(item.get('hk_market_impact', ''), 175)}")
    return "\n".join(lines)


def _render_global_asset_dashboard(bundle: Dict[str, Any]) -> str:
    rows = [
        ("Equities", "S&P 500", "S&P 500", "US large-cap risk appetite"),
        ("Equities", "Nasdaq 100", "Nasdaq 100", "Growth and duration leadership"),
        ("Equities", "Hang Seng Index", "Hang Seng Index", "Hong Kong broad beta"),
        ("Equities", "Hang Seng TECH ETF", "Hang Seng TECH", "Hong Kong growth / platform read-through"),
        ("Equities", "CSI 300", "CSI 300", "Mainland large-cap tone"),
        ("Rates", "10Y Treasury", "US 10Y", "Global discount-rate anchor"),
        ("Rates", "China 10Y", "China 10Y", "China local rates anchor"),
        ("Rates", "CN-US 10Y spread", "CN-US 10Y spread", "Relative carry and macro-pressure gauge"),
        ("FX", "DXY", "DXY", "Dollar liquidity impulse"),
        ("FX", "USD/CNH", "USD/CNH", "Offshore China risk appetite"),
        ("FX", "USD/HKD", "USD/HKD", "Linked-exchange regime stress check"),
        ("Commodities", "Brent Crude", "Brent crude", "Energy / geopolitics"),
        ("Commodities", "Gold", "COMEX gold", "Hedge demand / real yields"),
        ("Commodities", "Copper", "Copper", "Global growth proxy"),
        ("Vol", "VIX", "VIX", "US stress barometer"),
    ]

    table_rows = []
    china_10y_metric = _bundle_metric(bundle, "china_rates", "china_10y")
    spread_metric = _bundle_metric(bundle, "china_rates", "cn_us_10y_spread")

    for category, name, label, note in rows:
        if name == "China 10Y":
            price = china_10y_metric.get("display_value", "N/A")
            pct = china_10y_metric.get("change_display", "")
            read = f"{note} | {_status_label(china_10y_metric.get('status', 'unavailable'))} | {_compact_source_as_of(china_10y_metric)}"
        elif name == "CN-US 10Y spread":
            price = spread_metric.get("display_value", "N/A")
            pct = spread_metric.get("change_display", "")
            read = f"{note} | {_status_label(spread_metric.get('status', 'unavailable'))} | {_compact_source_as_of(spread_metric)}"
        else:
            price = _summary_price(bundle, category, name)
            pct = _summary_pct(bundle, category, name)
            read = note

        last_value = price if isinstance(price, str) else _fmt_price(price)
        move_value = pct if isinstance(pct, str) and pct else _fmt_alert_pct(pct)
        table_rows.append((label, last_value, move_value, _truncate(read, 92, suffix="")))

    return _make_table(["Asset", "Last", "1D move", "Read"], table_rows)


def _pick_metrics_by_name(items: List[Dict[str, Any]], preferred_names: Sequence[str]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used: set[int] = set()
    for preferred in preferred_names:
        preferred_lower = preferred.lower()
        for idx, item in enumerate(items):
            if idx in used:
                continue
            metric = str(item.get("metric", "")).lower()
            if preferred_lower in metric or metric in preferred_lower:
                selected.append(item)
                used.add(idx)
                break
    return selected


def _render_hk_quick_checks(bundle: Dict[str, Any]) -> str:
    rows = bundle.get("hk_quick_checks", []) or []
    rows = _pick_metrics_by_name(
        rows,
        (
            "Main Board turnover vs 20D",
            "Southbound / Northbound net flow",
            "Short-selling ratio",
            "AH premium index",
            "USD/HKD spot vs band",
            "HIBOR 1M",
            "Aggregate Balance",
            "Hong Kong leadership",
        ),
    )
    table_rows = [
        (
            item.get("metric", ""),
            item.get("value", ""),
            _status_label(str(item.get("status", ""))),
            _compact_source_as_of(item),
            _truncate(item.get("note", ""), 220),
        )
        for item in rows
    ]
    return _make_table(["Check", "Value", "Status", "Source / as of", "Why it matters"], table_rows)


def _render_risk_dashboard(bundle: Dict[str, Any]) -> str:
    risk = ((bundle.get("attribution", {}) or {}).get("risk_dashboard", {}) or {})
    if not risk:
        return "Risk dashboard was unavailable."

    components = risk.get("components", []) or []
    lines = [
        f"**Composite risk score:** `{risk.get('score', 'N/A')}/100` | **Regime:** `{risk.get('bucket', 'Mixed')}`",
    ]
    if components:
        rows = [
            (
                item.get("label", ""),
                f"{item.get('delta', 0):+}",
                _truncate(item.get("evidence", ""), 80, suffix=""),
            )
            for item in components[:6]
        ]
        lines.append("")
        lines.append(_make_table(["Component", "Score impact", "Evidence"], rows))
    return "\n".join(lines)


def _render_non_trading_focus(bundle: Dict[str, Any]) -> str:
    focus = bundle.get("non_trading_focus", {}) or {}
    if not focus:
        return ""

    lines: List[str] = ["#### Non-Trading Focus Map", f"- {focus.get('summary', '')}"]
    still_moving = focus.get("still_moving", []) or []
    if still_moving:
        rows = [
            (
                item.get("label", ""),
                item.get("category", ""),
                _fmt_price(item.get("price")),
                _fmt_pct(item.get("change_pct")),
                _truncate(item.get("interpretation", ""), 78, suffix=""),
            )
            for item in still_moving[:6]
        ]
        lines.append(_make_table(["Monitor", "Bucket", "Last", "Move", "Why it matters"], rows))

    event_watch = focus.get("event_watch", []) or []
    if event_watch:
        rows = [
            (
                item.get("channel", ""),
                _truncate(item.get("signal", ""), 66, suffix=""),
                _truncate(item.get("why", ""), 82, suffix=""),
                _truncate(item.get("next_check", ""), 82, suffix=""),
            )
            for item in event_watch[:8]
        ]
        lines.append("\n**Weekend / Holiday Event Docket**")
        lines.append(_make_table(["Channel", "Signal", "Why monitor", "Next check"], rows))

    action_items = focus.get("action_items", []) or []
    if action_items:
        lines.append("\n**Still-moving actions to track**")
        for item in action_items[:5]:
            lines.append(f"- **{item.get('bucket', '')}:** {_truncate(item.get('item', ''), 72, suffix='')} | {_truncate(item.get('read', ''), 96, suffix='')}")

    next_open = focus.get("next_open", []) or []
    if next_open:
        lines.append("\n**Next-open preparation**")
        lines.extend(f"- {line}" for line in next_open[:4])

    return "\n".join(lines)


def _render_weekly_review(bundle: Dict[str, Any]) -> str:
    weekly = bundle.get("weekly_review", {}) or {}
    if not weekly:
        return ""

    lines: List[str] = ["#### Weekly Review Map", f"- {weekly.get('summary', '')}"]
    method_note = weekly.get("method_note", "")
    if method_note:
        lines.append(f"- Method note: {method_note}")

    cross_assets = weekly.get("cross_assets", []) or []
    if cross_assets:
        rows = [
            (
                item.get("asset", ""),
                item.get("latest_move", "N/A"),
                _truncate(item.get("read", ""), 78, suffix=""),
            )
            for item in cross_assets[:8]
        ]
        lines.append("\n**Cross-asset weekly dashboard**")
        lines.append(_make_table(["Asset", "Latest move", "Weekly read"], rows))

    hk_tape = weekly.get("hk_tape", []) or []
    if hk_tape:
        rows = [
            (
                item.get("signal", ""),
                item.get("latest_move", "N/A"),
                _truncate(item.get("read", ""), 78, suffix=""),
            )
            for item in hk_tape[:6]
        ]
        lines.append("\n**Hong Kong / China weekly tape**")
        lines.append(_make_table(["Signal", "Latest move", "Read"], rows))

    trend_summary = weekly.get("trend_summary", {}) or {}
    trend_rows = trend_summary.get("rows", []) or []
    if trend_rows:
        window = trend_summary.get("window", {}) or {}
        lines.append("\n**Five-session trend evidence**")
        if window.get("start") and window.get("end"):
            lines.append(f"_Window: {window.get('start')} to {window.get('end')}_")
        rows = [
            (
                item.get("signal", ""),
                item.get("weekly_change", "N/A"),
                item.get("latest", "N/A"),
                _truncate(item.get("read", ""), 86, suffix=""),
            )
            for item in trend_rows[:6]
        ]
        lines.append(_make_table(["Signal", "Five-session change", "Latest", "Desk read"], rows))

    flow_lines = weekly.get("flow_lines", []) or []
    if flow_lines:
        lines.append("\n**Flow and attribution clues**")
        lines.extend(f"- {_truncate(line, 120, suffix='')}" for line in flow_lines[:4])

    desk_questions = weekly.get("desk_questions", []) or []
    if desk_questions:
        lines.append("\n**Next-week desk questions**")
        lines.extend(f"- {_truncate(line, 128, suffix='')}" for line in desk_questions[:6])

    developments = weekly.get("developments", []) or []
    if developments:
        rows = [
            (
                item.get("bucket", ""),
                _truncate(item.get("item", ""), 64, suffix=""),
                _truncate(item.get("read", ""), 88, suffix=""),
            )
            for item in developments[:6]
        ]
        lines.append("\n**Key developments to retain**")
        lines.append(_make_table(["Bucket", "Item", "Why it matters"], rows))

    next_week = weekly.get("next_week", []) or []
    if next_week:
        rows = [
            (
                item.get("date", ""),
                _truncate(item.get("event", ""), 64, suffix=""),
                _truncate(item.get("read", ""), 88, suffix=""),
            )
            for item in next_week[:8]
        ]
        lines.append("\n**Next-week playbook**")
        lines.append(_make_table(["Date", "Catalyst", "Desk read"], rows))

    return "\n".join(lines)
