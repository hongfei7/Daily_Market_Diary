from __future__ import annotations

from typing import Any, Dict, List, Sequence

from professional.report_formatting import (
    _bundle_metric,
    _fmt_alert_pct,
    _fmt_pct,
    _fmt_price,
    _make_table,
    _source_as_of,
    _status_label,
    _summary_pct,
    _summary_price,
    _truncate,
)


def _render_top_items(items: List[Dict[str, Any]], limit: int = 6) -> str:
    if not items:
        return "1. No priority items were available."
    lines = []
    for idx, item in enumerate(items[:limit], 1):
        lines.append(f"{idx}. **{item.get('title', '')}**")
        lines.append(f"   {item.get('bucket', '')} | {_truncate(item.get('summary', ''), 110)}")
    return "\n".join(lines)


def _render_selected_news(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    items = llm_sections.get("selected_news", []) or []
    if not items:
        return ""
    lines = ["**Curated overnight stories:**"]
    for item in items[:5]:
        lines.append(
            f"- **{item.get('headline', '')}** | {item.get('why_it_matters', '')} | HK read-through: {item.get('hk_market_impact', '')}"
        )
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
            read = f"{note} | {_status_label(china_10y_metric.get('status', 'unavailable'))} | {_source_as_of(china_10y_metric)}"
        elif name == "CN-US 10Y spread":
            price = spread_metric.get("display_value", "N/A")
            pct = spread_metric.get("change_display", "")
            read = f"{note} | {_status_label(spread_metric.get('status', 'unavailable'))} | {_source_as_of(spread_metric)}"
        else:
            price = _summary_price(bundle, category, name)
            pct = _summary_pct(bundle, category, name)
            read = note

        last_value = price if isinstance(price, str) else _fmt_price(price)
        move_value = pct if isinstance(pct, str) and pct else _fmt_alert_pct(pct)
        table_rows.append((label, last_value, move_value, read))

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
            _source_as_of(item),
            _truncate(item.get("note", ""), 88),
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
        f"- **Composite risk score:** {risk.get('score', 'N/A')}/100 | **Regime:** {risk.get('bucket', 'Mixed')}",
    ]
    if components:
        rows = [
            (
                item.get("label", ""),
                f"{item.get('delta', 0):+}",
                _truncate(item.get("evidence", ""), 80),
            )
            for item in components[:6]
        ]
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
                _truncate(item.get("interpretation", ""), 78),
            )
            for item in still_moving[:6]
        ]
        lines.append(_make_table(["Monitor", "Bucket", "Last", "Move", "Why it matters"], rows))

    action_items = focus.get("action_items", []) or []
    if action_items:
        lines.append("\n**Still-moving actions to track**")
        for item in action_items[:5]:
            lines.append(f"- **{item.get('bucket', '')}:** {_truncate(item.get('item', ''), 72)} | {_truncate(item.get('read', ''), 96)}")

    next_open = focus.get("next_open", []) or []
    if next_open:
        lines.append("\n**Next-open preparation**")
        lines.extend(f"- {line}" for line in next_open[:4])

    return "\n".join(lines)
