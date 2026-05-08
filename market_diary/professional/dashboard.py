from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


INK = "#0f172a"
SLATE = "#475467"
LINE = "#d0d5dd"
PANEL_BG = "#ffffff"
FIG_BG = "#eef2f6"
GREEN = "#1f7a3e"
RED = "#b42318"
AMBER = "#b54708"
BLUE = "#0b4f71"
DASHBOARD_LAYOUT_VERSION = "morning-dashboard-v6"
CHART_CLIP_MARK = "~"


def _parse_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace(",", "").replace("x", "").replace("HK$", "").replace("bn", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _safe_text(value: Any) -> str:
    return str(value or "").replace("$", r"\$")


def _wrap_text(value: Any, width: int, max_lines: int = 2) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if not lines:
        return ""
    clipped = lines[:max_lines]
    if len(lines) > max_lines and clipped:
        clipped[-1] = textwrap.shorten(clipped[-1], width=max(10, width - 2), placeholder=CHART_CLIP_MARK)
    return "\n".join(clipped)


def _wrap_segments(value: Any, primary_width: int = 18, secondary_width: int = 24, max_lines: int = 2) -> str:
    text = str(value or "").strip()
    if not text:
        return "N/A"
    parts = [part.strip() for part in text.split("|") if part.strip()]
    if not parts:
        return _wrap_text(text, width=secondary_width, max_lines=max_lines)
    lines: List[str] = []
    for index, part in enumerate(parts):
        width = primary_width if index == 0 else secondary_width
        wrapped = textwrap.wrap(part, width=width, break_long_words=False, break_on_hyphens=False) or [part]
        lines.extend(wrapped)
        if len(lines) >= max_lines:
            break
    lines = lines[:max_lines]
    if len(parts) > max_lines or len(lines) == max_lines and len(textwrap.wrap(text, width=secondary_width, break_long_words=False, break_on_hyphens=False)) > max_lines:
        lines[-1] = textwrap.shorten(lines[-1], width=max(12, secondary_width - 2), placeholder=CHART_CLIP_MARK)
    return "\n".join(lines)


def _status_chip(status: str) -> Tuple[str, str]:
    normalized = str(status or "").lower()
    if normalized.startswith("live"):
        return "LIVE", GREEN
    if normalized.startswith("stale"):
        return "STALE", AMBER
    if normalized == "proxy":
        return "PROXY", BLUE
    if normalized == "fallback":
        return "FALLBACK", AMBER
    if normalized == "coverage":
        return "GATED", AMBER
    return "UNAVAIL", RED


def _panel(ax) -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            transform=ax.transAxes,
            linewidth=1.0,
            edgecolor=LINE,
            facecolor=PANEL_BG,
            zorder=-10,
        )
    )


def _panel_title(ax, title: str, subtitle: str = "") -> None:
    ax.add_patch(
        Rectangle(
            (0.018, 0.835),
            0.96,
            0.145,
            transform=ax.transAxes,
            linewidth=0,
            facecolor=PANEL_BG,
            zorder=4,
            clip_on=False,
        )
    )
    ax.text(0.035, 0.95, title, transform=ax.transAxes, fontsize=15.8, fontweight="bold", color=INK, va="top", zorder=7)
    if subtitle:
        ax.text(
            0.035,
            0.895,
            _wrap_text(subtitle, width=58, max_lines=1),
            transform=ax.transAxes,
            fontsize=10.6,
            color=SLATE,
            va="top",
            zorder=7,
        )


def _top_snapshot_rows(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = (bundle.get("overview", {}) or {}).get("snapshot_rows", []) or []
    priority_order = [
        "S&P 500",
        "Nasdaq 100",
        "Hang Seng Index",
        "Hang Seng TECH",
        "China proxy (FXI)",
        "US 10Y",
        "DXY",
        "WTI crude",
        "Gold",
        "VIX",
    ]
    table = {row.get("label"): row for row in rows if isinstance(row, dict)}
    return [table[label] for label in priority_order if label in table and table[label].get("change_pct") is not None][:10]


def _bar_color(value: float) -> str:
    return GREEN if value >= 0 else RED


def _dashboard_label(label: Any) -> str:
    mapping = {
        "S&P 500": "S&P 500",
        "Nasdaq 100": "Nasdaq",
        "Hang Seng Index": "HSI",
        "Hang Seng TECH": "HSTECH",
        "China proxy (FXI)": "FXI",
        "US 10Y": "US 10Y",
        "DXY": "DXY",
        "WTI crude": "WTI",
        "Gold": "Gold",
        "VIX": "VIX",
    }
    return mapping.get(str(label or ""), textwrap.shorten(str(label or "N/A"), width=12, placeholder=CHART_CLIP_MARK))


def _name_label(item: Dict[str, Any], max_width: int = 18) -> str:
    ticker = str(item.get("ticker", "") or item.get("code", "") or "").strip()
    if ticker and ticker.isdigit():
        ticker = f"{ticker}.HK"
    name = str(item.get("name", "") or "").strip()
    if ticker:
        return ticker
    return textwrap.shorten(name or "N/A", width=max_width, placeholder=CHART_CLIP_MARK)


def _name_note(item: Dict[str, Any], max_width: int = 20) -> str:
    name = str(item.get("name", "") or item.get("ticker", "") or item.get("code", "") or "").strip()
    return textwrap.shorten(name or "N/A", width=max_width, placeholder=CHART_CLIP_MARK)


def _hk_metric_cards(bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    hk_local = bundle.get("hk_local", {}) or {}
    ordered = [
        ("Turnover vs 20D", hk_local.get("turnover_vs_20d", {}) or hk_local.get("main_board_turnover", {})),
        ("Southbound flow", hk_local.get("southbound_net_flow", {})),
        ("Short-selling ratio", hk_local.get("short_selling_ratio", {})),
        ("HIBOR 1M", hk_local.get("hibor_1m", {})),
        ("Aggregate Balance", hk_local.get("aggregate_balance", {})),
        ("A/H premium", hk_local.get("ah_premium_index", {})),
    ]
    cards: List[Dict[str, str]] = []
    hidden_count = 0
    for label, metric in ordered:
        status = str(metric.get("status", "unavailable") or "unavailable")
        if status == "unavailable":
            hidden_count += 1
            continue
        cards.append(
            {
                "label": label,
                "value": str(metric.get("display_value", "N/A") or "N/A"),
                "status": status,
                "note": str(metric.get("note", "") or ""),
                "as_of": str(metric.get("as_of", "") or ""),
            }
        )
    if hidden_count:
        cards.append(
            {
                "label": "Suppressed checks",
                "value": f"{hidden_count} not refreshed",
                "status": "coverage",
                "note": "Unavailable local fields are kept out of the visual read.",
                "as_of": "",
            }
        )
    return cards


def _draw_metric_cards(ax, cards: List[Dict[str, str]]) -> None:
    ax.axis("off")
    _panel(ax)
    _panel_title(ax, "Hong Kong local tape", "Refreshed local evidence; missing fields are gated")

    tile_w = 0.435
    tile_h = 0.168
    x_positions = [0.045, 0.52]
    y_positions = [0.64, 0.405, 0.17]
    for idx, card in enumerate(cards[:6]):
        chip_text, chip_color = _status_chip(card["status"])
        raw_value = str(card["value"] or "N/A").replace(" / ", " | ")
        value_text = _wrap_segments(raw_value, primary_width=18, secondary_width=20, max_lines=2)
        multiline_value = "\n" in value_text
        as_of_text = (
            ""
            if multiline_value
            else textwrap.shorten(str(card.get("as_of") or ""), width=18, placeholder=CHART_CLIP_MARK) if card.get("as_of") else ""
        )
        row_face = "#f8fafc" if idx % 2 == 0 else "#fdfefe"
        x = x_positions[idx % 2]
        y = y_positions[idx // 2]
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                tile_w,
                tile_h,
                boxstyle="round,pad=0.014,rounding_size=0.03",
                transform=ax.transAxes,
                linewidth=0.95,
                edgecolor="#d9e2ec",
                facecolor=row_face,
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (x + tile_w - 0.125, y + tile_h - 0.05),
                0.095,
                0.031,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                transform=ax.transAxes,
                linewidth=0.8,
                edgecolor=chip_color,
                facecolor="#ffffff",
            )
        )
        ax.text(x + 0.026, y + tile_h - 0.037, card["label"], transform=ax.transAxes, fontsize=10.3, color=SLATE, va="center", fontweight="bold")
        ax.text(
            x + tile_w - 0.077,
            y + tile_h - 0.034,
            chip_text,
            transform=ax.transAxes,
            fontsize=7.6,
            color=chip_color,
            ha="center",
            va="center",
            fontweight="bold",
        )
        ax.text(
            x + 0.026,
            y + (0.058 if multiline_value else 0.077),
            _safe_text(value_text),
            transform=ax.transAxes,
            fontsize=10.6 if multiline_value else 13.2,
            fontweight="bold",
            color=INK,
            va="center",
            linespacing=1.0,
        )
        if as_of_text:
            ax.text(
                x + 0.026,
                y + 0.025,
                _safe_text(f"As of {as_of_text}"),
                transform=ax.transAxes,
                fontsize=8.7,
                color=SLATE,
                va="bottom",
            )


def _available_data_labels(bundle: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    flow_tracker = bundle.get("flow_tracker", {}) or {}
    stock_connect = flow_tracker.get("stock_connect", {}) or {}
    stock_connect_data = stock_connect.get("data", {}) or {}
    if stock_connect.get("status") in {"ok", "partial"} or ((stock_connect_data.get("southbound", {}) or {}).get("top_active")):
        labels.append("Stock Connect")
    ah_premium = flow_tracker.get("ah_premium", {}) or {}
    if ah_premium.get("status") in {"ok", "partial"} or ((ah_premium.get("data", {}) or {}).get("top_premium")):
        labels.append("A/H premium")
    china_rates = bundle.get("china_rates", {}) or {}
    if any((item or {}).get("status") not in {None, "", "unavailable"} for item in china_rates.values() if isinstance(item, dict)):
        labels.append("China rates")
    company_events = bundle.get("company_events", {}) or {}
    if (company_events.get("hkex_meta", {}) or {}).get("status") in {"ok", "partial"} or company_events.get("announcements"):
        labels.append("HKEX news")
    return labels


def _coverage_header_text(bundle: Dict[str, Any]) -> str:
    quality = (bundle.get("meta", {}) or {}).get("market_quality", {}) or {}
    available = quality.get("available")
    total = quality.get("total")
    if isinstance(available, int) and isinstance(total, int) and total > 0:
        return f"{available}/{total} fields"
    labels = _available_data_labels(bundle)
    if labels:
        return ", ".join(labels[:3])
    return "Limited coverage"


def _draw_evidence_coverage(ax, bundle: Dict[str, Any]) -> None:
    ax.set_axis_off()
    _panel_title(ax, "Evidence coverage", "Use refreshed evidence first; keep missing quote fields out of the read")
    labels = _available_data_labels(bundle)
    if not labels:
        labels = ["No major public adapter refreshed"]

    y = 0.66
    for label in labels[:5]:
        ax.add_patch(
            FancyBboxPatch(
                (0.055, y),
                0.88,
                0.105,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=ax.transAxes,
                linewidth=0.85,
                edgecolor="#d9e2ec",
                facecolor="#f8fafc",
            )
        )
        ax.add_patch(Rectangle((0.055, y), 0.012, 0.105, transform=ax.transAxes, linewidth=0, facecolor=GREEN))
        ax.text(0.09, y + 0.066, label, transform=ax.transAxes, fontsize=11.4, color=INK, fontweight="bold", va="center")
        ax.text(0.09, y + 0.028, "Available for the main read", transform=ax.transAxes, fontsize=8.9, color=SLATE, va="center")
        y -= 0.13

    ax.text(
        0.055,
        0.10,
        _safe_text(_wrap_text("Read order today: local flow and A/H dispersion carry more weight than unavailable broad index quotes.", width=76, max_lines=2)),
        transform=ax.transAxes,
        fontsize=9.4,
        color=AMBER,
        fontweight="bold",
        linespacing=1.25,
    )


def _flow_focus(bundle: Dict[str, Any]) -> Tuple[str, str, List[Tuple[str, float, str]], str]:
    flow_tracker = bundle.get("flow_tracker", {}) or {}
    stock_connect = ((flow_tracker.get("stock_connect", {}) or {}).get("data", {}) or {})
    southbound_active = ((stock_connect.get("southbound", {}) or {}).get("top_active", []) or [])[:6]
    if southbound_active:
        rows = []
        for item in southbound_active:
            net = _parse_float(item.get("net_buy")) or 0.0
            label = _name_label(item)
            rows.append((label, net, _name_note(item, max_width=18)))
        return (
            "Flow concentration",
            "Top Southbound active names by net buy / sell",
            rows,
            "Net buy / sell (HKD mn)",
        )

    short_rows = (flow_tracker.get("short_sell_watchlist_hits", []) or flow_tracker.get("short_sell_top_ratio", []) or [])[:6]
    if short_rows:
        rows = []
        for item in short_rows:
            ratio = _parse_float(item.get("short_ratio_pct")) or 0.0
            label = _name_label(item)
            rows.append((label, ratio, _name_note(item, max_width=18)))
        return (
            "Pressure concentration",
            "Most stressed HKEX short-selling names",
            rows,
            "Short-selling ratio (%)",
        )

    drivers = ((bundle.get("attribution", {}) or {}).get("dominant_drivers", []) or [])[:6]
    rows = []
    for item in drivers:
        score = _parse_float(item.get("score")) or 0.0
        rows.append((str(item.get("name", "")), score, str(item.get("direction", ""))))
    return (
        "Cross-asset drivers",
        "Fallback ranking from deterministic attribution",
        rows,
        "Attribution score",
    )


def _flow_value_text(value: float, unit_label: str) -> str:
    lowered = str(unit_label or "").lower()
    if "hkd" in lowered or "mn" in lowered:
        return f"{value:+,.0f}mn"
    if "%" in lowered or "ratio" in lowered:
        return f"{value:.1f}%"
    return f"{value:+.1f}"


def _flow_unit_text(unit_label: str) -> str:
    lowered = str(unit_label or "").lower()
    if "hkd" in lowered or "mn" in lowered:
        return "HKD mn"
    if "%" in lowered or "ratio" in lowered:
        return "%"
    return "score"


def _draw_flow_focus(ax, bundle: Dict[str, Any]) -> None:
    title, subtitle, rows, xlabel = _flow_focus(bundle)
    ax.axis("off")
    _panel(ax)
    _panel_title(ax, title, subtitle)

    if not rows:
        ax.text(0.03, 0.55, "No ranked flow or pressure panel was available.", transform=ax.transAxes, fontsize=11, color=SLATE)
        return

    positives = sorted([row for row in rows if row[1] >= 0], key=lambda item: item[1], reverse=True)[:3]
    negatives = sorted([row for row in rows if row[1] < 0], key=lambda item: item[1])[:3]

    def draw_column(x: float, heading: str, items: List[Tuple[str, float, str]], color: str) -> None:
        ax.text(x, 0.74, heading.upper(), transform=ax.transAxes, fontsize=9.2, color=color, fontweight="bold")
        ax.text(x + 0.405, 0.74, _flow_unit_text(xlabel), transform=ax.transAxes, fontsize=8.6, color=SLATE, ha="right")
        if not items:
            ax.add_patch(
                FancyBboxPatch(
                    (x, 0.43),
                    0.43,
                    0.15,
                    boxstyle="round,pad=0.012,rounding_size=0.02",
                    transform=ax.transAxes,
                    linewidth=0.8,
                    edgecolor="#d9e2ec",
                    facecolor="#f8fafc",
                )
            )
            ax.text(x + 0.03, 0.505, "No ranked names", transform=ax.transAxes, fontsize=10.2, color=SLATE, va="center")
            return

        y = 0.60
        for rank, (label, value, note) in enumerate(items, start=1):
            ax.add_patch(
                FancyBboxPatch(
                    (x, y),
                    0.43,
                    0.115,
                    boxstyle="round,pad=0.012,rounding_size=0.022",
                    transform=ax.transAxes,
                    linewidth=0.85,
                    edgecolor="#d9e2ec",
                    facecolor="#f8fafc",
                )
            )
            ax.add_patch(
                Rectangle((x, y), 0.011, 0.115, transform=ax.transAxes, linewidth=0, facecolor=color)
            )
            ax.text(
                x + 0.028,
                y + 0.076,
                _safe_text(f"{rank}. {textwrap.shorten(label, width=14, placeholder=CHART_CLIP_MARK)}"),
                transform=ax.transAxes,
                fontsize=11.1,
                color=INK,
                va="center",
                fontweight="bold",
            )
            ax.text(
                x + 0.028,
                y + 0.031,
                _safe_text(textwrap.shorten(note or "Name unavailable", width=20, placeholder=CHART_CLIP_MARK)),
                transform=ax.transAxes,
                fontsize=8.9,
                color=SLATE,
                va="center",
            )
            ax.text(
                x + 0.405,
                y + 0.058,
                _safe_text(_flow_value_text(value, xlabel)),
                transform=ax.transAxes,
                fontsize=11.0,
                color=color,
                va="center",
                ha="right",
                fontweight="bold",
            )
            y -= 0.145

    if negatives:
        draw_column(0.045, "Top net buy", positives, GREEN)
        draw_column(0.525, "Top net sell", negatives, RED)
        ax.text(
            0.045,
            0.075,
            _safe_text(_wrap_text("Desk read: concentration matters more than headline flow; check whether buys are index-heavy or idiosyncratic.", width=88, max_lines=2)),
            transform=ax.transAxes,
            fontsize=9.2,
            color=SLATE,
            linespacing=1.25,
        )
        return

    ranked = sorted(rows, key=lambda item: abs(item[1]), reverse=True)[:4]
    ax.text(0.045, 0.74, "TOP RANKED NAMES", transform=ax.transAxes, fontsize=9.2, color=BLUE, fontweight="bold")
    max_abs = max(max((abs(item[1]) for item in ranked), default=1.0), 1.0)
    y = 0.61
    for rank, (label, value, note) in enumerate(ranked, start=1):
        bar_w = 0.30 * abs(value) / max_abs
        color = _bar_color(value)
        ax.add_patch(
            FancyBboxPatch(
                (0.045, y),
                0.91,
                0.105,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=ax.transAxes,
                linewidth=0.85,
                edgecolor="#d9e2ec",
                facecolor="#f8fafc",
            )
        )
        ax.text(0.07, y + 0.065, f"{rank}. {_safe_text(textwrap.shorten(label, width=18, placeholder=CHART_CLIP_MARK))}", transform=ax.transAxes, fontsize=11, color=INK, fontweight="bold")
        ax.text(0.07, y + 0.026, _safe_text(textwrap.shorten(note or "", width=26, placeholder=CHART_CLIP_MARK)), transform=ax.transAxes, fontsize=8.8, color=SLATE)
        ax.add_patch(Rectangle((0.60, y + 0.035), bar_w, 0.032, transform=ax.transAxes, linewidth=0, facecolor=color))
        ax.text(0.935, y + 0.052, _safe_text(_flow_value_text(value, xlabel)), transform=ax.transAxes, fontsize=10.7, color=color, fontweight="bold", ha="right", va="center")
        y -= 0.13


def _catalyst_lines(bundle: Dict[str, Any]) -> List[Tuple[str, str, str]]:
    items: List[Tuple[str, str, str]] = []
    for item in (bundle.get("catalysts", []) or [])[:5]:
        time_label = f"{item.get('date', '')} {item.get('time', '')}".strip()
        tag = str(item.get("category", "Catalyst") or "Catalyst")
        headline = str(item.get("event", "") or "")
        items.append((time_label, tag, headline))
    for item in (bundle.get("macro_agenda", []) or [])[:3]:
        time_label = f"{item.get('time', '')} {item.get('country', '')}".strip()
        tag = str(item.get("status", "Macro") or "Macro")
        headline = str(item.get("event", "") or "")
        row = (time_label, tag, headline)
        if row not in items:
            items.append(row)
    return items[:7]


def _draw_catalysts(ax, bundle: Dict[str, Any]) -> None:
    ax.axis("off")
    _panel(ax)
    _panel_title(ax, "Catalyst ladder", "Today's highest-conviction watchlist")
    lines = _catalyst_lines(bundle)
    if not lines:
        ax.text(0.03, 0.55, "No catalyst items were available.", transform=ax.transAxes, fontsize=11, color=SLATE)
        return
    row_height = 0.118
    row_gap = 0.018
    y = 0.66
    for idx, (time_label, tag, headline) in enumerate(lines[:5]):
        tag_color = BLUE if tag.lower() in {"upcoming", "released", "macro", "central bank"} else AMBER
        headline_text = textwrap.shorten(headline, width=58, placeholder=CHART_CLIP_MARK)
        row_face = "#f8fafc" if idx % 2 == 0 else "#fdfefe"
        ax.add_patch(
            FancyBboxPatch(
                (0.035, y),
                0.93,
                row_height,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=ax.transAxes,
                linewidth=0.8,
                edgecolor="#d9e2ec",
                facecolor=row_face,
            )
        )
        ax.text(0.055, y + 0.082, time_label or "Today", transform=ax.transAxes, fontsize=9.5, color=SLATE, va="center")
        ax.text(0.32, y + 0.082, f"[{tag}]", transform=ax.transAxes, fontsize=9.0, color=tag_color, va="center", fontweight="bold")
        ax.text(
            0.055,
            y + 0.034,
            _safe_text(headline_text),
            transform=ax.transAxes,
            fontsize=11.0,
            color=INK,
            va="center",
        )
        y -= row_height + row_gap


def _watchlist_rows(bundle: Dict[str, Any]) -> List[Tuple[str, str, str, str]]:
    rows: List[Tuple[str, str, str, str]] = []
    for bucket, items in (bundle.get("watchlists", {}) or {}).items():
        if not items:
            continue
        top = items[0]
        change = top.get("daily_change_pct")
        if isinstance(change, (int, float)):
            change_text = f"{change:+.2f}%"
        else:
            change_text = "N/A"
        rows.append(
            (
                bucket.replace("_", " ").title(),
                str(top.get("name", "") or ""),
                change_text,
                str(top.get("range_label", "") or "N/A"),
            )
        )
    return rows[:4]


def _draw_watchlist(ax, bundle: Dict[str, Any]) -> None:
    ax.axis("off")
    _panel(ax)
    _panel_title(ax, "Coverage pulse", "One line per bucket so the watchlist remains decision-useful")
    rows = _watchlist_rows(bundle)
    if not rows:
        ax.text(0.03, 0.55, "No watchlist items were available.", transform=ax.transAxes, fontsize=11, color=SLATE)
        return
    ax.text(0.03, 0.80, "Bucket", transform=ax.transAxes, fontsize=9.2, color=SLATE, fontweight="bold")
    ax.text(0.28, 0.80, "Name", transform=ax.transAxes, fontsize=9.2, color=SLATE, fontweight="bold")
    ax.text(0.63, 0.80, "1D", transform=ax.transAxes, fontsize=9.2, color=SLATE, fontweight="bold")
    ax.text(0.77, 0.80, "Range", transform=ax.transAxes, fontsize=9.2, color=SLATE, fontweight="bold")
    y = 0.68
    for bucket, name, change_text, range_text in rows:
        ax.text(0.03, y, bucket, transform=ax.transAxes, fontsize=10.2, color=SLATE)
        ax.text(0.28, y, textwrap.shorten(name, width=24, placeholder=CHART_CLIP_MARK), transform=ax.transAxes, fontsize=11, color=INK)
        ax.text(0.63, y, change_text, transform=ax.transAxes, fontsize=10.5, color=GREEN if change_text.startswith("+") else RED if change_text.startswith("-") else SLATE, fontweight="bold")
        ax.text(0.77, y, _wrap_text(range_text, width=14, max_lines=1), transform=ax.transAxes, fontsize=10.0, color=INK)
        y -= 0.14


def _draw_desk_frame(ax, bundle: Dict[str, Any]) -> None:
    ax.axis("off")
    _panel(ax)
    _panel_title(ax, "Desk framing", "What to say if someone asks for the one-minute market read")

    attribution = bundle.get("attribution", {}) or {}
    risk_dashboard = attribution.get("risk_dashboard", {}) or {}
    score = risk_dashboard.get("score", "N/A")
    bucket = risk_dashboard.get("bucket", "Mixed")
    leadership = ((bundle.get("hk_desk_view", {}) or {}).get("leadership", "") or "Leadership not available").strip()
    quality = (bundle.get("meta", {}) or {}).get("market_quality", {}) or {}
    must_watch = bundle.get("must_watch", []) or []

    bucket_color = GREEN if "on" in str(bucket).lower() else RED if "off" in str(bucket).lower() else AMBER
    quality_text = f"{quality.get('available', 'N/A')}/{quality.get('total', 'N/A')} market fields"

    for x in (0.04, 0.53):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.58),
                0.39,
                0.18,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=ax.transAxes,
                linewidth=0.9,
                edgecolor="#d9e2ec",
                facecolor="#f8fafc",
            )
        )

    ax.text(0.07, 0.72, "Risk score", transform=ax.transAxes, fontsize=9.8, color=SLATE)
    ax.text(0.07, 0.62, f"{score}/100", transform=ax.transAxes, fontsize=23, fontweight="bold", color=INK)
    ax.text(0.28, 0.64, bucket, transform=ax.transAxes, fontsize=11.5, color=bucket_color, fontweight="bold")

    ax.text(0.56, 0.72, "Data coverage", transform=ax.transAxes, fontsize=9.8, color=SLATE)
    ax.text(0.56, 0.63, quality_text, transform=ax.transAxes, fontsize=11.4, color=INK, fontweight="bold")
    ax.text(
        0.56,
        0.56,
        f"Fallbacks {len(quality.get('fallback', []) or [])} | Missing {len(quality.get('missing', []) or [])}",
        transform=ax.transAxes,
        fontsize=9.0,
        color=SLATE,
    )

    ax.text(0.04, 0.45, "Hong Kong style read", transform=ax.transAxes, fontsize=9.8, color=SLATE)
    ax.text(
        0.04,
        0.365,
        _safe_text(_wrap_text(leadership, width=44, max_lines=2)),
        transform=ax.transAxes,
        fontsize=11.6,
        color=INK,
        va="top",
        linespacing=1.15,
    )

    ax.text(0.04, 0.22, "First questions", transform=ax.transAxes, fontsize=9.8, color=SLATE)
    y = 0.15
    for item in must_watch[:3]:
        line = _wrap_text(f"[{item.get('bucket', '')}] {item.get('title', '')}", width=52, max_lines=1)
        ax.text(0.04, y, f"- {_safe_text(line)}", transform=ax.transAxes, fontsize=9.5, color=INK)
        y -= 0.08


def _header_chip(fig, x: float, y: float, label: str, value: str, color: str = INK) -> None:
    fig.text(
        x,
        y,
        f"{label}\n{value}",
        fontsize=9.6,
        color=color,
        linespacing=1.35,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.48,rounding_size=0.08", facecolor=PANEL_BG, edgecolor=LINE),
    )


def _draw_header_card(fig, rect: Tuple[float, float, float, float], label: str, value: str, color: str = INK) -> None:
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.018,rounding_size=0.04",
            transform=ax.transAxes,
            linewidth=1.0,
            edgecolor=LINE,
            facecolor=PANEL_BG,
        )
    )
    ax.text(0.055, 0.70, label, transform=ax.transAxes, fontsize=9.6, color=SLATE, va="center", fontweight="bold")
    ax.text(
        0.055,
        0.34,
        _safe_text(_wrap_text(value, width=26, max_lines=2)),
        transform=ax.transAxes,
        fontsize=11.8,
        color=color,
        va="center",
        fontweight="bold",
        linespacing=1.1,
    )


def generate_dashboard(bundle: Dict[str, Any], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.style.use("default")
    fig = plt.figure(figsize=(14.2, 10.8), facecolor=FIG_BG)
    grid = fig.add_gridspec(2, 2, left=0.09, right=0.965, top=0.785, bottom=0.092, hspace=0.30, wspace=0.14)

    report_date = bundle.get("meta", {}).get("report_date", "")
    theme = str((bundle.get("overview", {}) or {}).get("theme", "") or "")
    regime = str((bundle.get("overview", {}) or {}).get("risk_regime", "Neutral") or "Neutral")
    risk_dashboard = ((bundle.get("attribution", {}) or {}).get("risk_dashboard", {}) or {})
    risk_score = risk_dashboard.get("score", "N/A")
    risk_bucket = risk_dashboard.get("bucket", regime)
    leadership = ((bundle.get("hk_desk_view", {}) or {}).get("leadership", "") or "Leadership unavailable").strip()
    quality = (bundle.get("meta", {}) or {}).get("market_quality", {}) or {}
    quality_text = _coverage_header_text(bundle)
    mode_label = str(((bundle.get("day_mode", {}) or {}).get("label", "") or "Trading day"))

    regime_color = GREEN if regime.lower() == "risk-on" else RED if regime.lower() == "risk-off" else AMBER
    bucket_color = GREEN if "on" in str(risk_bucket).lower() else RED if "off" in str(risk_bucket).lower() else AMBER

    fig.text(
        0.045,
        0.955,
        f"Hong Kong Morning Dashboard | {report_date}",
        fontsize=24,
        fontweight="bold",
        ha="left",
        color=INK,
    )
    fig.text(0.045, 0.918, textwrap.shorten(theme, width=112, placeholder=CHART_CLIP_MARK), fontsize=12.2, color=SLATE)
    fig.text(
        0.905,
        0.942,
        regime.upper(),
        fontsize=10.8,
        fontweight="bold",
        color=regime_color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG, edgecolor=LINE),
    )

    _draw_header_card(fig, (0.045, 0.825, 0.215, 0.068), "Risk score", f"{risk_score}/100 | {risk_bucket}", bucket_color)
    _draw_header_card(fig, (0.285, 0.825, 0.275, 0.068), "HK style", leadership, INK)
    _draw_header_card(fig, (0.585, 0.825, 0.17, 0.068), "Data", quality_text, INK)
    _draw_header_card(fig, (0.78, 0.825, 0.16, 0.068), "Mode", mode_label, BLUE)

    ax_regime = fig.add_subplot(grid[0, 0])
    _panel(ax_regime)
    rows = _top_snapshot_rows(bundle)[:8]
    if not rows:
        _draw_evidence_coverage(ax_regime, bundle)
    else:
        _panel_title(ax_regime, "Global regime board", "The cross-asset tape that frames the Hong Kong open")
        labels = [_dashboard_label(row.get("label", "")) for row in rows]
        values = [float(row.get("change_pct", 0) or 0) for row in rows]
        colors = [_bar_color(value) for value in values]
        max_abs = max(max((abs(value) for value in values), default=1.0), 1.0)
        ax_regime.barh(labels, values, color=colors, height=0.56)
        ax_regime.axvline(0, color="#98a2b3", linewidth=1.0)
        ax_regime.set_xlim(min(0.0, min(values or [0])) - max_abs * 0.18, max(0.0, max(values or [0])) + max_abs * 0.24)
        ax_regime.invert_yaxis()
        ax_regime.set_ylim(len(labels) - 0.35, -1.75)
        ax_regime.tick_params(axis="x", labelsize=9.7)
        ax_regime.tick_params(axis="y", labelsize=11.0)
        ax_regime.grid(axis="x", color="#e4e7ec", linewidth=0.8, alpha=0.75)
        for idx, value in enumerate(values):
            if abs(value) < 0.35:
                continue
            ax_regime.text(
                value + (max_abs * 0.025 if value >= 0 else -max_abs * 0.025),
                idx,
                _safe_text(f"{value:+.2f}%"),
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9.8,
                color=INK,
                clip_on=True,
            )

    ax_hk = fig.add_subplot(grid[0, 1])
    _draw_metric_cards(ax_hk, _hk_metric_cards(bundle))

    ax_flow = fig.add_subplot(grid[1, 0])
    _draw_flow_focus(ax_flow, bundle)

    ax_catalyst = fig.add_subplot(grid[1, 1])
    _draw_catalysts(ax_catalyst, bundle)

    fig.text(
        0.045,
        0.025,
        "Read order: regime -> Hong Kong local tape -> flow concentration -> catalyst ladder. Detailed watchlist and desk framing stay in the markdown report.",
        fontsize=9.8,
        color=SLATE,
    )
    fig.savefig(output_path, dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.basename(output_path)
