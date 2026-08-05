from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


INK = "#111820"
SLATE = "#58656f"
LINE = "#d8dde1"
PANEL_BG = "#ffffff"
FIG_BG = "#f7f6f2"
GREEN = "#1f7a3e"
RED = "#b42318"
AMBER = "#b54708"
BLUE = "#123a56"
DASHBOARD_LAYOUT_VERSION = "morning-dashboard-v10"
CHART_CLIP_MARK = "…"


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
        Rectangle(
            (0, 0),
            1,
            1,
            transform=ax.transAxes,
            linewidth=0.8,
            edgecolor=LINE,
            facecolor=PANEL_BG,
            zorder=-10,
        )
    )


def _panel_title(ax, title: str, subtitle: str = "") -> None:
    ax.add_patch(
        Rectangle(
            (0.018, 0.79),
            0.96,
            0.19,
            transform=ax.transAxes,
            linewidth=0,
            facecolor=PANEL_BG,
            zorder=4,
            clip_on=False,
        )
    )
    ax.text(0.035, 0.95, title, transform=ax.transAxes, fontsize=15.0, fontweight="bold", color=INK, va="top", zorder=7)
    if subtitle:
        ax.text(
            0.035,
            0.82,
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
        "3033.HK ETF",
        "China proxy (FXI)",
        "DXY",
        "WTI crude",
        "Gold",
        "VIX",
    ]
    table = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        table[row.get("label")] = row
        table[row.get("short_label")] = row
    return [table[label] for label in priority_order if label in table and table[label].get("change_pct") is not None][:10]


def _bar_color(value: float) -> str:
    return BLUE if value >= 0 else AMBER


def _regime_impact_color(label: Any, value: float) -> str:
    normalized = str(label or "").strip().lower()
    if abs(value) < 0.05:
        return "#667085"
    if normalized in {"us 10y", "10y treasury"}:
        return AMBER if value > 0 else BLUE
    if normalized in {"dxy", "usd/cnh", "usd/hkd"}:
        return AMBER if value > 0 else BLUE
    if normalized in {"vix"}:
        return AMBER if value > 0 else BLUE
    if normalized in {"wti crude", "brent crude", "crude oil"}:
        return AMBER if value > 0 else BLUE
    if normalized in {"gold"}:
        return AMBER if value > 0 else BLUE
    return BLUE if value > 0 else AMBER


def _dashboard_label(label: Any) -> str:
    mapping = {
        "S&P 500": "S&P 500",
        "Nasdaq 100": "Nasdaq",
        "Hang Seng Index": "HSI",
        "Hang Seng TECH ETF (3033.HK)": "3033 ETF",
        "3033.HK ETF": "3033 ETF",
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
    _panel_title(ax, "Hong Kong confirmation", "Local evidence only; unavailable fields stay out of the signal")

    tile_w = 0.435
    tile_h = 0.16
    x_positions = [0.045, 0.52]
    y_positions = [0.55, 0.33, 0.11]
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
        x = x_positions[idx % 2]
        y = y_positions[idx // 2]
        ax.add_patch(Rectangle((x, y + tile_h - 0.006), tile_w, 0.006, transform=ax.transAxes, linewidth=0, facecolor=LINE))
        ax.text(x, y + tile_h - 0.03, card["label"].upper(), transform=ax.transAxes, fontsize=8.4, color=SLATE, va="center", fontweight="bold")
        ax.text(
            x + tile_w,
            y + tile_h - 0.03,
            chip_text,
            transform=ax.transAxes,
            fontsize=7.8,
            color=chip_color,
            ha="right",
            va="center",
            fontweight="bold",
        )
        ax.text(
            x,
            y + (0.052 if multiline_value else 0.075),
            _safe_text(value_text),
            transform=ax.transAxes,
            fontsize=10.8 if multiline_value else 13.4,
            fontweight="bold",
            color=INK,
            va="center",
            linespacing=1.0,
        )
        if as_of_text:
            ax.text(
                x,
                y + 0.005,
                _safe_text(f"As of {as_of_text}"),
                transform=ax.transAxes,
                fontsize=8.2,
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
    _panel(ax)
    _panel_title(ax, title, subtitle)

    if not rows:
        ax.axis("off")
        ax.text(0.03, 0.55, "No ranked flow or pressure panel was available.", transform=ax.transAxes, fontsize=11, color=SLATE)
        return

    ranked = sorted(rows, key=lambda item: abs(item[1]), reverse=True)[:6]
    ranked.sort(key=lambda item: item[1])
    labels = [textwrap.shorten(item[2] or item[0], width=22, placeholder=CHART_CLIP_MARK) for item in ranked]
    values = [item[1] for item in ranked]
    is_pressure = "%" in xlabel or "ratio" in xlabel.lower()
    colors = [AMBER if is_pressure else BLUE if value >= 0 else AMBER for value in values]
    max_abs = max(max((abs(value) for value in values), default=1.0), 1.0)
    ax.barh(labels, values, color=colors, height=0.55)
    ax.axvline(0, color="#98a2b3", linewidth=1.0)
    ax.set_xlim(min(0.0, min(values or [0])) - max_abs * 0.12, max(0.0, max(values or [0])) + max_abs * 0.20)
    ax.set_ylim(-1.15, len(labels) + 1.60)
    ax.tick_params(axis="x", labelsize=8.8, colors=SLATE)
    ax.tick_params(axis="y", labelsize=10.0, colors=INK, pad=7)
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.8, alpha=0.8)
    ax.set_xlabel(_flow_unit_text(xlabel), fontsize=8.8, color=SLATE, labelpad=4)
    for index, (value, color) in enumerate(zip(values, colors)):
        ax.text(
            value + (max_abs * 0.025 if value >= 0 else -max_abs * 0.025),
            index,
            _safe_text(_flow_value_text(value, xlabel)),
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=9.4,
            color=color,
            fontweight="bold",
            clip_on=False,
        )
    ax.text(
        0.99,
        0.05,
        "Read concentration before the headline aggregate; a narrow flow is not broad confirmation.",
        transform=ax.transAxes,
        fontsize=8.7,
        color=SLATE,
        ha="right",
    )


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
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    leadership = (hk_desk_view.get("lens", "") or hk_desk_view.get("leadership", "") or "Leadership not available").strip()
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
        _safe_text(_wrap_text(leadership, width=56, max_lines=3)),
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
    ax.add_patch(Rectangle((0, 0.94), 1, 0.06, transform=ax.transAxes, linewidth=0, facecolor=LINE))
    ax.text(0, 0.69, label.upper(), transform=ax.transAxes, fontsize=8.4, color=SLATE, va="center", fontweight="bold")
    ax.text(
        0,
        0.29,
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
    fig = plt.figure(figsize=(10.6, 13.2), facecolor=FIG_BG)
    grid = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.05], left=0.14, right=0.95, top=0.74, bottom=0.07, hspace=0.22)

    report_date = bundle.get("meta", {}).get("briefing_date", bundle.get("meta", {}).get("report_date", ""))
    theme = str((bundle.get("overview", {}) or {}).get("theme", "") or "")
    regime = str((bundle.get("overview", {}) or {}).get("risk_regime", "Neutral") or "Neutral")
    risk_dashboard = ((bundle.get("attribution", {}) or {}).get("risk_dashboard", {}) or {})
    risk_score = risk_dashboard.get("score", "N/A")
    risk_bucket = risk_dashboard.get("bucket", regime)
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    leadership = (hk_desk_view.get("headline", "") or hk_desk_view.get("leadership", "") or "Leadership unavailable").strip()
    quality = (bundle.get("meta", {}) or {}).get("market_quality", {}) or {}
    quality_text = _coverage_header_text(bundle)
    mode_label = str(((bundle.get("day_mode", {}) or {}).get("label", "") or "Trading day"))
    rate_row = next(
        (row for row in ((bundle.get("overview", {}) or {}).get("snapshot_rows", []) or []) if row.get("short_label") == "US 10Y"),
        {},
    )
    rate_text = (
        f"{float(rate_row.get('price')):.3f}% | {rate_row.get('change_display', 'N/A')}"
        if rate_row.get("price") is not None
        else "N/A"
    )

    regime_color = GREEN if regime.lower() == "risk-on" else RED if regime.lower() == "risk-off" else AMBER
    bucket_color = GREEN if "on" in str(risk_bucket).lower() else RED if "off" in str(risk_bucket).lower() else AMBER

    fig.text(
        0.045,
        0.965,
        f"Hong Kong Decision Board | {report_date}",
        fontsize=23,
        fontweight="bold",
        ha="left",
        color=INK,
    )
    fig.text(0.045, 0.934, textwrap.shorten(theme, width=82, placeholder=CHART_CLIP_MARK), fontsize=11.2, color=SLATE)
    fig.text(
        0.905,
        0.956,
        regime.upper(),
        fontsize=10.8,
        fontweight="bold",
        color=regime_color,
        ha="center",
        va="center",
        bbox=dict(boxstyle="square,pad=0.35", facecolor=PANEL_BG, edgecolor=LINE),
    )

    _draw_header_card(fig, (0.055, 0.855, 0.405, 0.048), "Risk score", f"{risk_score}/100 | {risk_bucket}", bucket_color)
    _draw_header_card(fig, (0.535, 0.855, 0.405, 0.048), "HK style", leadership, INK)
    _draw_header_card(fig, (0.055, 0.780, 0.405, 0.048), "Data", quality_text, INK)
    _draw_header_card(fig, (0.535, 0.780, 0.405, 0.048), "US 10Y", rate_text, BLUE)

    ax_regime = fig.add_subplot(grid[0, 0])
    _panel(ax_regime)
    rows = _top_snapshot_rows(bundle)[:8]
    if not rows:
        _draw_evidence_coverage(ax_regime, bundle)
    else:
        _panel_title(ax_regime, "Global regime", "Comparable 1D returns; colors reflect HK decision impact")
        raw_labels = [row.get("label", "") for row in rows]
        labels = [_dashboard_label(label) for label in raw_labels]
        values = [float(row.get("change_pct", 0) or 0) for row in rows]
        colors = [_regime_impact_color(label, value) for label, value in zip(raw_labels, values)]
        max_abs = max(max((abs(value) for value in values), default=1.0), 1.0)
        ax_regime.barh(labels, values, color=colors, height=0.56)
        ax_regime.axvline(0, color="#98a2b3", linewidth=1.0)
        ax_regime.set_xlim(min(0.0, min(values or [0])) - max_abs * 0.27, max(0.0, max(values or [0])) + max_abs * 0.28)
        ax_regime.invert_yaxis()
        ax_regime.set_ylim(len(labels) - 0.35, -2.65)
        ax_regime.tick_params(axis="x", labelsize=9.7)
        ax_regime.tick_params(axis="y", labelsize=10.5, pad=7)
        ax_regime.grid(axis="x", color="#e4e7ec", linewidth=0.8, alpha=0.75)
        for idx, value in enumerate(values):
            ax_regime.text(
                value + (max_abs * 0.025 if value >= 0 else -max_abs * 0.025),
                idx,
                _safe_text(f"{value:+.2f}%"),
                va="center",
                ha="left" if value >= 0 else "right",
                fontsize=9.8,
                color=INK,
                clip_on=False,
            )

    ax_hk = fig.add_subplot(grid[1, 0])
    _draw_metric_cards(ax_hk, _hk_metric_cards(bundle))

    ax_flow = fig.add_subplot(grid[2, 0])
    _draw_flow_focus(ax_flow, bundle)

    fig.text(
        0.045,
        0.025,
        "Decision sequence: global regime -> Hong Kong confirmation -> concentration. Event timing is handled separately in the radar.",
        fontsize=9.8,
        color=SLATE,
    )
    fig.savefig(output_path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    return os.path.basename(output_path)
