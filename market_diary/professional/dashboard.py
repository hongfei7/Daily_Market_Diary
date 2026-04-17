from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


INK = "#0f172a"
SLATE = "#475467"
LINE = "#d0d5dd"
PANEL_BG = "#ffffff"
FIG_BG = "#eef2f6"
GREEN = "#1f7a3e"
RED = "#b42318"
AMBER = "#b54708"
BLUE = "#0b4f71"


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
        clipped[-1] = textwrap.shorten(clipped[-1], width=max(10, width - 2), placeholder="...")
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
        lines[-1] = textwrap.shorten(lines[-1], width=max(12, secondary_width - 2), placeholder="...")
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
    ax.text(0.03, 0.94, title, transform=ax.transAxes, fontsize=13.5, fontweight="bold", color=INK, va="top")
    if subtitle:
        ax.text(0.03, 0.885, subtitle, transform=ax.transAxes, fontsize=9.5, color=SLATE, va="top")


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
    for label, metric in ordered:
        cards.append(
            {
                "label": label,
                "value": str(metric.get("display_value", "N/A") or "N/A"),
                "status": str(metric.get("status", "unavailable") or "unavailable"),
                "note": str(metric.get("note", "") or ""),
                "as_of": str(metric.get("as_of", "") or ""),
            }
        )
    return cards


def _draw_metric_cards(ax, cards: List[Dict[str, str]]) -> None:
    ax.axis("off")
    _panel(ax)
    _panel_title(ax, "Hong Kong local tape", "Funding, flow, and relative value at a glance")
    row_height = 0.102
    row_gap = 0.016
    y = 0.76
    for idx, card in enumerate(cards[:6]):
        chip_text, chip_color = _status_chip(card["status"])
        value_text = textwrap.shorten(str(card["value"] or "N/A").replace("|", " / "), width=34, placeholder="...")
        as_of_text = textwrap.shorten(str(card.get("as_of") or ""), width=24, placeholder="...") if card.get("as_of") else ""
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
        ax.add_patch(
            FancyBboxPatch(
                (0.81, y + 0.056),
                0.13,
                0.032,
                boxstyle="round,pad=0.01,rounding_size=0.015",
                transform=ax.transAxes,
                linewidth=0.8,
                edgecolor=chip_color,
                facecolor="#ffffff",
            )
        )
        ax.text(0.055, y + 0.073, card["label"], transform=ax.transAxes, fontsize=9.1, color=SLATE, va="center")
        ax.text(
            0.875,
            y + 0.072,
            chip_text,
            transform=ax.transAxes,
            fontsize=7.8,
            color=chip_color,
            ha="center",
            va="center",
            fontweight="bold",
        )
        ax.text(
            0.355,
            y + 0.073,
            _safe_text(value_text),
            transform=ax.transAxes,
            fontsize=10.9,
            fontweight="bold",
            color=INK,
            va="center",
        )
        if as_of_text:
            ax.text(
                0.055,
                y + 0.025,
                _safe_text(f"As of {as_of_text}"),
                transform=ax.transAxes,
                fontsize=8.0,
                color=SLATE,
                va="bottom",
            )
        y -= row_height + row_gap


def _flow_focus(bundle: Dict[str, Any]) -> Tuple[str, str, List[Tuple[str, float, str]], str]:
    flow_tracker = bundle.get("flow_tracker", {}) or {}
    stock_connect = ((flow_tracker.get("stock_connect", {}) or {}).get("data", {}) or {})
    southbound_active = ((stock_connect.get("southbound", {}) or {}).get("top_active", []) or [])[:6]
    if southbound_active:
        rows = []
        for item in southbound_active:
            net = _parse_float(item.get("net_buy")) or 0.0
            name = str(item.get("name", "") or item.get("ticker", ""))
            rows.append((name, net, f"{net:+,.0f} HKD mn"))
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
            name = str(item.get("name", "") or item.get("ticker", ""))
            rows.append((name, ratio, f"{ratio:.1f}%"))
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


def _draw_flow_focus(ax, bundle: Dict[str, Any]) -> None:
    title, subtitle, rows, xlabel = _flow_focus(bundle)
    _panel(ax)
    _panel_title(ax, title, subtitle)
    labels = [textwrap.shorten(item[0], width=18, placeholder="...") for item in rows][:6]
    values = [item[1] for item in rows][:6]
    tags = [item[2] for item in rows][:6]
    colors = [_bar_color(value) for value in values]

    if not labels:
        ax.axis("off")
        ax.text(0.03, 0.55, "No ranked flow or pressure panel was available.", transform=ax.transAxes, fontsize=11, color=SLATE)
        return

    ax.barh(labels, values, color=colors, height=0.56)
    ax.axvline(0, color="#98a2b3", linewidth=1.0)
    ax.invert_yaxis()
    ax.set_ylim(len(labels) - 0.45, -1.35)
    ax.set_xlabel(xlabel, fontsize=9.5, color=SLATE)
    ax.tick_params(axis="x", labelsize=9)
    ax.tick_params(axis="y", labelsize=9.5)
    for idx, (value, tag) in enumerate(zip(values, tags)):
        ax.text(value + (0.02 * max(abs(v) for v in values) if values else 0.1) * (1 if value >= 0 else -1), idx, _safe_text(tag), va="center", ha="left" if value >= 0 else "right", fontsize=8.8, color=INK)


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
    row_height = 0.12
    row_gap = 0.022
    y = 0.67
    for idx, (time_label, tag, headline) in enumerate(lines[:4]):
        tag_color = BLUE if tag.lower() in {"upcoming", "released", "macro", "central bank"} else AMBER
        headline_text = textwrap.shorten(headline, width=68, placeholder="...")
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
        ax.text(0.055, y + 0.08, time_label or "Today", transform=ax.transAxes, fontsize=8.8, color=SLATE, va="center")
        ax.text(0.29, y + 0.08, f"[{tag}]", transform=ax.transAxes, fontsize=8.6, color=tag_color, va="center", fontweight="bold")
        ax.text(
            0.055,
            y + 0.03,
            _safe_text(headline_text),
            transform=ax.transAxes,
            fontsize=9.9,
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
        ax.text(0.28, y, textwrap.shorten(name, width=24, placeholder="..."), transform=ax.transAxes, fontsize=11, color=INK)
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


def generate_dashboard(bundle: Dict[str, Any], output_path: str) -> str:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.style.use("default")
    fig = plt.figure(figsize=(19.0, 12.4), facecolor=FIG_BG)
    grid = fig.add_gridspec(3, 4, hspace=0.24, wspace=0.16)

    report_date = bundle.get("meta", {}).get("report_date", "")
    theme = str((bundle.get("overview", {}) or {}).get("theme", "") or "")
    regime = str((bundle.get("overview", {}) or {}).get("risk_regime", "Neutral") or "Neutral")

    fig.suptitle(
        f"Hong Kong Morning Dashboard | {report_date}",
        fontsize=24,
        fontweight="bold",
        x=0.03,
        ha="left",
        color=INK,
        y=0.985,
    )
    fig.text(0.03, 0.948, textwrap.shorten(theme, width=116, placeholder="..."), fontsize=12.2, color=SLATE)
    fig.text(
        0.86,
        0.952,
        regime.upper(),
        fontsize=11,
        fontweight="bold",
        color=GREEN if regime.lower() == "risk-on" else RED if regime.lower() == "risk-off" else AMBER,
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=PANEL_BG, edgecolor=LINE),
    )

    ax_regime = fig.add_subplot(grid[0, :2])
    _panel(ax_regime)
    _panel_title(ax_regime, "Global regime board", "Cross-asset moves framing the open")
    rows = _top_snapshot_rows(bundle)
    labels = [row.get("label", "") for row in rows]
    values = [float(row.get("change_pct", 0) or 0) for row in rows]
    colors = [_bar_color(value) for value in values]
    ax_regime.barh(labels, values, color=colors, height=0.68)
    ax_regime.axvline(0, color="#98a2b3", linewidth=1.0)
    ax_regime.invert_yaxis()
    ax_regime.set_ylim(len(labels) - 0.45, -1.7)
    ax_regime.tick_params(axis="x", labelsize=9)
    ax_regime.tick_params(axis="y", labelsize=10.5)
    ax_regime.grid(axis="x", color="#e4e7ec", linewidth=0.8, alpha=0.75)
    for idx, value in enumerate(values):
        if abs(value) < 0.5:
            continue
        ax_regime.text(value + (0.08 if value >= 0 else -0.08), idx, _safe_text(f"{value:+.2f}%"), va="center", ha="left" if value >= 0 else "right", fontsize=9.5, color=INK)

    ax_hk = fig.add_subplot(grid[0, 2:])
    _draw_metric_cards(ax_hk, _hk_metric_cards(bundle))

    ax_flow = fig.add_subplot(grid[1, :2])
    _draw_flow_focus(ax_flow, bundle)

    ax_catalyst = fig.add_subplot(grid[1, 2:])
    _draw_catalysts(ax_catalyst, bundle)

    ax_watch = fig.add_subplot(grid[2, :2])
    _draw_watchlist(ax_watch, bundle)

    ax_brief = fig.add_subplot(grid[2, 2:])
    _draw_desk_frame(ax_brief, bundle)

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return os.path.basename(output_path)
