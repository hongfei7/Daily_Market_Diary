from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


INK = "#102a43"
SLATE = "#486581"
LINE = "#d9e2ec"
FIG_BG = "#f8fafc"
PANEL_BG = "#ffffff"
GREEN = "#1f7a3e"
RED = "#b42318"
AMBER = "#d97706"
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


def _wrap_text(value: Any, width: int, max_lines: int) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    lines = textwrap.wrap(text, width=width, break_long_words=False, break_on_hyphens=False)
    if len(lines) > max_lines and lines:
        clipped = lines[:max_lines]
        clipped[-1] = textwrap.shorten(clipped[-1], width=max(12, width - 2), placeholder="...")
        return "\n".join(clipped)
    return "\n".join(lines)


def _summary_pct(bundle: Dict[str, Any], category: str, name: str) -> float | None:
    item = ((bundle.get("market_summary", {}) or {}).get(category, {}) or {}).get(name, {})
    if not isinstance(item, dict):
        return None
    return _parse_float(item.get("Pct Change"))


def _hk_metric(bundle: Dict[str, Any], key: str) -> Dict[str, Any]:
    item = (bundle.get("hk_local", {}) or {}).get(key, {})
    return item if isinstance(item, dict) else {}


def _metric_value(bundle: Dict[str, Any], key: str) -> float | None:
    return _parse_float(_hk_metric(bundle, key).get("value"))


def _fmt_hkd_bn(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    return f"HK${number / 1_000_000_000:.1f}bn"


def _fmt_hkd_mn(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    return f"HK${number:,.0f}mn"


def _fmt_signed_hkd_mn(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    sign = "+" if number >= 0 else "-"
    return f"{sign}HK${abs(number):,.0f}mn"


def _fmt_signed_hkd_flow(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    sign = "+" if number >= 0 else "-"
    if abs(number) >= 1_000:
        return f"{sign}HK${abs(number) / 1_000:.1f}bn"
    return f"{sign}HK${abs(number):,.0f}mn"


def _fmt_hkd_bn_from_mn(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    return f"HK${number / 1_000:.1f}bn"


def _fmt_hkd_flow(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    if abs(number) >= 1_000:
        return f"HK${abs(number) / 1_000:.1f}bn"
    return f"HK${abs(number):,.0f}mn"


def _ticker_label(item: Dict[str, Any]) -> str:
    ticker = str(item.get("ticker", "") or "").strip()
    if ticker:
        return ticker
    code = str(item.get("code", "") or "").strip()
    return f"{code}.HK" if code else "N/A"


def _story_box_lines(story: Dict[str, Any]) -> List[str]:
    rows: List[str] = []
    for key, value in story.get("data_points", []) or []:
        rows.append(f"{key}: {value}")
    return rows


def _short_pressure_signal(short_rows: List[Dict[str, Any]], short_ratio: float | None) -> Dict[str, Any]:
    ratios = [_parse_float(item.get("short_ratio_pct")) or 0 for item in short_rows]
    names_above_20 = sum(1 for ratio in ratios if ratio >= 20)
    top_ratio = max(ratios or [0])
    chart_worthy = bool(short_rows) and short_ratio is not None and (
        short_ratio >= 16.0 or names_above_20 >= 3 or top_ratio >= 25.0
    )
    if chart_worthy:
        reason = "short pressure is broad or concentrated enough to lead the daily chart"
    elif short_rows and short_ratio is not None and short_ratio >= 14.0:
        reason = "short pressure is elevated but better treated as context unless concentration worsens"
    else:
        reason = "short pressure is not the dominant visual signal"
    return {
        "chart_worthy": chart_worthy,
        "names_above_20": names_above_20,
        "top_ratio": top_ratio,
        "reason": reason,
    }


def _southbound_signal(southbound: Dict[str, Any], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total_turnover = _parse_float(southbound.get("total_turnover"))
    net_buy = _parse_float(southbound.get("net_buy"))
    totals = [_parse_float(item.get("total_turnover")) or 0.0 for item in rows]
    nets = [_parse_float(item.get("net_buy")) or 0.0 for item in rows]
    top_total = max(totals or [0.0])
    top_abs_net = max((abs(value) for value in nets), default=0.0)
    active_net_count = sum(1 for value in nets if abs(value) >= 250.0)
    top_share = (top_total / total_turnover * 100) if total_turnover else None
    chart_worthy = bool(rows) and (
        (net_buy is not None and abs(net_buy) >= 3_000.0)
        or (top_share is not None and top_share >= 8.0)
        or top_abs_net >= 400.0
        or top_total >= 1_500.0
        or active_net_count >= 3
    )
    return {
        "chart_worthy": chart_worthy,
        "top_share": top_share,
        "top_abs_net": top_abs_net,
        "active_net_count": active_net_count,
    }


def _choose_story(bundle: Dict[str, Any]) -> Dict[str, Any]:
    flow_tracker = bundle.get("flow_tracker", {}) or {}
    attribution = bundle.get("attribution", {}) or {}
    short_rows = flow_tracker.get("short_sell_watchlist_hits", []) or flow_tracker.get("short_sell_top_ratio", []) or []
    stock_connect = (flow_tracker.get("stock_connect", {}) or {}).get("data", {}) or {}
    southbound = (stock_connect.get("southbound", {}) or {})
    southbound_active = (southbound.get("top_active", []) or [])
    ah_premium = (flow_tracker.get("ah_premium", {}) or {}).get("data", {}) or {}
    ah_rows = ah_premium.get("top_premium", []) or []
    short_ratio = _metric_value(bundle, "short_selling_ratio")
    turnover_ratio = _metric_value(bundle, "turnover_vs_20d")
    dxy = _summary_pct(bundle, "FX", "DXY")
    usdcnh = _summary_pct(bundle, "FX", "USD/CNH")
    hstech = _summary_pct(bundle, "Equities", "Hang Seng TECH ETF")
    fxi = _summary_pct(bundle, "Equities", "China Large-Cap (FXI)")
    brent = _summary_pct(bundle, "Commodities", "Brent Crude")
    oil = brent if brent is not None else _summary_pct(bundle, "Commodities", "Crude Oil")
    risk_dashboard = attribution.get("risk_dashboard", {}) or {}
    short_signal = _short_pressure_signal(short_rows, short_ratio)
    southbound_signal = _southbound_signal(southbound, southbound_active)

    if short_rows and short_ratio is not None and short_signal["chart_worthy"]:
        top_name = short_rows[0]
        return {
            "kind": "short_selling",
            "title": "Daily One Chart | HKEX short-selling pressure map",
            "caption": f"Market short-selling reached {short_ratio:.2f}% of turnover. The chart leads today because the pressure is either broad, concentrated, or acute in the top name.",
            "source": "HKEX Daily Quotations - Short Selling Turnover",
            "takeaway": "Use the chart to separate broad hedging from single-name conviction shorts.",
            "data_points": [
                ("Market short ratio", f"{short_ratio:.2f}%"),
                ("Names above 20%", str(short_signal["names_above_20"])),
                ("Top pressure", f"{_ticker_label(top_name)} {(_parse_float(top_name.get('short_ratio_pct')) or 0):.1f}%"),
            ],
            "watch_points": [
                "If ETF-heavy pressure dominates, the move is more about macro hedging than company-specific stress.",
                "If watchlist names dominate, the desk should prepare for stock-specific follow-up questions.",
            ],
        }

    if southbound_active and southbound_signal["chart_worthy"]:
        top_name = southbound_active[0]
        return {
            "kind": "stock_connect",
            "title": "Daily One Chart | Southbound active-name concentration",
            "caption": "This chart shows where disclosed Southbound activity was actually concentrated, which matters more for Hong Kong morning framing than the headline net-flow number alone.",
            "source": "HKEX Stock Connect Historical Daily",
            "takeaway": "When Connect flow is concentrated in only a few names, index strength can look broader than the underlying buying really is.",
            "data_points": [
                ("Southbound net", _hk_metric(bundle, "southbound_net_flow").get("display_value", "N/A")),
                ("Top active name", f"{_ticker_label(top_name)} {top_name.get('name', '')}"),
                ("Top name net", _fmt_hkd_mn(top_name.get("net_buy"))),
            ],
            "watch_points": [
                "Check platform, SOE, and ETF mix.",
                "Few-name flow is not broad confirmation.",
            ],
        }

    if ah_rows:
        top_name = ah_rows[0]
        average_premium = _hk_metric(bundle, "ah_premium_index").get("display_value", "N/A")
        return {
            "kind": "ah_premium",
            "title": "Daily One Chart | A/H premium dispersion",
            "caption": "Average A/H premium is less informative than dispersion. The desk should care about which names still show the widest cross-listing dislocation and whether that gap is starting to close.",
            "source": "Public Yahoo Finance quotes - calculated A/H premium",
            "takeaway": "Dispersion highlights relative-value stress and possible Southbound hunting ground better than the headline average alone.",
            "data_points": [
                ("Average premium", average_premium),
                ("Widest pair", f"{top_name.get('name', '')}"),
                ("Top premium", f"{(_parse_float(top_name.get('premium_pct')) or 0):.1f}%"),
            ],
            "watch_points": [
                "A narrowing premium in crowded names can signal cross-border arbitrage or Southbound demand.",
                "Persistently wide premiums often sit in policy-heavy or less liquid names; they are not automatically actionable.",
            ],
        }

    if turnover_ratio is not None and abs(turnover_ratio - 1.0) >= 0.10:
        turnover_display = _hk_metric(bundle, "turnover_vs_20d").get("display_value", "N/A")
        short_display = _hk_metric(bundle, "short_selling_ratio").get("display_value", "N/A")
        return {
            "kind": "turnover",
            "title": "Daily One Chart | Turnover conviction versus short pressure",
            "caption": "Turnover tells you whether the move deserves respect. Overlaying the market short ratio keeps you from confusing high activity with clean risk-taking.",
            "source": "HKEX Daily Quotations",
            "takeaway": "A high-turnover day with contained short-selling is conviction; a high-turnover day with heavy shorts can be a squeeze or hedge-driven churn.",
            "data_points": [
                ("Turnover vs 20D", turnover_display),
                ("Main Board turnover", _hk_metric(bundle, "main_board_turnover").get("display_value", "N/A")),
                ("Short ratio", short_display),
            ],
            "watch_points": [
                "Use it to judge whether the index move is investable or just noisy.",
                "Pair the signal with Southbound flow and style leadership before drawing a conclusion.",
            ],
        }

    fx_values = [value for value in (dxy, usdcnh) if value is not None]
    if fx_values and max(abs(value) for value in fx_values) >= 0.25:
        return {
            "kind": "fx_vs_growth",
            "title": "Daily One Chart | FX pressure versus Hong Kong growth",
            "caption": "The desk should always test whether FX pressure is confirming or contradicting Hong Kong growth leadership. DXY and USD/CNH are often the cleanest first filter.",
            "source": "Yahoo Finance market snapshot",
            "takeaway": "If Hong Kong growth is strong while USD/CNH is firming, the rally is working against a currency headwind and deserves closer scrutiny.",
            "data_points": [
                ("DXY", f"{(dxy or 0):+.2f}%"),
                ("USD/CNH", f"{(usdcnh or 0):+.2f}%"),
                ("HSTECH", f"{(hstech or 0):+.2f}%"),
            ],
            "watch_points": [
                "Strong growth with a firm dollar can still work, but the bar for follow-through is higher.",
                "If FX and growth agree, the market narrative is cleaner and easier to defend.",
            ],
        }

    if oil is not None and abs(oil) >= 2.5:
        return {
            "kind": "oil",
            "title": "Daily One Chart | Oil shock read-through",
            "caption": f"Oil moved {oil:+.2f}%. In a Hong Kong morning note, the right question is whether the move is geopolitical, inflationary, or demand-positive.",
            "source": "Yahoo Finance commodity snapshot",
            "takeaway": "Oil is not just an energy chart; it changes inflation expectations, rates pressure, and sector leadership.",
            "data_points": [
                ("Brent", f"{(brent or 0):+.2f}%"),
                ("HSTECH", f"{(hstech or 0):+.2f}%"),
                ("FXI", f"{(fxi or 0):+.2f}%"),
            ],
            "watch_points": [
                "If oil rises with gold and a stronger dollar, treat it as a macro stress signal first.",
                "If oil rises with cyclicals and weaker dollar pressure, the move is less threatening for risk assets.",
            ],
        }

    drivers = attribution.get("dominant_drivers", []) or []
    if drivers:
        top_driver = drivers[0]
        return {
            "kind": "attribution",
            "title": "Daily One Chart | Cross-asset attribution board",
            "caption": "When no single local-market chart dominates, the right fallback is to rank the rule-based drivers that are actually shaping the Hong Kong setup.",
            "source": "Deterministic attribution engine",
            "takeaway": "This keeps the morning note anchored in explicit drivers rather than vague narrative.",
            "data_points": [
                ("Risk score", f"{risk_dashboard.get('score', 'N/A')}/100"),
                ("Top driver", str(top_driver.get("name", "") or "N/A")),
                ("Direction", str(top_driver.get("direction", "") or "N/A")),
            ],
            "watch_points": [
                "A mixed board means the desk should avoid overfitting to one macro explanation.",
                "Use attribution as a framing tool, not a replacement for price action.",
            ],
        }

    return {
        "kind": "risk_score",
        "title": "Daily One Chart | Composite risk score",
        "caption": "No single signal dominated, so the daily chart falls back to the composite risk board. That is still useful because it forces a ranked rather than anecdotal market read.",
        "source": "Deterministic risk dashboard",
        "takeaway": "Use the score as a framing device, then check whether local Hong Kong flow metrics confirm the same regime.",
        "data_points": [
            ("Risk score", f"{risk_dashboard.get('score', 'N/A')}/100"),
            ("Regime", str(risk_dashboard.get("bucket", "Mixed"))),
        ],
        "watch_points": [
            "If local flow disagrees with the composite score, trust the conflict and dig deeper.",
            "A fallback risk score is acceptable for framing, but not enough for stock-specific conviction.",
        ],
    }


def _decorate_main(ax, title: str) -> None:
    ax.set_title(title, loc="left", fontsize=18, fontweight="bold", color=INK, pad=18)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="#e4e7ec", linewidth=0.8, alpha=0.75)


def _plot_short_selling(ax, bundle: Dict[str, Any]) -> None:
    rows = (bundle.get("flow_tracker", {}) or {}).get("short_sell_watchlist_hits", []) or (bundle.get("flow_tracker", {}) or {}).get("short_sell_top_ratio", []) or []
    visible = rows[:8]
    labels = [f"{_ticker_label(item)}\n{item.get('name', '')}" for item in visible]
    ratios = [_parse_float(item.get("short_ratio_pct")) or 0 for item in visible]
    values = [_parse_float(item.get("short_turnover_hkd")) or 0 for item in visible]
    colors = [RED if value >= 20 else AMBER if value >= 14 else GREEN for value in ratios]
    ax.barh(labels, ratios, color=colors)
    ax.axvline(14, color=AMBER, linewidth=1.2, linestyle="--")
    ax.axvline(20, color=RED, linewidth=1.2, linestyle="--")
    ax.invert_yaxis()
    ax.set_xlabel("Short-selling turnover / total turnover (%)")
    max_ratio = max(ratios or [20])
    ax.set_xlim(0, max(25, max_ratio + max(6.0, max_ratio * 0.32)))
    for idx, (ratio, value) in enumerate(zip(ratios, values)):
        ax.text(ratio + 0.5, idx, f"{ratio:.1f}% | {_fmt_hkd_bn(value)}", va="center", fontsize=9.2, color=INK, clip_on=False)


def _plot_turnover(ax, bundle: Dict[str, Any]) -> None:
    turnover_ratio = _metric_value(bundle, "turnover_vs_20d") or 0
    short_ratio = _metric_value(bundle, "short_selling_ratio")
    labels = ["Turnover vs 20D", "Neutral baseline"]
    values = [turnover_ratio, 1.0]
    colors = [GREEN if turnover_ratio >= 1 else RED, "#98a2b3"]
    ax.bar(labels, values, color=colors, width=0.52)
    ax.axhline(1.0, color="#475467", linewidth=1.2, linestyle="--")
    ax.set_ylabel("Multiple of 20-session average")
    ax.set_ylim(0, max(1.5, turnover_ratio + 0.3))
    ax.text(0, turnover_ratio + 0.05, f"{turnover_ratio:.2f}x", ha="center", fontsize=12, fontweight="bold", color=INK)
    if short_ratio is not None:
        ax2 = ax.twinx()
        ax2.scatter([0], [short_ratio], s=260, color=AMBER, edgecolor="white", linewidth=1.5, zorder=5)
        ax2.set_ylabel("Short ratio (%)", color=SLATE)
        ax2.set_ylim(0, max(25, short_ratio + 5))
        ax2.text(0.10, short_ratio, f"{short_ratio:.1f}%", va="center", color=AMBER, fontsize=10)
        for spine in ax2.spines.values():
            spine.set_visible(False)


def _plot_fx_vs_growth(ax, bundle: Dict[str, Any]) -> None:
    rows: List[Tuple[str, float]] = [
        ("DXY", _summary_pct(bundle, "FX", "DXY") or 0),
        ("USD/CNH", _summary_pct(bundle, "FX", "USD/CNH") or 0),
        ("HSTECH", _summary_pct(bundle, "Equities", "Hang Seng TECH ETF") or 0),
        ("FXI", _summary_pct(bundle, "Equities", "China Large-Cap (FXI)") or 0),
    ]
    labels = [item[0] for item in rows]
    values = [item[1] for item in rows]
    colors = [RED if (label in {"DXY", "USD/CNH"} and value > 0) or (label not in {"DXY", "USD/CNH"} and value < 0) else GREEN for label, value in rows]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#98a2b3", linewidth=1)
    ax.set_ylabel("1D move (%)")
    for idx, value in enumerate(values):
        ax.text(idx, value + (0.08 if value >= 0 else -0.08), f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=10, color=INK)


def _plot_oil(ax, bundle: Dict[str, Any]) -> None:
    rows = [
        ("Brent", _summary_pct(bundle, "Commodities", "Brent Crude") or 0),
        ("WTI", _summary_pct(bundle, "Commodities", "Crude Oil") or 0),
        ("Gold", _summary_pct(bundle, "Commodities", "Gold") or 0),
        ("HSI", _summary_pct(bundle, "Equities", "Hang Seng Index") or 0),
        ("HSTECH", _summary_pct(bundle, "Equities", "Hang Seng TECH ETF") or 0),
    ]
    labels = [item[0] for item in rows]
    values = [item[1] for item in rows]
    colors = [AMBER if label in {"Brent", "WTI"} else GREEN if value >= 0 else RED for label, value in rows]
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color="#98a2b3", linewidth=1)
    ax.set_ylabel("1D move (%)")
    for idx, value in enumerate(values):
        ax.text(idx, value + (0.08 if value >= 0 else -0.08), f"{value:+.2f}%", ha="center", va="bottom" if value >= 0 else "top", fontsize=10, color=INK)


def _plot_attribution(ax, bundle: Dict[str, Any]) -> None:
    drivers = ((bundle.get("attribution", {}) or {}).get("dominant_drivers", []) or [])[:6]
    if not drivers:
        _plot_risk_score(ax, bundle)
        return
    labels = [item.get("name", "") for item in drivers]
    values = [float(item.get("score", 0) or 0) for item in drivers]
    colors = [GREEN if item.get("direction") == "supportive" else RED if item.get("direction") == "drag" else AMBER for item in drivers]
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Attribution score")
    max_value = max(values or [1])
    ax.set_xlim(0, max_value + max(1.2, max_value * 0.22))
    for idx, item in enumerate(drivers):
        ax.text(values[idx] + 0.08, idx, item.get("direction", ""), va="center", fontsize=9, color=INK)


def _plot_stock_connect(ax, bundle: Dict[str, Any]) -> None:
    stock_connect = (((bundle.get("flow_tracker", {}) or {}).get("stock_connect", {}) or {}).get("data", {}) or {})
    southbound = stock_connect.get("southbound", {}) or {}
    rows = (southbound.get("top_active", []) or [])[:5]
    if not rows:
        _plot_risk_score(ax, bundle)
        return
    ax.axis("off")

    total_turnover = _parse_float(southbound.get("total_turnover")) or 0.0
    buy_turnover_raw = _parse_float(southbound.get("buy_turnover"))
    sell_turnover_raw = _parse_float(southbound.get("sell_turnover"))
    buy_turnover = buy_turnover_raw or 0.0
    sell_turnover = sell_turnover_raw or 0.0
    net_buy = _parse_float(southbound.get("net_buy"))
    active_rows: List[Dict[str, Any]] = []
    for item in rows:
        buy_raw = _parse_float(item.get("buy_turnover"))
        sell_raw = _parse_float(item.get("sell_turnover"))
        buy = buy_raw or 0.0
        sell = sell_raw or 0.0
        net = _parse_float(item.get("net_buy")) or 0.0
        total = _parse_float(item.get("total_turnover")) or buy + sell or abs(net)
        active_rows.append({**item, "_buy": buy, "_sell": sell, "_net": net, "_total": total, "_has_split": buy_raw is not None and sell_raw is not None and buy + sell > 0})
    active_total = sum(item["_total"] for item in active_rows)
    top_total = max((item["_total"] for item in active_rows), default=0.0)
    denominator = total_turnover or active_total
    share_scope = "tape" if total_turnover else "active list"
    top_share = (top_total / denominator * 100) if denominator else None
    has_aggregate_split = buy_turnover_raw is not None and sell_turnover_raw is not None and buy_turnover + sell_turnover > 0
    buy_sell_total = max(buy_turnover + sell_turnover, 1.0)
    buy_share = buy_turnover / buy_sell_total if has_aggregate_split else 0.0
    sell_share = sell_turnover / buy_sell_total if has_aggregate_split else 0.0
    positive_count = sum(1 for item in active_rows if item["_net"] >= 0)
    negative_count = len(active_rows) - positive_count
    max_abs_net = max(max((abs(item["_net"]) for item in active_rows), default=1.0), 1.0)

    def card(x: float, y: float, w: float, h: float, face: str = "#f8fafc") -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                transform=ax.transAxes,
                linewidth=0.9,
                edgecolor=LINE,
                facecolor=face,
            )
        )

    net_color = GREEN if (net_buy or 0) >= 0 else RED
    card(0.02, 0.775, 0.24, 0.145)
    ax.text(0.045, 0.875, "SOUTHBOUND NET", transform=ax.transAxes, fontsize=8.5, color=SLATE, fontweight="bold", va="center")
    ax.text(
        0.045,
        0.818,
        _safe_text(_fmt_signed_hkd_flow(net_buy) if net_buy is not None else _hk_metric(bundle, "southbound_net_flow").get("display_value", "N/A")),
        transform=ax.transAxes,
        fontsize=18.0,
        color=net_color,
        fontweight="bold",
        va="center",
    )

    card(0.30, 0.775, 0.39, 0.145)
    ax.text(0.325, 0.875, "BUY / SELL TURNOVER MIX", transform=ax.transAxes, fontsize=8.5, color=SLATE, fontweight="bold", va="center")
    mix_x, mix_y, mix_w, mix_h = 0.325, 0.815, 0.31, 0.030
    ax.add_patch(Rectangle((mix_x, mix_y), mix_w, mix_h, transform=ax.transAxes, linewidth=0, facecolor="#e4e7ec"))
    if has_aggregate_split:
        ax.add_patch(Rectangle((mix_x, mix_y), mix_w * buy_share, mix_h, transform=ax.transAxes, linewidth=0, facecolor=GREEN))
        ax.add_patch(Rectangle((mix_x + mix_w * buy_share, mix_y), mix_w * sell_share, mix_h, transform=ax.transAxes, linewidth=0, facecolor=RED))
        ax.text(0.325, 0.795, _safe_text(f"Buy {_fmt_hkd_bn_from_mn(buy_turnover)}"), transform=ax.transAxes, fontsize=9.2, color=GREEN, fontweight="bold", va="top")
        ax.text(0.635, 0.795, _safe_text(f"Sell {_fmt_hkd_bn_from_mn(sell_turnover)}"), transform=ax.transAxes, fontsize=9.2, color=RED, fontweight="bold", va="top", ha="right")
    else:
        ax.text(0.325, 0.795, "Split unavailable", transform=ax.transAxes, fontsize=9.2, color=SLATE, fontweight="bold", va="top")

    card(0.73, 0.775, 0.23, 0.145)
    concentration_value = top_share if top_share is not None else 0.0
    concentration_label = "single-name heavy" if concentration_value >= 20 else "moderate" if concentration_value >= 12 else "distributed"
    ax.text(0.755, 0.875, "CONCENTRATION", transform=ax.transAxes, fontsize=8.5, color=SLATE, fontweight="bold", va="center")
    ax.text(0.755, 0.823, f"{concentration_value:.1f}%", transform=ax.transAxes, fontsize=17.0, color=BLUE, fontweight="bold", va="center")
    ax.text(0.855, 0.823, concentration_label, transform=ax.transAxes, fontsize=9.2, color=SLATE, va="center")

    ax.text(0.025, 0.692, "ACTIVE-NAME LEDGER", transform=ax.transAxes, fontsize=9.2, color=BLUE, fontweight="bold")
    ax.text(0.045, 0.653, "Name", transform=ax.transAxes, fontsize=8.8, color=SLATE, fontweight="bold")
    ax.text(0.310, 0.653, "Net flow", transform=ax.transAxes, fontsize=8.8, color=SLATE, fontweight="bold", ha="right")
    ax.text(0.510, 0.653, "Direction", transform=ax.transAxes, fontsize=8.8, color=SLATE, fontweight="bold", ha="center")
    ax.text(0.650, 0.653, "Active share", transform=ax.transAxes, fontsize=8.8, color=SLATE, fontweight="bold", ha="center")
    ax.text(0.845, 0.653, "Buy / sell", transform=ax.transAxes, fontsize=8.8, color=SLATE, fontweight="bold", ha="center")

    y = 0.575
    row_h = 0.080
    for rank, item in enumerate(active_rows, start=1):
        ticker = _ticker_label(item)
        name = str(item.get("name", "") or "").strip()
        net = float(item["_net"])
        buy = float(item["_buy"])
        sell = float(item["_sell"])
        total = float(item["_total"])
        has_split = bool(item.get("_has_split"))
        color = GREEN if net >= 0 else RED
        card(0.02, y, 0.94, row_h, "#ffffff" if rank % 2 else "#f8fafc")
        ax.add_patch(Rectangle((0.02, y), 0.008, row_h, transform=ax.transAxes, linewidth=0, facecolor=color))
        ax.text(0.045, y + 0.050, f"{rank}. {ticker}", transform=ax.transAxes, fontsize=10.6, color=INK, va="center", fontweight="bold")
        ax.text(0.045, y + 0.023, _safe_text(textwrap.shorten(name or "Name unavailable", width=18, placeholder="...")), transform=ax.transAxes, fontsize=8.6, color=SLATE, va="center")
        ax.text(0.310, y + 0.040, _safe_text(_fmt_signed_hkd_mn(net)), transform=ax.transAxes, fontsize=10.2, color=color, va="center", ha="right", fontweight="bold")

        baseline = 0.495
        span = 0.095 * abs(net) / max_abs_net
        ax.add_patch(Rectangle((baseline - 0.095, y + 0.032), 0.190, 0.018, transform=ax.transAxes, linewidth=0, facecolor="#e4e7ec"))
        if net >= 0:
            ax.add_patch(Rectangle((baseline, y + 0.032), span, 0.018, transform=ax.transAxes, linewidth=0, facecolor=GREEN))
        else:
            ax.add_patch(Rectangle((baseline - span, y + 0.032), span, 0.018, transform=ax.transAxes, linewidth=0, facecolor=RED))
        ax.add_patch(Rectangle((baseline - 0.001, y + 0.026), 0.002, 0.030, transform=ax.transAxes, linewidth=0, facecolor="#98a2b3"))

        active_share = total / denominator * 100 if denominator else 0.0
        ax.text(0.650, y + 0.048, _safe_text(_fmt_hkd_flow(total)), transform=ax.transAxes, fontsize=9.3, color=INK, va="center", ha="center", fontweight="bold")
        ax.text(0.650, y + 0.022, f"{active_share:.1f}% of {share_scope}", transform=ax.transAxes, fontsize=8.0, color=SLATE, va="center", ha="center")

        bar_x = 0.765
        bar_y = y + 0.032
        bar_w = 0.085
        mix_total = max(buy + sell, 1.0)
        buy_w = bar_w * max(0.0, buy) / mix_total
        sell_w = bar_w * max(0.0, sell) / mix_total
        ax.add_patch(Rectangle((bar_x, bar_y), bar_w, 0.030, transform=ax.transAxes, linewidth=0, facecolor="#e4e7ec"))
        if has_split:
            ax.add_patch(Rectangle((bar_x, bar_y), buy_w, 0.030, transform=ax.transAxes, linewidth=0, facecolor=GREEN))
            ax.add_patch(Rectangle((bar_x + buy_w, bar_y), sell_w, 0.030, transform=ax.transAxes, linewidth=0, facecolor=RED))
            split_text = f"{buy:,.0f}/{sell:,.0f}"
        else:
            split_text = "N/A"
        ax.text(0.955, y + 0.047, split_text, transform=ax.transAxes, fontsize=8.2, color=SLATE, va="center", ha="right")
        y -= 0.095

    read_color = RED if concentration_value >= 20 else AMBER if concentration_value >= 12 else BLUE
    read = (
        f"Read: {positive_count} net-bought / {negative_count} net-sold active names; "
        f"top {share_scope} share {concentration_value:.1f}%. "
        f"{'Flow is concentrated; test single-name sustainability.' if concentration_value >= 12 else 'Flow is distributed; use breadth confirmation before extrapolating.'}"
    )
    card(0.02, 0.075, 0.94, 0.095, "#ffffff")
    ax.add_patch(Rectangle((0.02, 0.075), 0.008, 0.095, transform=ax.transAxes, linewidth=0, facecolor=read_color))
    ax.text(0.045, 0.123, _safe_text(_wrap_text(read, width=104, max_lines=2)), transform=ax.transAxes, fontsize=9.2, color=INK, va="center")


def _plot_ah_premium(ax, bundle: Dict[str, Any]) -> None:
    ah_premium = (((bundle.get("flow_tracker", {}) or {}).get("ah_premium", {}) or {}).get("data", {}) or {})
    rows = (ah_premium.get("top_premium", []) or [])[:8]
    if not rows:
        _plot_risk_score(ax, bundle)
        return
    labels = [f"{item.get('name', '')}\n{item.get('h_ticker', '')}" for item in rows]
    values = [_parse_float(item.get("premium_pct")) or 0 for item in rows]
    colors = [RED if value >= 80 else AMBER if value >= 40 else GREEN for value in values]
    ax.barh(labels, values, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("A-share premium versus H-share (%)")
    max_value = max(values or [40])
    ax.set_xlim(0, max(100, max_value + max(10.0, max_value * 0.18)))
    for idx, value in enumerate(values):
        ax.text(value + 1.0, idx, f"{value:+.1f}%", va="center", fontsize=9, color=INK)


def _plot_risk_score(ax, bundle: Dict[str, Any]) -> None:
    dashboard = ((bundle.get("attribution", {}) or {}).get("risk_dashboard", {}) or {})
    score = float(dashboard.get("score", 50) or 50)
    components = dashboard.get("components", []) or []
    ax.barh(["Composite risk score"], [score], color=GREEN if score >= 65 else RED if score <= 35 else AMBER, height=0.56)
    ax.set_xlim(0, 100)
    ax.axvspan(0, 35, color="#fee4e2", alpha=0.65)
    ax.axvspan(35, 65, color="#fef3c7", alpha=0.65)
    ax.axvspan(65, 100, color="#dcfce7", alpha=0.65)
    ax.text(score + 1, 0, f"{score:.1f}/100 | {dashboard.get('bucket', 'Mixed')}", va="center", fontsize=12, fontweight="bold", color=INK)
    if components:
        detail = " | ".join(f"{item.get('label')}: {item.get('delta'):+}" for item in components[:4])
        ax.text(0, -0.55, textwrap.fill(detail, width=56), fontsize=9.8, color=SLATE)


PLOTTERS = {
    "short_selling": _plot_short_selling,
    "turnover": _plot_turnover,
    "fx_vs_growth": _plot_fx_vs_growth,
    "oil": _plot_oil,
    "stock_connect": _plot_stock_connect,
    "ah_premium": _plot_ah_premium,
    "attribution": _plot_attribution,
    "risk_score": _plot_risk_score,
}


def _draw_side_panel(ax, story: Dict[str, Any]) -> None:
    ax.axis("off")
    ax.add_patch(
        FancyBboxPatch(
            (0, 0),
            1,
            1,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            transform=ax.transAxes,
            linewidth=1.0,
            edgecolor=LINE,
            facecolor=PANEL_BG,
        )
    )
    ax.text(0.07, 0.92, "Desk takeaway", transform=ax.transAxes, fontsize=12, fontweight="bold", color=INK, va="top")
    ax.text(0.07, 0.82, _safe_text(textwrap.fill(story.get("takeaway", ""), width=32)), transform=ax.transAxes, fontsize=10.2, color=SLATE, va="top")

    ax.text(0.07, 0.61, "Key datapoints", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)
    y = 0.53
    for line in _story_box_lines(story)[:4]:
        ax.text(0.07, y, _safe_text(textwrap.fill(f"- {line}", width=34)), transform=ax.transAxes, fontsize=9.8, color=INK, va="top")
        y -= 0.11

    ax.text(0.07, 0.24, "What to watch", transform=ax.transAxes, fontsize=11, fontweight="bold", color=INK)
    y = 0.17
    for line in (story.get("watch_points", []) or [])[:2]:
        ax.text(0.07, y, _safe_text(_wrap_text(f"- {line}", width=34, max_lines=2)), transform=ax.transAxes, fontsize=9.1, color=SLATE, va="top")
        y -= 0.10


def generate_daily_one_chart(bundle: Dict[str, Any], output_path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    story = _choose_story(bundle)

    plt.style.use("default")
    fig = plt.figure(figsize=(15.2, 8.2), facecolor=FIG_BG)
    grid = fig.add_gridspec(1, 2, width_ratios=[3.2, 1.35], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0], facecolor=PANEL_BG)
    side_ax = fig.add_subplot(grid[0, 1], facecolor=PANEL_BG)

    _decorate_main(ax, story["title"])
    PLOTTERS.get(story["kind"], _plot_risk_score)(ax, bundle)
    _draw_side_panel(side_ax, story)

    fig.text(0.07, 0.08, _safe_text(textwrap.fill(story["caption"], width=120)), fontsize=10.5, color="#334e68")
    fig.text(0.07, 0.045, _safe_text(f"Source: {story['source']}"), fontsize=9.4, color="#627d98")

    fig.subplots_adjust(left=0.09, right=0.96, top=0.92, bottom=0.18)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": os.path.basename(output_path),
        "title": story["title"],
        "caption": story["caption"],
        "source": story["source"],
        "kind": story["kind"],
    }
