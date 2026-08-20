from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

from market_diary.professional.instruments import (
    MAX_FRESH_TRADING_DAYS,
    format_summary_change,
    summary_change,
)
from market_diary.professional.metric_history import describe, percentile_context


def _parse_pct(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _format_signed(value: Optional[float], digits: int = 2, suffix: str = "%") -> str:
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}{suffix}"


def _summary_item(summary: Dict[str, Any], category: str, name: str) -> Dict[str, Any]:
    item = (summary or {}).get(category, {}).get(name, {})
    return item if isinstance(item, dict) else {}


def _freshness_days(item: Dict[str, Any]) -> Optional[int]:
    value = item.get("Trading Freshness Days", item.get("Freshness Days"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _snapshot_row(summary: Dict[str, Any], category: str, name: str, label: str, question: str) -> Dict[str, Any]:
    item = _summary_item(summary, category, name)
    price = _parse_float(item.get("Price"))
    change_value, change_unit = summary_change(item)
    freshness_days = _freshness_days(item)
    quality = str(item.get("Quality", "fresh") or "fresh")
    if freshness_days is not None and freshness_days > MAX_FRESH_TRADING_DAYS:
        quality = "stale"
    return {
        "freshness_days": freshness_days,
        "quality": quality,
        "is_stale": quality == "stale",
        "label": str(item.get("Display Name") or label),
        "short_label": label,
        "category": category,
        "symbol": name,
        "price": price,
        "price_unit": str(item.get("Price Unit", "quoted_price") or "quoted_price"),
        "security_type": str(item.get("Security Type", category.lower()) or category.lower()),
        "change_value": change_value,
        "change_unit": change_unit,
        "change_display": format_summary_change(item),
        "change_pct": change_value if change_unit == "pct" else None,
        "question": question,
    }


def build_market_snapshot(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracked = [
        ("Equities", "S&P 500", "S&P 500", "Risk appetite"),
        ("Equities", "Nasdaq 100", "Nasdaq 100", "Growth style"),
        ("Equities", "Euro Stoxx 50", "Euro Stoxx 50", "European risk appetite"),
        ("Equities", "Hang Seng Index", "Hang Seng Index", "Hong Kong beta"),
        ("Equities", "Hang Seng China Enterprises", "HSCEI", "China SOE / H-share tone"),
        ("Equities", "Hang Seng TECH ETF", "3033.HK ETF", "Hong Kong growth / internet proxy"),
        ("Equities", "China Large-Cap (FXI)", "China proxy (FXI)", "China sentiment"),
        ("Rates", "10Y Treasury", "US 10Y", "Rates path"),
        ("FX", "DXY", "DXY", "Global liquidity"),
        ("FX", "USD/CNH", "USD/CNH", "China FX sentiment"),
        ("FX", "USD/HKD", "USD/HKD", "Hong Kong funding lens"),
        ("Commodities", "Crude Oil", "WTI crude", "Geopolitics and demand"),
        ("Commodities", "Gold", "Gold", "Hedge / real yields"),
        ("Commodities", "Copper", "Copper", "Growth expectations"),
        ("Crypto", "Bitcoin", "Bitcoin", "Risk appetite supplement"),
        ("Vol", "VIX", "VIX", "Volatility and hedging"),
    ]
    return [_snapshot_row(summary, *row) for row in tracked]


def _get_row(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for row in rows:
        if row.get("label") == label or row.get("short_label") == label:
            return row
    return {}


def _fresh_change_pct(rows: Iterable[Dict[str, Any]], label: str) -> tuple[Optional[float], Optional[int]]:
    """Return ``(change_pct, stale_days)`` for a tracked instrument.

    A stale quote yields ``(None, age_in_trading_days)`` so callers cannot use
    the value by accident while still being able to explain why it was dropped.
    A fresh or missing quote yields ``(value, None)``.
    """
    row = _get_row(rows, label)
    if row.get("is_stale"):
        return None, row.get("freshness_days")
    return row.get("change_pct"), None


def _stale_note(label: str, stale_days: Optional[int]) -> str:
    if stale_days is None:
        return f"{label} was unavailable"
    suffix = "trading day" if stale_days == 1 else "trading days"
    return f"{label} was stale ({stale_days} {suffix} old)"


def build_chart_read(chart_features: Dict[str, Any]) -> Dict[str, List[str]]:
    fx_bullets: List[str] = []
    asset_bullets: List[str] = []

    fx = chart_features.get("fx_composite", {}) or {}
    if fx.get("available"):
        turning_points = fx.get("turning_points", []) or []
        direction = "higher" if (fx.get("net_pp") or 0) > 0 else "lower"
        fx_bullets.append(
            f"USD composite moved {direction} by {_format_signed(fx.get('net_pp'), digits=2, suffix='pct')} intraday, with a {fx.get('range_pp', 'N/A')}pct range."
        )
        if turning_points:
            tp = " / ".join(f"{item['time']} {item['direction']}" for item in turning_points[:3])
            fx_bullets.append(f"Main turning points appeared around {tp}, suggesting event-driven repricing rather than a clean one-way USD move.")
    else:
        fx_bullets.append("USD chart features were unavailable, so the read should lean more on closing data and the event tape.")

    divergence = chart_features.get("divergence", {}) or {}
    if divergence:
        asset_bullets.append(
            f"The biggest cross-asset divergence was {divergence.get('best_asset')} versus {divergence.get('worst_asset')}, a spread of {divergence.get('spread_pp')}pct."
        )

    assets = chart_features.get("assets", {}) or {}
    for label in ("Gold", "Oil", "Bitcoin"):
        asset = assets.get(label, {}) or {}
        if asset.get("available"):
            asset_bullets.append(
                f"{label} moved {_format_signed(asset.get('net_pp'), digits=2, suffix='pct')} intraday with a {asset.get('range_pp', 'N/A')}pct high-low range."
            )

    if not asset_bullets:
        asset_bullets.append("Gold / oil / Bitcoin chart features were unavailable.")

    return {"fx": fx_bullets[:3], "assets": asset_bullets[:4]}


def build_market_overview(summary: Dict[str, Any], chart_features: Dict[str, Any]) -> Dict[str, Any]:
    rows = build_market_snapshot(summary)

    spx = _get_row(rows, "S&P 500").get("change_pct")
    ndx = _get_row(rows, "Nasdaq 100").get("change_pct")
    hsi = _get_row(rows, "Hang Seng Index").get("change_pct")
    hstech = _get_row(rows, "3033.HK ETF").get("change_pct")
    fxi = _get_row(rows, "China proxy (FXI)").get("change_pct")
    dxy = _get_row(rows, "DXY").get("change_pct")
    us10y_bp = _get_row(rows, "US 10Y").get("change_value")
    vix = _get_row(rows, "VIX").get("change_pct")
    oil = _get_row(rows, "WTI crude").get("change_pct")
    gold = _get_row(rows, "Gold").get("change_pct")

    score = 0
    for value in (spx, ndx, hsi, hstech, fxi):
        if value is not None:
            score += 1 if value > 0.3 else -1 if value < -0.3 else 0
    if dxy is not None:
        score += -1 if dxy > 0.3 else 1 if dxy < -0.3 else 0
    if vix is not None:
        score += -1 if vix > 1.0 else 1 if vix < -1.0 else 0
    if us10y_bp is not None:
        score += -1 if us10y_bp > 5.0 else 1 if us10y_bp < -5.0 else 0

    if score >= 2:
        risk_regime = "Risk-On"
    elif score <= -2:
        risk_regime = "Risk-Off"
    else:
        risk_regime = "Neutral"

    chart_read = build_chart_read(chart_features)
    usd_net = (chart_features.get("fx_composite", {}) or {}).get("net_pp") or 0
    usd_bias = "USD stronger" if usd_net > 0.15 else "USD softer" if usd_net < -0.15 else "USD range-bound"
    rate_bias = "lower yields supported duration" if (us10y_bp or 0) < -5.0 else "higher yields pressured valuations" if (us10y_bp or 0) > 5.0 else "rates were not the dominant driver"
    asset_div = chart_features.get("divergence", {}) or {}
    divergence_text = f"{asset_div.get('best_asset')} outperformed {asset_div.get('worst_asset')}" if asset_div else "cross-asset divergence stayed modest"
    theme = f"{risk_regime} backdrop with {usd_bias}, {rate_bias}, and {divergence_text}"

    notes = [
        f"Risk appetite snapshot: S&P 500 {_format_signed(spx)} / Nasdaq 100 {_format_signed(ndx)} / Hang Seng {_format_signed(hsi)} / 3033.HK ETF {_format_signed(hstech)} / FXI {_format_signed(fxi)}.",
        f"Rates and liquidity: US 10Y {_format_signed(us10y_bp, digits=1, suffix='bp')} / DXY {_format_signed(dxy)} / VIX {_format_signed(vix)}.",
        f"Commodities and hedges: WTI {_format_signed(oil)} / Gold {_format_signed(gold)}.",
    ]

    key_questions = [
        f"Is risk appetite rising or fading? The overnight tape reads closer to `{risk_regime}`.",
        "Did rates expectations move? Focus on whether US 10Y, DXY, and growth style moved together.",
        "Were commodities the real story? Watch WTI and gold for geopolitics versus inflation signals.",
        "Is Hong Kong setup growth-led, SOE-led, or broad beta-led? Compare the 3033.HK ETF proxy, HSCEI, and HSI.",
        "Did offshore markets move China risk appetite? Focus on HSI, FXI, USD/CNH, and USD/HKD.",
    ]

    return {
        "theme": theme,
        "risk_regime": risk_regime,
        "snapshot_rows": rows,
        "notes": notes,
        "questions": key_questions,
        "chart_read": chart_read,
    }


def build_hk_desk_view(
    summary: Dict[str, Any],
    hk_local: Optional[Dict[str, Any]] = None,
    metric_history: Optional[Dict[str, Any]] = None,
    report_date: str = "",
) -> Dict[str, Any]:
    return build_hk_investor_lens(summary, hk_local, metric_history, report_date)


def _local_metric_value(hk_local: Dict[str, Any], key: str) -> Optional[float]:
    metric = (hk_local or {}).get(key, {}) or {}
    value = _parse_float(metric.get("value"))
    if value is not None:
        return value
    display = str(metric.get("display_value", "") or "")
    match = re.search(r"([+-]?[0-9][0-9,]*(?:\.[0-9]+)?)", display)
    if not match:
        return None
    parsed = _parse_float(match.group(1))
    if parsed is None:
        return None
    if key == "southbound_net_flow":
        lowered = display.lower()
        if "-" in display[: match.start()]:
            parsed = -abs(parsed)
        if "bn" in lowered:
            parsed *= 1_000_000_000
        elif "mn" in lowered:
            parsed *= 1_000_000
    return parsed


def _compact_hkd_flow(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else "" if value == 0 else "-"
    return f"{sign}HK${abs(value) / 1_000_000_000:.1f}bn"


def build_hk_investor_lens(
    summary: Dict[str, Any],
    hk_local: Optional[Dict[str, Any]] = None,
    metric_history: Optional[Dict[str, Any]] = None,
    report_date: str = "",
) -> Dict[str, Any]:
    """Build a fact-led Hong Kong style read that survives an LLM failure.

    ``leadership`` remains a compact compatibility label for tables and charts.
    ``lens`` is the decision-useful conclusion: relative performance, quality of
    confirmation, investment implication, and explicit prove/kill conditions.
    """
    rows = build_market_snapshot(summary)
    # The style call compares relative performance, so a stale leg would produce a
    # spread between two different dates. Drop stale legs and record why.
    hsi, hsi_stale = _fresh_change_pct(rows, "Hang Seng Index")
    hscei, hscei_stale = _fresh_change_pct(rows, "HSCEI")
    hstech, hstech_stale = _fresh_change_pct(rows, "3033.HK ETF")
    fxi, _ = _fresh_change_pct(rows, "China proxy (FXI)")
    usdcnh, _ = _fresh_change_pct(rows, "USD/CNH")
    usdhkd = _get_row(rows, "USD/HKD").get("price")
    turnover_ratio = _local_metric_value(hk_local or {}, "turnover_vs_20d")
    short_ratio = _local_metric_value(hk_local or {}, "short_selling_ratio")
    southbound = _local_metric_value(hk_local or {}, "southbound_net_flow")

    stale_inputs: List[str] = []
    if hstech is None and hstech_stale is not None:
        stale_inputs.append(_stale_note("3033.HK ETF", hstech_stale))
    if hscei is None and hscei_stale is not None:
        stale_inputs.append(_stale_note("HSCEI", hscei_stale))
    if hsi is None and hsi_stale is not None:
        stale_inputs.append(_stale_note("Hang Seng Index", hsi_stale))

    style_spread = hstech - hscei if hstech is not None and hscei is not None else None
    beta_spread = hstech - hsi if hstech is not None and hsi is not None else None

    if style_spread is not None:
        if style_spread > 0.5:
            leadership = "Hong Kong growth / internet led"
            style = "growth"
        elif style_spread < -0.5:
            leadership = "State-owned / old-economy H-shares led"
            style = "value"
        else:
            leadership = "Leadership was broad and balanced"
            style = "balanced"
    else:
        leadership = "Leadership could not be determined cleanly"
        style = "unconfirmed"

    participation_flags: List[str] = []
    confirmation_flags: List[str] = []
    if turnover_ratio is not None:
        if turnover_ratio < 0.9:
            participation_flags.append(f"turnover was only {turnover_ratio:.2f}x its 20-day average")
        elif turnover_ratio >= 1.05:
            confirmation_flags.append(f"turnover reached {turnover_ratio:.2f}x its 20-day average")
    if short_ratio is not None:
        # A level alone cannot say "elevated": Hong Kong market short-selling
        # normally runs in the mid-to-high teens. Prefer the trailing
        # distribution, and be explicit when there is not enough history for one.
        short_context = percentile_context(
            metric_history or {}, "short_selling_ratio", short_ratio, report_date or ""
        )
        detail = describe(short_context)
        if short_context.get("available"):
            band = short_context.get("band")
            if band in {"high", "very high"}:
                participation_flags.append(f"short selling was {band} at {short_ratio:.1f}% ({detail})")
            elif band in {"low", "very low"}:
                confirmation_flags.append(f"short selling was {band} at {short_ratio:.1f}% ({detail})")
        elif short_ratio >= 16.0:
            participation_flags.append(
                f"short selling was {short_ratio:.1f}%, above the 16% absolute reference ({detail})"
                if detail
                else f"short selling was {short_ratio:.1f}%, above the 16% absolute reference"
            )
        elif short_ratio <= 12.0:
            confirmation_flags.append(
                f"short selling was {short_ratio:.1f}%, below the 12% absolute reference ({detail})"
                if detail
                else f"short selling was {short_ratio:.1f}%, below the 12% absolute reference"
            )
    if southbound is not None:
        if southbound > 0:
            confirmation_flags.append(f"Southbound recorded {_compact_hkd_flow(southbound)} net buying")
        elif southbound < 0:
            participation_flags.append(f"Southbound recorded {_compact_hkd_flow(southbound)} net selling")
    if fxi is not None and hsi is not None:
        if fxi < hsi - 0.5:
            participation_flags.append(f"FXI ({_format_signed(fxi)}) lagged HSI ({_format_signed(hsi)})")
        elif fxi > hsi + 0.5:
            confirmation_flags.append(f"FXI ({_format_signed(fxi)}) outperformed HSI ({_format_signed(hsi)})")
    if usdcnh is not None:
        if usdcnh > 0.2:
            participation_flags.append(f"CNH weakened ({_format_signed(usdcnh)} in USD/CNH)")
        elif usdcnh < -0.2:
            confirmation_flags.append(f"CNH strengthened ({_format_signed(usdcnh)} in USD/CNH)")

    if style == "growth":
        headline = (
            "Selective growth leadership"
            if participation_flags
            else "Growth leadership with improving confirmation"
            if confirmation_flags
            else "Price-led growth leadership; confirmation incomplete"
        )
        relative_evidence = (
            f"3033.HK moved {_format_signed(hstech)} and beat HSCEI by {style_spread:+.2f}pp"
            + (f" and HSI by {beta_spread:+.2f}pp" if beta_spread is not None else "")
        )
        implication = "Favor relative strength in internet/platform names, but do not treat the move as broad China risk-on until participation confirms."
        confirmation = "Confirm if 3033.HK keeps outperforming HSCEI with at least normal turnover, broader Southbound buying, stable CNH, and easing short pressure."
        invalidation = "Invalidate if 3033.HK loses its lead to HSCEI, or if weak turnover, rising shorts, and CNH depreciation persist together."
    elif style == "value":
        headline = "Old-economy / H-share leadership" if not participation_flags else "Selective old-economy / H-share leadership"
        relative_evidence = f"HSCEI beat 3033.HK by {abs(style_spread):.2f}pp ({_format_signed(hscei)} versus {_format_signed(hstech)})"
        implication = "Treat banks, energy, telecoms, and SOE yield as the cleaner relative expression; growth beta still needs proof."
        confirmation = "Confirm if HSCEI keeps outperforming 3033.HK with healthy turnover and broad Southbound participation."
        invalidation = "Invalidate if 3033.HK retakes leadership while CNH firms and local participation broadens."
    elif style == "balanced":
        headline = "Broad beta, with no decisive style winner"
        relative_evidence = f"3033.HK and HSCEI were separated by only {abs(style_spread):.2f}pp ({_format_signed(hstech)} versus {_format_signed(hscei)})"
        implication = "Avoid forcing a growth-versus-value call; let volume, Southbound concentration, and the first-hour relative tape identify leadership."
        confirmation = "Confirm a broad-beta session only if HSI breadth and turnover improve without a sharp 3033.HK-versus-HSCEI split."
        invalidation = "Reclassify the tape when either 3033.HK or HSCEI opens a sustained 0.5pp relative lead with flow confirmation."
    else:
        if stale_inputs:
            headline = "Leadership unconfirmed — a required input is stale"
            relative_evidence = (
                f"No same-date 3033.HK-versus-HSCEI comparison was possible because {'; '.join(stale_inputs)}"
            )
        else:
            headline = "Leadership unconfirmed — coverage is insufficient"
            relative_evidence = "A comparable 3033.HK-versus-HSCEI move was not available"
        implication = "Do not infer Hong Kong style from the headline index alone; wait for relative price and local-flow evidence."
        confirmation = "Confirm only after HSI, HSCEI, 3033.HK, turnover, and Southbound evidence refresh on a comparable date."
        invalidation = "Any style claim remains invalid while the required relative-performance fields are missing or stale."

    lens = f"{headline}: {relative_evidence}."
    if participation_flags:
        lens += f" Conviction is limited because {'; '.join(participation_flags[:3])}."
        if confirmation_flags:
            lens += f" Partial support came from {confirmation_flags[0]}."
    elif confirmation_flags:
        lens += f" Conviction is improving because {'; '.join(confirmation_flags[:2])}."
    lens += f" {implication}"

    def _with_stale(value: Optional[float], stale_days: Optional[int]) -> str:
        if value is None and stale_days is not None:
            suffix = "d" if stale_days != 1 else "d"
            return f"stale {stale_days}{suffix}"
        return _format_signed(value)

    lines = [
        f"Hang Seng {_with_stale(hsi, hsi_stale)} / HSCEI {_with_stale(hscei, hscei_stale)}"
        f" / 3033.HK ETF {_with_stale(hstech, hstech_stale)}.",
        f"Offshore China proxy FXI {_format_signed(fxi)} and USD/CNH {_format_signed(usdcnh)} frame cross-border risk appetite.",
        f"USD/HKD last traded around {_fmt_price_for_hk(usdhkd)}, which keeps the Hong Kong funding lens in focus.",
    ]

    return {
        "leadership": leadership,
        "headline": headline,
        "lens": lens,
        "evidence": relative_evidence,
        "implication": implication,
        "confirmation": confirmation,
        "invalidation": invalidation,
        "style": style,
        "style_spread_pp": style_spread,
        "stale_inputs": stale_inputs,
        "participation_flags": participation_flags,
        "confirmation_flags": confirmation_flags,
        "lines": lines,
    }


def _fmt_price_for_hk(value: Any) -> str:
    price = _parse_float(value)
    if price is None:
        return "N/A"
    return f"{price:.4f}"
