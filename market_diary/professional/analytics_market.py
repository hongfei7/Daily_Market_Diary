from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


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


def _snapshot_row(summary: Dict[str, Any], category: str, name: str, label: str, question: str) -> Dict[str, Any]:
    item = _summary_item(summary, category, name)
    price = _parse_float(item.get("Price"))
    change_pct = _parse_pct(item.get("Pct Change"))
    return {
        "label": label,
        "category": category,
        "symbol": name,
        "price": price,
        "change_pct": change_pct,
        "question": question,
    }


def build_market_snapshot(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracked = [
        ("Equities", "S&P 500", "S&P 500", "Risk appetite"),
        ("Equities", "Nasdaq 100", "Nasdaq 100", "Growth style"),
        ("Equities", "Euro Stoxx 50", "Euro Stoxx 50", "European risk appetite"),
        ("Equities", "Hang Seng Index", "Hang Seng Index", "Hong Kong beta"),
        ("Equities", "Hang Seng China Enterprises", "HSCEI", "China SOE / H-share tone"),
        ("Equities", "Hang Seng TECH ETF", "Hang Seng TECH", "Hong Kong growth / internet tone"),
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
        if row.get("label") == label:
            return row
    return {}


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
    hstech = _get_row(rows, "Hang Seng TECH").get("change_pct")
    fxi = _get_row(rows, "China proxy (FXI)").get("change_pct")
    dxy = _get_row(rows, "DXY").get("change_pct")
    us10y = _get_row(rows, "US 10Y").get("change_pct")
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
    if us10y is not None:
        score += -1 if us10y > 0.5 else 1 if us10y < -0.5 else 0

    if score >= 2:
        risk_regime = "Risk-On"
    elif score <= -2:
        risk_regime = "Risk-Off"
    else:
        risk_regime = "Neutral"

    chart_read = build_chart_read(chart_features)
    usd_net = (chart_features.get("fx_composite", {}) or {}).get("net_pp") or 0
    usd_bias = "USD stronger" if usd_net > 0.15 else "USD softer" if usd_net < -0.15 else "USD range-bound"
    rate_bias = "lower yields supported duration" if (us10y or 0) < -0.5 else "higher yields pressured valuations" if (us10y or 0) > 0.5 else "rates were not the dominant driver"
    asset_div = chart_features.get("divergence", {}) or {}
    divergence_text = f"{asset_div.get('best_asset')} outperformed {asset_div.get('worst_asset')}" if asset_div else "cross-asset divergence stayed modest"
    theme = f"{risk_regime} backdrop with {usd_bias}, {rate_bias}, and {divergence_text}"

    notes = [
        f"Risk appetite snapshot: S&P 500 {_format_signed(spx)} / Nasdaq 100 {_format_signed(ndx)} / Hang Seng {_format_signed(hsi)} / HSTECH {_format_signed(hstech)} / FXI {_format_signed(fxi)}.",
        f"Rates and liquidity: US 10Y {_format_signed(us10y)} / DXY {_format_signed(dxy)} / VIX {_format_signed(vix)}.",
        f"Commodities and hedges: WTI {_format_signed(oil)} / Gold {_format_signed(gold)}.",
    ]

    key_questions = [
        f"Is risk appetite rising or fading? The overnight tape reads closer to `{risk_regime}`.",
        "Did rates expectations move? Focus on whether US 10Y, DXY, and growth style moved together.",
        "Were commodities the real story? Watch WTI and gold for geopolitics versus inflation signals.",
        "Is Hong Kong setup internet-led, SOE-led, or broad beta-led? Compare HSTECH, HSCEI, and HSI.",
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


def build_hk_desk_view(summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = build_market_snapshot(summary)
    hsi = _get_row(rows, "Hang Seng Index").get("change_pct")
    hscei = _get_row(rows, "HSCEI").get("change_pct")
    hstech = _get_row(rows, "Hang Seng TECH").get("change_pct")
    fxi = _get_row(rows, "China proxy (FXI)").get("change_pct")
    usdcnh = _get_row(rows, "USD/CNH").get("change_pct")
    usdhkd = _get_row(rows, "USD/HKD").get("price")

    if hstech is not None and hscei is not None:
        if hstech - hscei > 0.5:
            leadership = "Hong Kong growth / internet led"
        elif hscei - hstech > 0.5:
            leadership = "State-owned / old-economy H-shares led"
        else:
            leadership = "Leadership was broad and balanced"
    else:
        leadership = "Leadership could not be determined cleanly"

    lines = [
        f"Hang Seng {_format_signed(hsi)} / HSCEI {_format_signed(hscei)} / HSTECH {_format_signed(hstech)}.",
        f"Offshore China proxy FXI {_format_signed(fxi)} and USD/CNH {_format_signed(usdcnh)} frame cross-border risk appetite.",
        f"USD/HKD last traded around {_fmt_price_for_hk(usdhkd)}, which keeps the Hong Kong funding lens in focus.",
    ]

    return {"leadership": leadership, "lines": lines}


def _fmt_price_for_hk(value: Any) -> str:
    price = _parse_float(value)
    if price is None:
        return "N/A"
    return f"{price:.4f}"
