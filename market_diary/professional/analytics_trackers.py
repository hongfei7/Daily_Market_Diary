from __future__ import annotations

from typing import Any, Dict, List, Optional

from market_diary.professional.analytics_market import _parse_float, _parse_pct, _summary_item
from market_diary.professional.instruments import format_summary_change, summary_change


def _tracker_interpretation(label: str, change_value: Optional[float], chart_features: Dict[str, Any]) -> str:
    value = change_value or 0.0
    if label == "DXY":
        if value > 0.3:
            return "A stronger dollar points to a more defensive or rate-differential driven tape."
        if value < -0.3:
            return "A softer dollar makes it easier for risk appetite and duration to extend."
    if label == "US 10Y":
        if value > 5.0:
            return "Higher yields can pressure long-duration and growth valuations."
        if value < -5.0:
            return "Lower yields tend to support growth style and gold."
    if label == "WTI crude":
        if value > 1.0:
            return "A strong crude move needs to be split between geopolitics and demand repair."
        if value < -1.0:
            return "Softer oil argues for a more cautious cyclical-growth read."
    if label == "Gold":
        fx_net = chart_features.get("fx_composite", {}).get("net_pp")
        if value > 0.5 and (fx_net or 0) > 0:
            return "Gold rising with the dollar looks more like geopolitical or pure hedge demand."
        if value > 0.5:
            return "A firm gold price suggests hedge demand or lower real yields."
    if label == "Copper" and value < -0.8:
        return "Weak copper argues for more caution on cyclical growth."
    if label == "Bitcoin" and value > 1.5:
        return "A strong crypto tape reinforces the risk-on read."
    if label == "VIX":
        if value > 2.0:
            return "Higher volatility argues for tighter sizing and tighter stops."
        if value < -2.0:
            return "Lower volatility signals easing stress in the tape."
    return "Keep tracking it to confirm whether the core daily narrative is holding."


# How strongly a move in each instrument transmits to Hong Kong equities, for a
# desk covering AI / TMT. Ranking on raw magnitude alone put Bitcoin +7.48% and
# Gold +4.92% at the top of the morning checklist on 2026-08-20, ahead of
# everything that actually bears on Hong Kong tech.
HK_TRANSMISSION_WEIGHTS: Dict[str, float] = {
    # Direct read-through to the Hong Kong tech complex.
    "Nasdaq 100": 1.0,
    "SOXX": 1.0,
    "TSMC": 1.0,
    "NVDA": 1.0,
    "3033.HK ETF": 1.0,
    "USD/CNH": 1.0,
    # Broad beta and the rates channel through the peg.
    "S&P 500": 0.7,
    "US 10Y": 0.7,
    # Macro colour: real but second-order for a TMT desk.
    "DXY": 0.4,
    "Copper": 0.4,
    "WTI crude": 0.4,
    "VIX": 0.6,
    # Weakly connected to Hong Kong tech; kept for cross-asset context only.
    "Gold": 0.15,
    "Bitcoin": 0.15,
}
DEFAULT_TRANSMISSION_WEIGHT = 0.5


def transmission_weight(label: str, overrides: Optional[Dict[str, Any]] = None) -> float:
    """Weight a move by how strongly it reaches Hong Kong equities."""
    if overrides:
        try:
            return float(overrides[label])
        except (KeyError, TypeError, ValueError):
            pass
    return HK_TRANSMISSION_WEIGHTS.get(label, DEFAULT_TRANSMISSION_WEIGHT)


def _relevance_note(label: str, weight: float) -> str:
    """Say why an item earned its place, so the ranking is auditable."""
    if weight >= 1.0:
        return "direct read-through to HK tech"
    if weight >= 0.7:
        return "broad beta / rates channel"
    if weight >= 0.4:
        return "second-order macro context"
    return "weak HK linkage; context only"


def build_high_frequency_trackers(
    summary: Dict[str, Any],
    chart_features: Dict[str, Any],
    weight_overrides: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    tracked = [
        ("Rates", "10Y Treasury", "US 10Y"),
        ("FX", "DXY", "DXY"),
        ("FX", "USD/CNH", "USD/CNH"),
        ("Commodities", "Crude Oil", "WTI crude"),
        ("Commodities", "Gold", "Gold"),
        ("Commodities", "Copper", "Copper"),
        ("Crypto", "Bitcoin", "Bitcoin"),
        ("Vol", "VIX", "VIX"),
        ("Equities", "Semiconductors (SOXX)", "SOXX"),
        ("Equities", "TSMC ADR", "TSMC"),
        ("Equities", "NVIDIA", "NVDA"),
    ]
    rows: List[Dict[str, Any]] = []
    for category, name, label in tracked:
        item = _summary_item(summary, category, name)
        if not item:
            continue
        change_value, change_unit = summary_change(item)
        magnitude = abs(change_value or 0.0) / (10.0 if change_unit == "bp" else 1.0)
        weight = transmission_weight(label, weight_overrides)
        rows.append(
            {
                "label": label,
                "category": category,
                "symbol": name,
                "price": _parse_float(item.get("Price")),
                "change_value": change_value,
                "change_unit": change_unit,
                "change_display": format_summary_change(item),
                "change_pct": change_value if change_unit == "pct" else None,
                "interpretation": _tracker_interpretation(label, change_value, chart_features),
                "priority": magnitude * weight,
                "raw_magnitude": magnitude,
                "hk_weight": weight,
                "relevance": _relevance_note(label, weight),
            }
        )
    rows.sort(key=lambda row: row.get("priority", 0), reverse=True)
    return rows
