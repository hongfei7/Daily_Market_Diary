from __future__ import annotations

from typing import Any, Dict, List, Optional

from market_diary.professional.analytics_market import _parse_float, _parse_pct, _summary_item


def _tracker_interpretation(label: str, change_pct: Optional[float], chart_features: Dict[str, Any]) -> str:
    value = change_pct or 0.0
    if label == "DXY":
        if value > 0.3:
            return "A stronger dollar points to a more defensive or rate-differential driven tape."
        if value < -0.3:
            return "A softer dollar makes it easier for risk appetite and duration to extend."
    if label == "US 10Y":
        if value > 0.5:
            return "Higher yields can pressure long-duration and growth valuations."
        if value < -0.5:
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


def build_high_frequency_trackers(summary: Dict[str, Any], chart_features: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracked = [
        ("Rates", "10Y Treasury", "US 10Y"),
        ("FX", "DXY", "DXY"),
        ("FX", "USD/CNH", "USD/CNH"),
        ("Commodities", "Crude Oil", "WTI crude"),
        ("Commodities", "Gold", "Gold"),
        ("Commodities", "Copper", "Copper"),
        ("Crypto", "Bitcoin", "Bitcoin"),
        ("Vol", "VIX", "VIX"),
    ]
    rows: List[Dict[str, Any]] = []
    for category, name, label in tracked:
        item = _summary_item(summary, category, name)
        if not item:
            continue
        change_pct = _parse_pct(item.get("Pct Change"))
        rows.append(
            {
                "label": label,
                "category": category,
                "symbol": name,
                "price": _parse_float(item.get("Price")),
                "change_pct": change_pct,
                "interpretation": _tracker_interpretation(label, change_pct, chart_features),
                "priority": abs(change_pct or 0.0),
            }
        )
    rows.sort(key=lambda row: row.get("priority", 0), reverse=True)
    return rows
