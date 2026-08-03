"""Rule-based cross-asset attribution for the Hong Kong morning workbench."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from market_diary.professional.instruments import summary_change


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
        cleaned = value.replace("%", "").replace(",", "").replace("x", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _summary_item(summary: Dict[str, Any], category: str, name: str) -> Dict[str, Any]:
    item = (summary or {}).get(category, {}).get(name, {})
    return item if isinstance(item, dict) else {}


def _summary_pct(summary: Dict[str, Any], category: str, name: str) -> Optional[float]:
    return _parse_pct(_summary_item(summary, category, name).get("Pct Change"))


def _metric_value(metrics: Dict[str, Any], key: str) -> Optional[float]:
    item = metrics.get(key, {}) if isinstance(metrics, dict) else {}
    if not isinstance(item, dict):
        return None
    return _parse_float(item.get("value"))


def _add_driver(
    drivers: List[Dict[str, Any]],
    *,
    name: str,
    direction: str,
    score: float,
    evidence: str,
    implication: str,
) -> None:
    if score <= 0:
        return
    drivers.append(
        {
            "name": name,
            "direction": direction,
            "score": round(score, 2),
            "evidence": evidence,
            "implication": implication,
        }
    )


def _risk_bucket(score: float) -> str:
    if score >= 65:
        return "Risk-on"
    if score <= 35:
        return "Risk-off"
    return "Mixed"


def _short_sell_market(movers_digest: Dict[str, Any], hk_local: Dict[str, Any]) -> Optional[float]:
    short_sell = (movers_digest.get("short_sell", {}) or {}).get("data", {}) or {}
    market = short_sell.get("market", {}) or {}
    if market.get("short_ratio_pct") is not None:
        return _parse_float(market.get("short_ratio_pct"))
    return _metric_value(hk_local, "short_selling_ratio")


def build_attribution(
    summary: Dict[str, Any],
    hk_local: Dict[str, Any],
    movers_digest: Dict[str, Any],
    overview: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a first-version rule-based attribution context.

    This is intentionally transparent.  It does not claim statistical causality;
    it ranks plausible drivers using observable cross-asset and local-market
    signals, which is safer for a daily research note than premature regression.
    """

    drivers: List[Dict[str, Any]] = []

    spx = _summary_pct(summary, "Equities", "S&P 500")
    ndx = _summary_pct(summary, "Equities", "Nasdaq 100")
    hsi = _summary_pct(summary, "Equities", "Hang Seng Index")
    hscei = _summary_pct(summary, "Equities", "Hang Seng China Enterprises")
    hstech = _summary_pct(summary, "Equities", "Hang Seng TECH ETF")
    fxi = _summary_pct(summary, "Equities", "China Large-Cap (FXI)")
    dxy = _summary_pct(summary, "FX", "DXY")
    usdcnh = _summary_pct(summary, "FX", "USD/CNH")
    us10y, us10y_unit = summary_change(_summary_item(summary, "Rates", "10Y Treasury"))
    vix = _summary_pct(summary, "Vol", "VIX")
    brent = _summary_pct(summary, "Commodities", "Brent Crude")
    wti = _summary_pct(summary, "Commodities", "Crude Oil")
    turnover_ratio = _metric_value(hk_local, "turnover_vs_20d")
    short_ratio = _short_sell_market(movers_digest, hk_local)

    if ndx is not None and abs(ndx) >= 0.8:
        _add_driver(
            drivers,
            name="US growth-style transmission",
            direction="supportive" if ndx > 0 else "drag",
            score=abs(ndx) * 1.4,
            evidence=f"Nasdaq 100 moved {ndx:+.2f}%.",
            implication="Hong Kong growth and platform names should be tested first at the open.",
        )

    if spx is not None and abs(spx) >= 0.7:
        _add_driver(
            drivers,
            name="Global equity beta",
            direction="supportive" if spx > 0 else "drag",
            score=abs(spx),
            evidence=f"S&P 500 moved {spx:+.2f}%.",
            implication="Broad beta matters if Hong Kong breadth confirms the overseas signal.",
        )

    if us10y is not None and us10y_unit == "bp" and abs(us10y) >= 5.0:
        _add_driver(
            drivers,
            name="Rates impulse",
            direction="supportive" if us10y < 0 else "drag",
            score=abs(us10y) * 0.15,
            evidence=f"US 10Y yield moved {us10y:+.1f}bp.",
            implication="Lower yields help long-duration growth; higher yields pressure valuation-sensitive sectors.",
        )

    fx_pressure = max(abs(dxy or 0), abs(usdcnh or 0))
    if fx_pressure >= 0.25:
        direction = "drag" if (dxy or 0) > 0.25 or (usdcnh or 0) > 0.25 else "supportive"
        _add_driver(
            drivers,
            name="Dollar / CNH pressure",
            direction=direction,
            score=fx_pressure * 1.5,
            evidence=f"DXY {dxy:+.2f}% / USD-CNH {usdcnh:+.2f}%" if dxy is not None and usdcnh is not None else "FX pressure moved materially.",
            implication="A stable or softer CNH makes Hong Kong follow-through more credible.",
        )

    oil_move = brent if brent is not None else wti
    if oil_move is not None and abs(oil_move) >= 1.5:
        _add_driver(
            drivers,
            name="Oil and geopolitics",
            direction="mixed" if oil_move > 0 else "supportive",
            score=abs(oil_move) * 0.8,
            evidence=f"Oil moved {oil_move:+.2f}%.",
            implication="Split the read between energy/cyclicals support and margin/geopolitical pressure.",
        )

    if turnover_ratio is not None:
        if turnover_ratio >= 1.10:
            _add_driver(
                drivers,
                name="Hong Kong participation",
                direction="supportive",
                score=(turnover_ratio - 1.0) * 10,
                evidence=f"Main Board turnover was {turnover_ratio:.2f}x the 20-session average.",
                implication="Higher participation makes local price action more credible.",
            )
        elif turnover_ratio <= 0.90:
            _add_driver(
                drivers,
                name="Hong Kong participation",
                direction="drag",
                score=(1.0 - turnover_ratio) * 10,
                evidence=f"Main Board turnover was {turnover_ratio:.2f}x the 20-session average.",
                implication="Thin turnover makes index moves easier to fade.",
            )

    if short_ratio is not None:
        if short_ratio >= 18.0:
            _add_driver(
                drivers,
                name="Short-selling pressure",
                direction="drag",
                score=(short_ratio - 15.0) * 0.7,
                evidence=f"HKEX short-selling ratio was {short_ratio:.2f}%.",
                implication="Elevated short activity argues for extra confirmation before chasing rebounds.",
            )
        elif short_ratio <= 12.0:
            _add_driver(
                drivers,
                name="Short-selling pressure",
                direction="supportive",
                score=(12.0 - short_ratio) * 0.5,
                evidence=f"HKEX short-selling ratio was {short_ratio:.2f}%.",
                implication="Lower short pressure gives risk appetite more room to follow through.",
            )

    drivers.sort(key=lambda item: item.get("score", 0), reverse=True)

    risk_score = 50.0
    components: List[Dict[str, Any]] = []

    def add_component(label: str, delta: float, evidence: str) -> None:
        components.append({"label": label, "delta": round(delta, 1), "evidence": evidence})

    if spx is not None:
        delta = max(min(spx * 6, 10), -10)
        risk_score += delta
        add_component("US beta", delta, f"S&P 500 {spx:+.2f}%")
    if hstech is not None:
        delta = max(min(hstech * 5, 10), -10)
        risk_score += delta
        add_component("HK growth ETF proxy", delta, f"3033.HK ETF {hstech:+.2f}%")
    if fxi is not None:
        delta = max(min(fxi * 5, 8), -8)
        risk_score += delta
        add_component("Offshore China proxy", delta, f"FXI {fxi:+.2f}%")
    if dxy is not None:
        delta = max(min(-dxy * 10, 8), -8)
        risk_score += delta
        add_component("Dollar pressure", delta, f"DXY {dxy:+.2f}%")
    if vix is not None:
        delta = max(min(-vix * 2, 10), -10)
        risk_score += delta
        add_component("Volatility", delta, f"VIX {vix:+.2f}%")
    if short_ratio is not None:
        delta = -8 if short_ratio >= 18 else 4 if short_ratio <= 12 else 0
        risk_score += delta
        add_component("HK short-selling", delta, f"Short-selling ratio {short_ratio:.2f}%")
    if turnover_ratio is not None:
        delta = 5 if turnover_ratio >= 1.10 else -5 if turnover_ratio <= 0.90 else 0
        risk_score += delta
        add_component("HK turnover", delta, f"Turnover {turnover_ratio:.2f}x 20D")

    risk_score = max(0.0, min(100.0, risk_score))

    flow_summary = "Local flow evidence is not yet strong enough to override the cross-asset setup."
    if short_ratio is not None and turnover_ratio is not None:
        if turnover_ratio >= 1.10 and short_ratio < 18.0:
            flow_summary = "Hong Kong participation is active and short pressure is not excessive, so local confirmation looks healthier."
        elif short_ratio >= 18.0:
            flow_summary = "Short-selling pressure is elevated, so local rebounds need cleaner breadth and flow confirmation."
        elif turnover_ratio <= 0.90:
            flow_summary = "Turnover is light versus the 20-session baseline, so local moves have weaker conviction."

    return {
        "status": "ok",
        "market_read": overview.get("theme", ""),
        "dominant_drivers": drivers[:6],
        "flow_summary": flow_summary,
        "risk_dashboard": {
            "score": round(risk_score, 1),
            "bucket": _risk_bucket(risk_score),
            "components": components[:8],
        },
        "style_snapshot": {
            "hsi": hsi,
            "hscei": hscei,
            "hstech": hstech,
            "fxi": fxi,
        },
    }
