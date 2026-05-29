"""Deterministic fact checks for narrative report sections."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


CHANGE_PCT_TOLERANCE = 0.25
LEVEL_PCT_TOLERANCE = 0.05
BP_TOLERANCE = 3.0
CLAIM_GAP = 12


def _parse_pct(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace("%", "").replace(",", "").strip()
    if not cleaned or cleaned.upper() in {"N/A", "NO DATA"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_bp(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).lower().replace(",", "").replace("bps", "").replace("bp", "").strip()
    if not cleaned or cleaned.upper() in {"N/A", "NO DATA"}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _summary_item(bundle: Dict[str, Any], category: str, name: str) -> Dict[str, Any]:
    summary = bundle.get("market_summary", {}) or {}
    item = (summary.get(category, {}) or {}).get(name, {})
    return item if isinstance(item, dict) else {}


def _metric_value(bundle: Dict[str, Any], section: str, key: str) -> Optional[float]:
    section_data = bundle.get(section, {}) or {}
    item = section_data.get(key, {}) if isinstance(section_data, dict) else {}
    if not isinstance(item, dict):
        return None
    return _parse_pct(item.get("value") if item.get("value") is not None else item.get("display_value"))


def _market_fact(bundle: Dict[str, Any], category: str, name: str, label: str, aliases: List[str]) -> Dict[str, Any]:
    item = _summary_item(bundle, category, name)
    return {
        "label": label,
        "aliases": aliases,
        "change_pct": _parse_pct(item.get("Pct Change")),
        "level_pct": _parse_pct(item.get("Price")) if category in {"Rates"} else None,
        "level": _parse_pct(item.get("Price")) if category in {"Vol"} else None,
    }


def _china_rate_fact(bundle: Dict[str, Any], key: str, label: str, aliases: List[str]) -> Dict[str, Any]:
    metric = ((bundle.get("china_rates", {}) or {}).get(key, {}) or {})
    return {
        "label": label,
        "aliases": aliases,
        "level_pct": _parse_pct(metric.get("value") if metric.get("value") is not None else metric.get("display_value")),
        "change_bp": _parse_bp(metric.get("change_display")),
    }


def _fact_registry(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts = [
        _market_fact(bundle, "Equities", "S&P 500", "S&P 500", ["S&P 500", "SPX"]),
        _market_fact(bundle, "Equities", "Nasdaq 100", "Nasdaq 100", ["Nasdaq", "Nasdaq 100", "NDX"]),
        _market_fact(bundle, "Equities", "Hang Seng Index", "Hang Seng Index", ["Hang Seng", "HSI", "Hang Seng Index"]),
        _market_fact(bundle, "Equities", "Hang Seng TECH ETF", "Hang Seng TECH", ["HSTECH", "Hang Seng TECH"]),
        _market_fact(bundle, "Equities", "China Large-Cap (FXI)", "FXI", ["FXI"]),
        _market_fact(bundle, "FX", "DXY", "DXY", ["DXY", "dollar index"]),
        _market_fact(bundle, "FX", "USD/CNH", "USD/CNH", ["USD/CNH", "USDCNH"]),
        _market_fact(bundle, "Rates", "10Y Treasury", "US 10Y", ["US 10Y", "10Y Treasury", "Treasury yield"]),
        _market_fact(bundle, "Commodities", "Brent Crude", "Brent crude", ["Brent", "Brent crude"]),
        _market_fact(bundle, "Commodities", "Gold", "Gold", ["Gold"]),
        _market_fact(bundle, "Vol", "VIX", "VIX", ["VIX"]),
        {
            "label": "Short-selling ratio",
            "aliases": ["short-selling ratio", "short selling ratio"],
            "level_pct": _metric_value(bundle, "hk_local", "short_selling_ratio"),
        },
        {
            "label": "A/H premium",
            "aliases": ["AH premium", "A/H premium"],
            "level_pct": _metric_value(bundle, "hk_local", "ah_premium_index"),
        },
        _china_rate_fact(bundle, "china_10y", "China 10Y", ["China 10Y", "China government bond yield"]),
        _china_rate_fact(bundle, "cn_us_10y_spread", "CN-US 10Y spread", ["CN-US 10Y spread", "China-US 10Y spread"]),
    ]
    return [
        fact
        for fact in facts
        if any(fact.get(key) is not None for key in ("change_pct", "level_pct", "change_bp", "level"))
    ]


def _iter_texts(value: Any, path: str = "") -> Iterable[Tuple[str, str]]:
    if isinstance(value, str) and value.strip():
        yield path, value.strip()
    elif isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _iter_texts(child, child_path)
    elif isinstance(value, list):
        for idx, child in enumerate(value):
            child_path = f"{path}[{idx}]"
            yield from _iter_texts(child, child_path)


def _claim_patterns(alias: str) -> List[Tuple[str, re.Pattern[str]]]:
    escaped = re.escape(alias)
    pct_number = r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*%"
    bp_number = r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?:bp|bps)"
    change_verb = r"(?:rose|fell|gained|lost|up|down|added|shed|changed|moved|increased|decreased|climbed|dropped|slipped)"
    level_verb = r"(?:at|to|around|near|last|closed at|finished at|ended at|yield at|traded at)"
    return [
        (
            "change_pct",
            re.compile(rf"\b{escaped}\b[^\n.%;]{{0,{CLAIM_GAP}}}?\b{change_verb}\b[^\n.;]{{0,{CLAIM_GAP}}}?{pct_number}", re.IGNORECASE),
        ),
        (
            "level_pct",
            re.compile(rf"\b{escaped}\b[^\n.;]{{0,{CLAIM_GAP}}}?\b{level_verb}\b[^\n.;]{{0,{CLAIM_GAP}}}?{pct_number}", re.IGNORECASE),
        ),
        (
            "change_bp",
            re.compile(rf"\b{escaped}\b[^\n.;]{{0,{CLAIM_GAP}}}?\b{change_verb}\b[^\n.;]{{0,{CLAIM_GAP}}}?{bp_number}", re.IGNORECASE),
        ),
    ]


def _claim_tolerance(kind: str, expected: float) -> float:
    if kind == "level_pct":
        return max(LEVEL_PCT_TOLERANCE, abs(expected) * 0.01)
    if kind == "change_bp":
        return BP_TOLERANCE
    return max(CHANGE_PCT_TOLERANCE, abs(expected) * 0.10)


def _severity_for_numeric(kind: str, claimed: float, expected: float, tolerance: float) -> str:
    error = abs(claimed - expected)
    if kind == "level_pct" and error <= max(tolerance * 2.0, 0.15):
        return "review"
    if kind == "change_bp" and error <= tolerance * 2.0:
        return "review"
    return "critical"


def _claim_mismatches(bundle: Dict[str, Any], texts: List[Tuple[str, str]]) -> Tuple[int, List[Dict[str, Any]]]:
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    seen = set()
    for fact in _fact_registry(bundle):
        for path, text in texts:
            for alias in fact["aliases"]:
                for kind, pattern in _claim_patterns(alias):
                    if fact.get(kind) is None:
                        continue
                    expected = float(fact[kind])
                    tolerance = _claim_tolerance(kind, expected)
                    for match in pattern.finditer(text):
                        claimed = _parse_pct(match.group("value"))
                        if claimed is None:
                            continue
                        checked += 1
                        if abs(claimed - expected) > tolerance:
                            snippet = text[max(match.start() - 50, 0) : min(match.end() + 50, len(text))]
                            severity = _severity_for_numeric(kind, claimed, expected, tolerance)
                            dedupe_key = (fact["label"], kind, round(claimed, 3), round(expected, 3), snippet.strip())
                            if dedupe_key in seen:
                                continue
                            seen.add(dedupe_key)
                            mismatches.append(
                                {
                                    "field": path,
                                    "label": fact["label"],
                                    "claim_type": kind,
                                    "severity": severity,
                                    "claimed": round(claimed, 3),
                                    "expected": round(expected, 3),
                                    "tolerance": round(tolerance, 3),
                                    "snippet": snippet,
                                }
                            )
    return checked, mismatches


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def _contains_unhedged_any(text: str, phrases: Iterable[str]) -> bool:
    """Return true only when a phrase is not part of a conditional watchpoint."""
    conditional_markers = [
        "could",
        "if ",
        "may ",
        "might",
        "monitor",
        "scenario",
        "watch",
        "watchpoint",
        "would",
    ]
    lowered = text.lower()
    for phrase in phrases:
        start = 0
        while True:
            idx = lowered.find(phrase, start)
            if idx == -1:
                break
            context = lowered[max(0, idx - 80) : min(len(lowered), idx + len(phrase) + 80)]
            if not any(marker in context for marker in conditional_markers):
                return True
            start = idx + len(phrase)
    return False


def _logic_warnings(bundle: Dict[str, Any], full_text: str) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    text = full_text.lower()
    if not text:
        return warnings

    risk_regime = str((bundle.get("overview", {}) or {}).get("risk_regime", "")).lower()
    risk_on_assertions = ["risk-on backdrop", "risk-on regime", "risk-on setup", "risk-on tape", "risk appetite improved", "risk appetite rose"]
    risk_off_assertions = ["risk-off backdrop", "risk-off regime", "risk-off setup", "risk-off tape", "risk appetite deteriorated", "risk appetite faded"]
    if "risk-on" in risk_regime and _contains_any(text, risk_off_assertions):
        warnings.append({"type": "risk_regime", "severity": "review", "message": "Narrative asserts a risk-off setup while the deterministic overview is risk-on."})
    if "risk-off" in risk_regime and _contains_any(text, risk_on_assertions):
        warnings.append({"type": "risk_regime", "severity": "review", "message": "Narrative asserts a risk-on setup while the deterministic overview is risk-off."})

    dxy = _parse_pct(_summary_item(bundle, "FX", "DXY").get("Pct Change"))
    if dxy is not None:
        if dxy > 0.30 and _contains_unhedged_any(text, ["softer dollar", "weaker dollar", "dollar softened"]):
            warnings.append({"type": "fx_logic", "severity": "review", "message": "Narrative says the dollar softened, but DXY was materially higher."})
        if dxy < -0.30 and _contains_unhedged_any(text, ["stronger dollar", "firmer dollar", "dollar strengthened"]):
            warnings.append({"type": "fx_logic", "severity": "review", "message": "Narrative says the dollar strengthened, but DXY was materially lower."})

    us10y = _parse_pct(_summary_item(bundle, "Rates", "10Y Treasury").get("Pct Change"))
    if us10y is not None:
        if us10y > 0.50 and _contains_unhedged_any(text, ["lower yields", "yields fell", "yields declined"]):
            warnings.append({"type": "rates_logic", "severity": "review", "message": "Narrative says yields fell, but US 10Y was materially higher."})
        if us10y < -0.50 and _contains_unhedged_any(text, ["higher yields", "yields rose", "yields climbed"]):
            warnings.append({"type": "rates_logic", "severity": "review", "message": "Narrative says yields rose, but US 10Y was materially lower."})

    southbound = ((bundle.get("hk_local", {}) or {}).get("southbound_net_flow", {}) or {})
    if southbound.get("status") == "unavailable" and _contains_any(text, ["southbound net buy", "southbound net inflow", "southbound bought"]):
        warnings.append({"type": "flow_availability", "severity": "review", "message": "Narrative discusses Southbound net buying although the normalized metric is unavailable."})

    return warnings


def run_fact_check(bundle: Dict[str, Any]) -> Dict[str, Any]:
    llm_sections = bundle.get("llm_sections", {}) or {}
    texts = list(_iter_texts(llm_sections))
    if not texts:
        return {
            "status": "skipped",
            "summary": "No narrative overlay text was available for validation.",
            "numeric_claims_checked": 0,
            "numeric_mismatches": [],
            "logic_warnings": [],
        }

    checked, mismatches = _claim_mismatches(bundle, texts)
    full_text = "\n".join(text for _, text in texts)
    logic_warnings = _logic_warnings(bundle, full_text)
    critical_count = sum(1 for item in mismatches if item.get("severity") == "critical")
    review_count = (
        sum(1 for item in mismatches if item.get("severity") == "review")
        + sum(1 for item in logic_warnings if item.get("severity", "review") == "review")
    )
    status = "warning" if critical_count or review_count else "ok"
    summary = (
        f"Checked {checked} numeric claims; "
        f"{len(mismatches)} numeric mismatch(es), {len(logic_warnings)} logic warning(s); "
        f"{critical_count} critical, {review_count} review."
    )
    return {
        "status": status,
        "summary": summary,
        "numeric_claims_checked": checked,
        "numeric_mismatches": mismatches[:12],
        "logic_warnings": logic_warnings[:12],
    }
