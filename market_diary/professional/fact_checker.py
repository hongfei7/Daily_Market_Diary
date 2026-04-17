"""Deterministic fact checks for LLM-generated report sections."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


PCT_TOLERANCE = 0.25
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


def _fact_registry(bundle: Dict[str, Any]) -> List[Dict[str, Any]]:
    facts = [
        ("S&P 500", ["S&P 500", "SPX"], _parse_pct(_summary_item(bundle, "Equities", "S&P 500").get("Pct Change")), "pct"),
        ("Nasdaq 100", ["Nasdaq", "Nasdaq 100", "NDX"], _parse_pct(_summary_item(bundle, "Equities", "Nasdaq 100").get("Pct Change")), "pct"),
        ("Hang Seng Index", ["Hang Seng", "HSI", "Hang Seng Index"], _parse_pct(_summary_item(bundle, "Equities", "Hang Seng Index").get("Pct Change")), "pct"),
        ("Hang Seng TECH", ["HSTECH", "Hang Seng TECH"], _parse_pct(_summary_item(bundle, "Equities", "Hang Seng TECH ETF").get("Pct Change")), "pct"),
        ("FXI", ["FXI"], _parse_pct(_summary_item(bundle, "Equities", "China Large-Cap (FXI)").get("Pct Change")), "pct"),
        ("DXY", ["DXY", "dollar index"], _parse_pct(_summary_item(bundle, "FX", "DXY").get("Pct Change")), "pct"),
        ("USD/CNH", ["USD/CNH", "USDCNH"], _parse_pct(_summary_item(bundle, "FX", "USD/CNH").get("Pct Change")), "pct"),
        ("US 10Y", ["US 10Y", "10Y Treasury", "Treasury yield"], _parse_pct(_summary_item(bundle, "Rates", "10Y Treasury").get("Price")), "level_pct"),
        ("Brent crude", ["Brent", "Brent crude"], _parse_pct(_summary_item(bundle, "Commodities", "Brent Crude").get("Pct Change")), "pct"),
        ("Gold", ["Gold"], _parse_pct(_summary_item(bundle, "Commodities", "Gold").get("Pct Change")), "pct"),
        ("VIX", ["VIX"], _parse_pct(_summary_item(bundle, "Vol", "VIX").get("Pct Change")), "pct"),
        ("Short-selling ratio", ["short-selling ratio", "short selling ratio"], _metric_value(bundle, "hk_local", "short_selling_ratio"), "level_pct"),
        ("A/H premium", ["AH premium", "A/H premium"], _metric_value(bundle, "hk_local", "ah_premium_index"), "level_pct"),
    ]
    return [
        {"label": label, "aliases": aliases, "value": value, "kind": kind}
        for label, aliases, value, kind in facts
        if value is not None
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


def _claim_patterns(alias: str) -> List[re.Pattern[str]]:
    escaped = re.escape(alias)
    number = r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*%"
    verb = r"(?:was|were|rose|fell|gained|lost|up|down|added|shed|closed|finished|ended|changed|moved|at)"
    return [
        re.compile(rf"\b{escaped}\b[^\n.%;]{{0,{CLAIM_GAP}}}?\b{verb}\b[^\n.%;]{{0,{CLAIM_GAP}}}?{number}", re.IGNORECASE),
    ]


def _claim_mismatches(bundle: Dict[str, Any], texts: List[Tuple[str, str]]) -> Tuple[int, List[Dict[str, Any]]]:
    checked = 0
    mismatches: List[Dict[str, Any]] = []
    seen = set()
    for fact in _fact_registry(bundle):
        expected = float(fact["value"])
        tolerance = max(PCT_TOLERANCE, abs(expected) * 0.10)
        for path, text in texts:
            for alias in fact["aliases"]:
                for pattern in _claim_patterns(alias):
                    for match in pattern.finditer(text):
                        claimed = _parse_pct(match.group("value"))
                        if claimed is None:
                            continue
                        checked += 1
                        if abs(claimed - expected) > tolerance:
                            snippet = text[max(match.start() - 32, 0) : min(match.end() + 32, len(text))]
                            dedupe_key = (fact["label"], round(claimed, 3), round(expected, 3), snippet.strip())
                            if dedupe_key in seen:
                                continue
                            seen.add(dedupe_key)
                            mismatches.append(
                                {
                                    "field": path,
                                    "label": fact["label"],
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


def _logic_warnings(bundle: Dict[str, Any], full_text: str) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    text = full_text.lower()
    if not text:
        return warnings

    risk_regime = str((bundle.get("overview", {}) or {}).get("risk_regime", "")).lower()
    if "risk-on" in risk_regime and "risk-off" in text:
        warnings.append({"type": "risk_regime", "message": "Narrative mentions risk-off while the deterministic overview is risk-on."})
    if "risk-off" in risk_regime and "risk-on" in text:
        warnings.append({"type": "risk_regime", "message": "Narrative mentions risk-on while the deterministic overview is risk-off."})

    dxy = _parse_pct(_summary_item(bundle, "FX", "DXY").get("Pct Change"))
    if dxy is not None:
        if dxy > 0.30 and _contains_any(text, ["softer dollar", "weaker dollar", "dollar softened"]):
            warnings.append({"type": "fx_logic", "message": "Narrative says the dollar softened, but DXY was materially higher."})
        if dxy < -0.30 and _contains_any(text, ["stronger dollar", "firmer dollar", "dollar strengthened"]):
            warnings.append({"type": "fx_logic", "message": "Narrative says the dollar strengthened, but DXY was materially lower."})

    us10y = _parse_pct(_summary_item(bundle, "Rates", "10Y Treasury").get("Pct Change"))
    if us10y is not None:
        if us10y > 0.50 and _contains_any(text, ["lower yields", "yields fell", "yields declined"]):
            warnings.append({"type": "rates_logic", "message": "Narrative says yields fell, but US 10Y was materially higher."})
        if us10y < -0.50 and _contains_any(text, ["higher yields", "yields rose", "yields climbed"]):
            warnings.append({"type": "rates_logic", "message": "Narrative says yields rose, but US 10Y was materially lower."})

    southbound = ((bundle.get("hk_local", {}) or {}).get("southbound_net_flow", {}) or {})
    if southbound.get("status") == "unavailable" and _contains_any(text, ["southbound net buy", "southbound net inflow", "southbound bought"]):
        warnings.append({"type": "flow_availability", "message": "Narrative discusses Southbound net buying although the normalized metric is unavailable."})

    return warnings


def run_fact_check(bundle: Dict[str, Any]) -> Dict[str, Any]:
    llm_sections = bundle.get("llm_sections", {}) or {}
    texts = list(_iter_texts(llm_sections))
    if not texts:
        return {
            "status": "skipped",
            "summary": "No LLM-generated text was available for validation.",
            "numeric_claims_checked": 0,
            "numeric_mismatches": [],
            "logic_warnings": [],
        }

    checked, mismatches = _claim_mismatches(bundle, texts)
    full_text = "\n".join(text for _, text in texts)
    logic_warnings = _logic_warnings(bundle, full_text)
    status = "warning" if mismatches or logic_warnings else "ok"
    summary = (
        f"Checked {checked} numeric claims; "
        f"{len(mismatches)} numeric mismatch(es), {len(logic_warnings)} logic warning(s)."
    )
    return {
        "status": status,
        "summary": summary,
        "numeric_claims_checked": checked,
        "numeric_mismatches": mismatches[:12],
        "logic_warnings": logic_warnings[:12],
    }
