"""Deterministic fact checks for narrative report sections."""

from __future__ import annotations

from datetime import datetime
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

from market_diary.professional.instruments import summary_change


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
    change_value, change_unit = summary_change(item)
    return {
        "label": label,
        "aliases": aliases,
        "change_pct": change_value if change_unit == "pct" else None,
        "change_bp": change_value if change_unit == "bp" else None,
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
        _market_fact(
            bundle,
            "Equities",
            "Hang Seng TECH ETF",
            "Hang Seng TECH ETF (3033.HK)",
            ["3033.HK ETF", "HSTECH ETF", "Hang Seng TECH ETF", "HSTECH"],
        ),
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


# Prose states direction with a verb ("fell 1.68%") or a trailing noun
# ("0.69% decline") rather than a sign. Reading the magnitude without the
# direction turns correct English into a release-blocking false positive.
NEGATIVE_VERBS = r"fell|lost|shed|decreased|dropped|slipped|declined|down|sank|slid|retreated|weakened"
POSITIVE_VERBS = r"rose|gained|added|increased|climbed|advanced|up|jumped|rallied|firmed|strengthened"
NEUTRAL_VERBS = r"changed|moved"

NEGATIVE_NOUNS = r"decline|declines|drop|drops|fall|loss|losses|selloff|sell-off|slide|pullback|retreat|decrease"
POSITIVE_NOUNS = r"gain|gains|rise|advance|rally|increase|jump|climb"
_DIRECTION_NOUNS = f"{NEGATIVE_NOUNS}|{POSITIVE_NOUNS}"

_NEGATIVE_NOUN_RE = re.compile(rf"^\W{{0,3}}(?:\w+\s+){{0,2}}?(?:{NEGATIVE_NOUNS})\b", re.IGNORECASE)
_POSITIVE_NOUN_RE = re.compile(rf"^\W{{0,3}}(?:\w+\s+){{0,2}}?(?:{POSITIVE_NOUNS})\b", re.IGNORECASE)


def _claim_patterns(alias: str) -> List[Tuple[str, re.Pattern[str]]]:
    escaped = re.escape(alias)
    pct_number = r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*%"
    bp_number = r"(?P<value>[+-]?\d+(?:\.\d+)?)\s*(?:bp|bps)"
    change_verb = rf"(?P<verb>{NEGATIVE_VERBS}|{POSITIVE_VERBS}|{NEUTRAL_VERBS})"
    level_verb = r"(?:at|to|around|near|last|closed at|finished at|ended at|yield at|traded at)"
    return [
        (
            "change_pct",
            re.compile(rf"\b{escaped}\b\s*(?:[:|,/·]\s*)?{pct_number}", re.IGNORECASE),
        ),
        (
            "change_pct",
            re.compile(rf"\b{escaped}\b[^\n.%;]{{0,{CLAIM_GAP}}}?\b{change_verb}\b[^\n.;]{{0,{CLAIM_GAP}}}?{pct_number}", re.IGNORECASE),
        ),
        # "the S&P 500's 0.69% decline" — direction carried by a trailing noun,
        # which the verb patterns above cannot reach across the possessive.
        (
            "change_pct",
            re.compile(
                rf"\b{escaped}\b[^\n.;%]{{0,{CLAIM_GAP}}}?{pct_number}\s+(?:\w+\s+){{0,2}}?"
                rf"(?P<noun>{_DIRECTION_NOUNS})\b",
                re.IGNORECASE,
            ),
        ),
        (
            "level_pct",
            re.compile(rf"\b{escaped}\b[^\n.;]{{0,{CLAIM_GAP}}}?\b{level_verb}\b[^\n.;]{{0,{CLAIM_GAP}}}?{pct_number}", re.IGNORECASE),
        ),
        (
            "change_bp",
            re.compile(rf"\b{escaped}\b[^\n.;]{{0,{CLAIM_GAP}}}?\b{change_verb}\b[^\n.;]{{0,{CLAIM_GAP}}}?{bp_number}", re.IGNORECASE),
        ),
        (
            "change_bp",
            re.compile(rf"\b{escaped}\b\s*(?:[:|,/·]\s*)?{bp_number}", re.IGNORECASE),
        ),
    ]


def _signed_claim(match: re.Match[str], text: str) -> Optional[float]:
    """Resolve a claimed magnitude into a signed value using prose direction.

    An explicit sign in the text always wins. Otherwise the direction comes from
    the matched verb, then from a direction noun immediately after the number.
    """
    raw = match.group("value")
    value = _parse_pct(raw)
    if value is None:
        return None

    # An explicit sign is authoritative; never second-guess it.
    if raw.strip().startswith(("+", "-")):
        return value

    groups = match.groupdict()

    verb = (groups.get("verb") or "").strip().lower()
    if verb:
        if re.fullmatch(NEGATIVE_VERBS, verb, re.IGNORECASE):
            return -abs(value)
        if re.fullmatch(POSITIVE_VERBS, verb, re.IGNORECASE):
            return abs(value)

    noun = (groups.get("noun") or "").strip().lower()
    if noun:
        if re.fullmatch(NEGATIVE_NOUNS, noun, re.IGNORECASE):
            return -abs(value)
        if re.fullmatch(POSITIVE_NOUNS, noun, re.IGNORECASE):
            return abs(value)

    trailing = text[match.end() : match.end() + 40]
    if _NEGATIVE_NOUN_RE.match(trailing):
        return -abs(value)
    if _POSITIVE_NOUN_RE.match(trailing):
        return abs(value)
    return value


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
    checked_seen = set()
    for fact in _fact_registry(bundle):
        for path, text in texts:
            for alias in fact["aliases"]:
                for kind, pattern in _claim_patterns(alias):
                    if fact.get(kind) is None:
                        continue
                    expected = float(fact[kind])
                    tolerance = _claim_tolerance(kind, expected)
                    for match in pattern.finditer(text):
                        claimed = _signed_claim(match, text)
                        if claimed is None:
                            continue
                        # A level is a magnitude, not a direction; prose polarity
                        # must not flip it (e.g. "the yield fell to 4.70%").
                        if kind == "level_pct":
                            claimed = abs(claimed) if expected >= 0 else claimed
                        checked_key = (path, fact["label"], kind, match.start(), match.end())
                        if checked_key in checked_seen:
                            continue
                        checked_seen.add(checked_key)
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
            prefix = lowered[max(0, idx - 28) : idx]
            negated = re.search(r"(?:\bnot\b|\bno\b|\bwithout\b|\bneither\b|n't).{0,24}$", prefix)
            if not negated and not any(marker in context for marker in conditional_markers):
                return True
            start = idx + len(phrase)
    return False


def _logic_warnings(bundle: Dict[str, Any], texts: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    if not texts:
        return warnings

    risk_regime = str((bundle.get("overview", {}) or {}).get("risk_regime", "")).lower()
    risk_on_assertions = ["risk-on backdrop", "risk-on regime", "risk-on setup", "risk-on tape", "risk appetite improved", "risk appetite rose"]
    risk_off_assertions = ["risk-off backdrop", "risk-off regime", "risk-off setup", "risk-off tape", "risk appetite deteriorated", "risk appetite faded"]
    dxy = _parse_pct(_summary_item(bundle, "FX", "DXY").get("Pct Change"))
    us10y, us10y_unit = summary_change(_summary_item(bundle, "Rates", "10Y Treasury"))
    southbound = ((bundle.get("hk_local", {}) or {}).get("southbound_net_flow", {}) or {})

    relative_facts = {
        "S&P 500": (summary_change(_summary_item(bundle, "Equities", "S&P 500"))[0], "us"),
        "Nasdaq 100": (summary_change(_summary_item(bundle, "Equities", "Nasdaq 100"))[0], "us"),
        "Hang Seng": (summary_change(_summary_item(bundle, "Equities", "Hang Seng Index"))[0], "hk_prior"),
        "3033.HK": (summary_change(_summary_item(bundle, "Equities", "Hang Seng TECH ETF"))[0], "hk_prior"),
        "FXI": (summary_change(_summary_item(bundle, "Equities", "China Large-Cap (FXI)"))[0], "us"),
    }
    relative_aliases = {
        "S&P 500": ["S&P 500", "SPX"],
        "Nasdaq 100": ["Nasdaq 100", "Nasdaq", "NDX"],
        "Hang Seng": ["Hang Seng Index", "Hang Seng", "HSI"],
        "3033.HK": ["3033.HK ETF", "3033.HK", "HSTECH ETF", "HSTECH"],
        "FXI": ["FXI"],
    }
    relative_verbs = {
        "outperformed": lambda left, right: left > right,
        "beat": lambda left, right: left > right,
        "lagged": lambda left, right: left < right,
        "underperformed": lambda left, right: left < right,
    }
    relative_seen = set()

    for path, raw_text in texts:
        text = raw_text.lower()
        if "risk-on" in risk_regime and _contains_any(text, risk_off_assertions):
            warnings.append({"field": path, "type": "risk_regime", "severity": "review", "message": "Narrative asserts a risk-off setup while the deterministic overview is risk-on."})
        if "risk-off" in risk_regime and _contains_any(text, risk_on_assertions):
            warnings.append({"field": path, "type": "risk_regime", "severity": "review", "message": "Narrative asserts a risk-on setup while the deterministic overview is risk-off."})
        if dxy is not None:
            if dxy > 0.30 and _contains_unhedged_any(text, ["softer dollar", "weaker dollar", "dollar softened"]):
                warnings.append({"field": path, "type": "fx_logic", "severity": "review", "message": "Narrative says the dollar softened, but DXY was materially higher."})
            if dxy < -0.30 and _contains_unhedged_any(text, ["stronger dollar", "firmer dollar", "dollar strengthened"]):
                warnings.append({"field": path, "type": "fx_logic", "severity": "review", "message": "Narrative says the dollar strengthened, but DXY was materially lower."})
        if us10y is not None and us10y_unit == "bp":
            if us10y > 5.0 and _contains_unhedged_any(text, ["lower yields", "yields fell", "yields declined"]):
                warnings.append({"field": path, "type": "rates_logic", "severity": "review", "message": "Narrative says yields fell, but US 10Y was materially higher."})
            if us10y < -5.0 and _contains_unhedged_any(text, ["higher yields", "yields rose", "yields climbed"]):
                warnings.append({"field": path, "type": "rates_logic", "severity": "review", "message": "Narrative says yields rose, but US 10Y was materially lower."})
        if southbound.get("status") == "unavailable" and _contains_any(text, ["southbound net buy", "southbound net inflow", "southbound bought"]):
            warnings.append({"field": path, "type": "flow_availability", "severity": "review", "message": "Narrative discusses Southbound net buying although the normalized metric is unavailable."})

        for left_label, left_aliases in relative_aliases.items():
            left_value, left_session = relative_facts[left_label]
            if left_value is None:
                continue
            for right_label, right_aliases in relative_aliases.items():
                if right_label == left_label:
                    continue
                right_value, right_session = relative_facts[right_label]
                if right_value is None:
                    continue
                for left_alias in left_aliases:
                    for right_alias in right_aliases:
                        pattern = re.compile(
                            rf"(?<!\w){re.escape(left_alias)}(?!\w)[^\n.;]{{0,28}}?"
                            rf"\b(?P<verb>{'|'.join(relative_verbs)})\b[^\n.;]{{0,28}}?"
                            rf"(?<!\w){re.escape(right_alias)}(?!\w)",
                            re.IGNORECASE,
                        )
                        for match in pattern.finditer(raw_text):
                            key = (path, match.start(), left_label, right_label)
                            if key in relative_seen:
                                continue
                            relative_seen.add(key)
                            verb = match.group("verb").lower()
                            if not relative_verbs[verb](float(left_value), float(right_value)):
                                warnings.append(
                                    {
                                        "field": path,
                                        "type": "relative_performance",
                                        "severity": "critical",
                                        "message": (
                                            f"Narrative says {left_label} {verb} {right_label}, but the supplied moves were "
                                            f"{float(left_value):+.2f}% and {float(right_value):+.2f}%."
                                        ),
                                    }
                                )
                            elif left_session != right_session:
                                warnings.append(
                                    {
                                        "field": path,
                                        "type": "period_alignment",
                                        "severity": "review",
                                        "message": (
                                            f"{left_label} and {right_label} come from different trading sessions; "
                                            "describe confirmation/reversal rather than outperform/lag."
                                        ),
                                    }
                                )

    return warnings


def _source_and_date_warnings(bundle: Dict[str, Any]) -> Tuple[int, List[Dict[str, str]]]:
    """Validate structured event claims that numeric regexes cannot cover."""
    checked = 0
    warnings: List[Dict[str, str]] = []
    enforce_item_sources = bool(bundle.get("provenance_audit"))
    if not enforce_item_sources:
        return checked, warnings

    company_events = bundle.get("company_events", {}) or {}
    for bucket in ("earnings", "ratings"):
        for index, item in enumerate(company_events.get(bucket, []) or []):
            if not isinstance(item, dict):
                continue
            checked += 1
            if not str(item.get("source_url", item.get("url", "")) or "").strip():
                warnings.append(
                    {
                        "type": "missing_event_source",
                        "severity": "critical",
                        "message": f"{bucket}[{index}] is a publishable company-event claim without source_url.",
                    }
                )
            if not str(item.get("as_of", item.get("release_time", "")) or "").strip():
                warnings.append(
                    {
                        "type": "missing_event_as_of",
                        "severity": "critical",
                        "message": f"{bucket}[{index}] is missing an as_of or release timestamp.",
                    }
                )

    for index, item in enumerate(bundle.get("macro_agenda", []) or []):
        if not isinstance(item, dict):
            continue
        checked += 1
        if not str(item.get("source_url", item.get("url", "")) or "").strip():
            warnings.append(
                {
                    "type": "missing_macro_source",
                    "severity": "critical",
                    "message": f"macro_agenda[{index}] is missing source_url.",
                }
            )
        event_date = str(item.get("date", "") or "").strip()
        if event_date:
            try:
                datetime.strptime(event_date, "%Y-%m-%d")
            except ValueError:
                warnings.append(
                    {
                        "type": "invalid_event_date",
                        "severity": "critical",
                        "message": f"macro_agenda[{index}] has invalid date `{event_date}`.",
                    }
                )

    for index, item in enumerate(bundle.get("catalysts", []) or []):
        if not isinstance(item, dict):
            continue
        checked += 1
        if not str(item.get("source_url", item.get("url", "")) or "").strip():
            warnings.append(
                {
                    "type": "missing_catalyst_source",
                    "severity": "critical",
                    "message": f"catalysts[{index}] is a dated event without source_url.",
                }
            )

    for index, item in enumerate((bundle.get("risk", {}) or {}).get("geopolitical_risks", []) or []):
        if not isinstance(item, dict):
            continue
        checked += 1
        if not str(item.get("source_url", item.get("url", "")) or "").strip():
            warnings.append(
                {
                    "type": "missing_risk_source",
                    "severity": "critical",
                    "message": f"geopolitical_risks[{index}] is missing source_url.",
                }
            )

    return checked, warnings


def _truncation_warnings(llm_sections: Dict[str, Any]) -> List[Dict[str, str]]:
    warnings: List[Dict[str, str]] = []
    incomplete_tail = re.compile(r"\b(?:a|an|and|at|by|for|from|in|its|of|on|or|the|to|with)\.?$", re.IGNORECASE)
    for path, text in _iter_texts(llm_sections):
        normalized = text.strip()
        if len(normalized) < 40:
            continue
        if normalized.endswith(("...", "…", "[trimmed]")) or incomplete_tail.search(normalized):
            warnings.append(
                {
                    "field": path,
                    "type": "truncated_text",
                    "severity": "critical",
                    "message": f"Narrative field `{path}` appears truncated.",
                }
            )
    return warnings[:12]


def run_fact_check(bundle: Dict[str, Any]) -> Dict[str, Any]:
    llm_sections = {
        key: value
        for key, value in (bundle.get("llm_sections", {}) or {}).items()
        if key != "task_meta"
    }
    texts = list(_iter_texts(llm_sections))
    deterministic_sections = {
        "overview": bundle.get("overview", {}),
        "macro_agenda": bundle.get("macro_agenda", []),
        "company_events": bundle.get("company_events", {}),
        "sector_digest": bundle.get("sector_digest", {}),
        "movers_digest": bundle.get("movers_digest", {}),
        "risk": bundle.get("risk", {}),
    }
    all_texts = texts + list(_iter_texts(deterministic_sections, "deterministic"))
    checked, mismatches = _claim_mismatches(bundle, all_texts)
    logic_warnings = _logic_warnings(bundle, texts)
    structured_checked, source_warnings = _source_and_date_warnings(bundle)
    source_warnings.extend(_truncation_warnings(llm_sections))
    logic_critical = sum(1 for item in logic_warnings if item.get("severity") == "critical")
    critical_count = sum(1 for item in mismatches if item.get("severity") == "critical") + logic_critical
    review_count = (
        sum(1 for item in mismatches if item.get("severity") == "review")
        + sum(1 for item in logic_warnings if item.get("severity") == "review")
    )
    source_critical = sum(1 for item in source_warnings if item.get("severity") == "critical")
    source_review = sum(1 for item in source_warnings if item.get("severity") == "review")
    status = "warning" if critical_count or review_count or source_critical or source_review else "ok"
    release_blocking = bool(critical_count or source_critical)
    summary = (
        f"Checked {checked} numeric claims; "
        f"{len(mismatches)} numeric mismatch(es), {len(logic_warnings)} logic warning(s); "
        f"{structured_checked} structured claims, {len(source_warnings)} source/text warning(s); "
        f"{critical_count + source_critical} critical, {review_count + source_review} review."
    )
    return {
        "status": status,
        "release_blocking": release_blocking,
        "summary": summary,
        "numeric_claims_checked": checked,
        "numeric_mismatches": mismatches[:12],
        "logic_warnings": logic_warnings[:12],
        "structured_claims_checked": structured_checked,
        "source_warnings": source_warnings[:12],
    }


_PATH_TOKEN = re.compile(r"([^.[\]]+)|\[(\d+)\]")


def _clear_path(root: Dict[str, Any], path: str) -> bool:
    tokens: List[str | int] = []
    for key, index in _PATH_TOKEN.findall(path):
        tokens.append(int(index) if index else key)
    if not tokens:
        return False
    cursor: Any = root
    for token in tokens[:-1]:
        try:
            cursor = cursor[token]
        except (KeyError, IndexError, TypeError):
            return False
    final = tokens[-1]
    try:
        if isinstance(cursor, dict) and isinstance(final, str):
            cursor[final] = ""
        elif isinstance(cursor, list) and isinstance(final, int):
            cursor[final] = ""
        else:
            return False
    except (IndexError, TypeError):
        return False
    return True


def apply_fact_check_fallbacks(bundle: Dict[str, Any], fact_check: Dict[str, Any]) -> List[str]:
    """Remove unsafe LLM fields so deterministic report copy can take over."""
    candidate_paths = set()
    for group in ("numeric_mismatches", "logic_warnings", "source_warnings"):
        for item in fact_check.get(group, []) or []:
            path = str(item.get("field", "") or "")
            if path and not path.startswith("deterministic"):
                candidate_paths.add(path)
    llm_sections = bundle.get("llm_sections", {}) or {}
    cleared = [path for path in sorted(candidate_paths) if _clear_path(llm_sections, path)]
    return cleared
