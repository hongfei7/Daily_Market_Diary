"""Report completeness and quality scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


STATUS_SCORE = {
    "ok": 1.0,
    "cached": 1.0,
    "partial": 0.65,
    "live_local": 1.0,
    "live_public": 1.0,
    "live_quote": 0.9,
    "live_hybrid": 1.0,
    "stale_local": 0.7,
    "stale_public": 0.7,
    "proxy": 0.35,
    "skipped": 0.55,
    "disabled": 0.55,
    "unavailable": 0.0,
    "error": 0.0,
}


def _score_to_grade(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "E"


def _score_to_status(score: float) -> str:
    if score >= 85:
        return "production_ready"
    if score >= 70:
        return "usable_with_caveats"
    if score >= 55:
        return "partial"
    return "weak"


def _market_coverage_score(bundle: Dict[str, Any]) -> Tuple[float, str]:
    quality = ((bundle.get("meta", {}) or {}).get("market_quality", {}) or {})
    available = quality.get("available")
    total = quality.get("total")
    if isinstance(available, int) and isinstance(total, int) and total > 0:
        score = max(0.0, min(100.0, available / total * 100.0))
        return score, f"{available}/{total} core market fields available."
    return 50.0, "Market-data coverage metadata was not available."


def _status_score(status: Any) -> float:
    return STATUS_SCORE.get(str(status or "").lower(), 0.4)


def _local_metrics_score(bundle: Dict[str, Any]) -> Tuple[float, str]:
    rows = bundle.get("hk_quick_checks", []) or []
    if not rows:
        return 0.0, "Hong Kong quick-check metrics were unavailable."
    scores = [_status_score(item.get("status")) for item in rows]
    live_count = sum(1 for item in rows if str(item.get("status", "")).startswith("live"))
    proxy_count = sum(1 for item in rows if item.get("status") == "proxy")
    unavailable_count = sum(1 for item in rows if item.get("status") == "unavailable")
    score = sum(scores) / len(scores) * 100.0
    return score, f"{live_count} live, {proxy_count} proxy, {unavailable_count} unavailable local checks."


def _adapter_score(bundle: Dict[str, Any]) -> Tuple[float, str, List[Dict[str, str]]]:
    adapters = [
        ("Stock Connect", (bundle.get("stock_connect", {}) or {}).get("status", "unavailable")),
        ("A/H premium", (bundle.get("ah_premium", {}) or {}).get("status", "unavailable")),
        ("HKEX announcements", ((bundle.get("company_events", {}) or {}).get("hkex_meta", {}) or {}).get("status", "ok")),
        ("China rates", "ok" if any((bundle.get("china_rates", {}) or {}).values()) else "unavailable"),
    ]
    rows = [{"name": name, "status": str(status)} for name, status in adapters]
    if not adapters:
        return 0.0, "No adapter statuses were available.", rows
    score = sum(_status_score(status) for _, status in adapters) / len(adapters) * 100.0
    active = sum(1 for _, status in adapters if _status_score(status) >= 0.65)
    return score, f"{active}/{len(adapters)} key adapters were available or partially available.", rows


def _llm_score(bundle: Dict[str, Any]) -> Tuple[float, str]:
    task_meta = (((bundle.get("llm_sections", {}) or {}).get("task_meta", {}) or {}).get("tasks", {}) or {})
    if not task_meta:
        return 55.0, "Narrative overlay was not run or did not provide task metadata."
    scores = [_status_score(meta.get("status")) for meta in task_meta.values() if isinstance(meta, dict)]
    if not scores:
        return 55.0, "Narrative overlay task metadata was empty."
    ok_count = sum(1 for meta in task_meta.values() if isinstance(meta, dict) and meta.get("status") in {"ok", "cached"})
    error_count = sum(1 for meta in task_meta.values() if isinstance(meta, dict) and meta.get("status") == "error")
    score = sum(scores) / len(scores) * 100.0
    return score, f"{ok_count}/{len(scores)} narrative overlay tasks succeeded or used validated cache; {error_count} error(s)."


def _fact_check_score(bundle: Dict[str, Any]) -> Tuple[float, str]:
    fact_check = bundle.get("fact_check", {}) or {}
    status = fact_check.get("status", "skipped")
    if status == "skipped":
        return 75.0, fact_check.get("summary", "Fact check was skipped.")
    mismatches = fact_check.get("numeric_mismatches", []) or []
    warnings = fact_check.get("logic_warnings", []) or []
    critical = sum(1 for item in mismatches if item.get("severity", "critical") == "critical")
    review = (
        sum(1 for item in mismatches if item.get("severity", "critical") == "review")
        + sum(1 for item in warnings if item.get("severity", "review") == "review")
    )
    info = sum(1 for item in warnings if item.get("severity") == "info")
    penalty = critical * 25.0 + review * 12.5 + info * 4.0
    score = max(0.0, 100.0 - penalty)
    return score, fact_check.get("summary", "Fact-check diagnostics were available.")


def _component(name: str, score: float, weight: float, read: str) -> Dict[str, Any]:
    return {
        "name": name,
        "score": round(score, 1),
        "weight": weight,
        "weighted_score": round(score * weight, 1),
        "read": read,
    }


def build_report_quality(bundle: Dict[str, Any]) -> Dict[str, Any]:
    market_score, market_read = _market_coverage_score(bundle)
    local_score, local_read = _local_metrics_score(bundle)
    adapter_score, adapter_read, adapter_rows = _adapter_score(bundle)
    llm_score, llm_read = _llm_score(bundle)
    fact_score, fact_read = _fact_check_score(bundle)

    components = [
        _component("Market data coverage", market_score, 0.30, market_read),
        _component("Hong Kong local metrics", local_score, 0.25, local_read),
        _component("Key public adapters", adapter_score, 0.20, adapter_read),
        _component("Narrative overlay health", llm_score, 0.15, llm_read),
        _component("Fact-check guardrail", fact_score, 0.10, fact_read),
    ]
    score = sum(item["weighted_score"] for item in components)

    warnings: List[str] = []
    for item in components:
        if item["score"] < 60:
            warnings.append(f"{item['name']} is weak: {item['read']}")
    fact_check = bundle.get("fact_check", {}) or {}
    if fact_check.get("status") == "warning":
        warnings.append("Narrative fact-check guardrail produced warnings; review the validation table before relying on narrative sections.")

    return {
        "score": round(score, 1),
        "grade": _score_to_grade(score),
        "status": _score_to_status(score),
        "components": components,
        "adapter_status": adapter_rows,
        "warnings": warnings[:8],
    }
