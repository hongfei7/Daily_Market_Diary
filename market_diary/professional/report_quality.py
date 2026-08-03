"""Report completeness and quality scoring."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


STATUS_SCORE = {
    "ok": 1.0,
    "cached": 1.0,
    "partial": 0.65,
    "partial_public": 0.65,
    "partial_local": 0.65,
    "live_local": 1.0,
    "live_public": 1.0,
    "live_quote": 0.9,
    "live_hybrid": 1.0,
    "stale_local": 0.7,
    "stale_public": 0.7,
    "proxy": 0.35,
    "derived": 0.75,
    "skipped": 0.55,
    "disabled": 0.55,
    "timeout": 0.0,
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


def _runtime_bucket(status: str) -> str:
    normalized = str(status or "").lower()
    if normalized in {"ok", "cached", "live_local", "live_public", "live_quote", "live_hybrid"}:
        return "healthy"
    if normalized in {"partial", "partial_public", "partial_local", "stale_local", "stale_public", "proxy", "derived", "skipped", "disabled"}:
        return "caveat"
    return "failed"


def _runtime_summary(bundle: Dict[str, Any]) -> Tuple[str, List[Dict[str, str]]]:
    source_inputs = bundle.get("source_health_inputs", {}) or {}
    llm_meta = ((bundle.get("llm_sections", {}) or {}).get("task_meta", {}) or {})
    llm_status = str(llm_meta.get("status", "") or "").strip() or (
        "ok" if (llm_meta.get("tasks", {}) or {}) else "skipped"
    )

    rows = [
        {"name": "Market data", "status": str(((source_inputs.get("market_data", {}) or {}).get("status", "unavailable"))), "bucket": ""},
        {"name": "Sector and company news", "status": str(((source_inputs.get("sector_news", {}) or {}).get("status", "unavailable"))), "bucket": ""},
        {"name": "Movers and short selling", "status": str(((source_inputs.get("movers", {}) or {}).get("status", "unavailable"))), "bucket": ""},
        {"name": "Stock Connect", "status": str(((source_inputs.get("stock_connect", {}) or {}).get("status", "unavailable"))), "bucket": ""},
        {"name": "A/H premium", "status": str(((source_inputs.get("ah_premium", {}) or {}).get("status", "unavailable"))), "bucket": ""},
        {"name": "Hong Kong local package", "status": str(((source_inputs.get("hk_local", {}) or {}).get("status", "unavailable"))), "bucket": ""},
        {"name": "China rates", "status": str(((source_inputs.get("china_rates", {}) or {}).get("status", "unavailable"))), "bucket": ""},
        {"name": "Narrative overlay", "status": llm_status, "bucket": ""},
    ]
    if "macro_calendar" in source_inputs:
        rows.insert(-1, {"name": "Macro calendar", "status": str(((source_inputs.get("macro_calendar", {}) or {}).get("status", "unavailable"))), "bucket": ""})
    if "risk_feed" in source_inputs:
        rows.insert(-1, {"name": "Risk and sentiment feed", "status": str(((source_inputs.get("risk_feed", {}) or {}).get("status", "unavailable"))), "bucket": ""})

    counts = {"healthy": 0, "caveat": 0, "failed": 0}
    for row in rows:
        row["bucket"] = _runtime_bucket(row["status"])
        counts[row["bucket"]] += 1

    summary = f"{counts['healthy']} healthy | {counts['caveat']} caveat | {counts['failed']} failed"
    return summary, rows


def _guidance(level: str, message: str) -> Dict[str, str]:
    return {"level": level, "message": message}


def _runtime_guidance(runtime_rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    by_name = {str(item.get("name", "")): str(item.get("bucket", "")) for item in runtime_rows}
    guidance: List[Dict[str, str]] = []

    def _needs_attention(name: str) -> bool:
        return by_name.get(name) in {"failed", "caveat"}

    def _is_failed(name: str) -> bool:
        return by_name.get(name) == "failed"

    if by_name and all(bucket == "healthy" for bucket in by_name.values()):
        return [_guidance("advisory", "All monitored sources were healthy for this run; the report can be used as a normal desk-starting point.")]

    if _needs_attention("Market data"):
        guidance.append(
            _guidance(
                "blocking" if _is_failed("Market data") else "advisory",
                "Cross-asset framing is incomplete; verify major equity, FX, and rates moves manually before leaning on the narrative setup.",
            )
        )
    if _needs_attention("Hong Kong local package"):
        guidance.append(
            _guidance(
                "blocking" if _is_failed("Hong Kong local package") else "advisory",
                "Treat Hong Kong liquidity and participation reads cautiously until turnover, HIBOR, and local funding checks are manually confirmed.",
            )
        )
    if _needs_attention("Stock Connect"):
        guidance.append(
            _guidance(
                "blocking" if _is_failed("Stock Connect") else "advisory",
                "Do not overstate mainland flow confirmation; Southbound and Northbound signals are incomplete for this run.",
            )
        )
    if _needs_attention("A/H premium"):
        guidance.append(
            _guidance(
                "advisory",
                "Avoid strong A/H valuation-dispersion conclusions until the premium snapshot refreshes.",
            )
        )
    if _needs_attention("Sector and company news"):
        guidance.append(
            _guidance(
                "blocking" if _is_failed("Sector and company news") else "advisory",
                "Company-event coverage may be incomplete; scan HKEXnews and key wire headlines manually before acting on single-name catalysts.",
            )
        )
    if _needs_attention("Narrative overlay"):
        guidance.append(
            _guidance(
                "advisory",
                "Use deterministic sections as primary support; the narrative overlay was partial or unavailable on this run.",
            )
        )
    if _needs_attention("China rates") and len(guidance) < 4:
        guidance.append(
            _guidance(
                "advisory",
                "Be careful with China macro carry and rates-spread conclusions until the China rates adapter refreshes.",
            )
        )
    if _needs_attention("Movers and short selling") and len(guidance) < 4:
        guidance.append(
            _guidance(
                "blocking" if _is_failed("Movers and short selling") else "advisory",
                "Short-term leadership and pressure signals may be incomplete; confirm movers and short-selling concentration manually.",
            )
        )

    return guidance[:4]


def _release_recommendation(
    runtime_rows: List[Dict[str, str]],
    runtime_guidance: List[Dict[str, str]],
    provenance_status: str = "",
    fact_status: str = "",
    fact_blocking: bool = False,
    source_health_status: str = "",
) -> Dict[str, str]:
    blocking_count = sum(1 for item in runtime_guidance if item.get("level") == "blocking")
    has_caveat = any(str(item.get("bucket", "")) == "caveat" for item in runtime_rows)
    has_failed = any(str(item.get("bucket", "")) == "failed" for item in runtime_rows)

    if provenance_status == "error":
        return {
            "action": "manual_review",
            "label": "Manual review",
            "reason": "Source provenance validation failed; automatic distribution is blocked.",
        }
    if source_health_status == "failed":
        return {
            "action": "manual_review",
            "label": "Manual review",
            "reason": "A critical data source failed its availability or freshness policy; automatic distribution is blocked.",
        }
    if fact_status == "error" or fact_blocking:
        return {
            "action": "manual_review",
            "label": "Manual review",
            "reason": "Fact validation produced a release-blocking error; automatic distribution is blocked.",
        }
    if blocking_count:
        return {
            "action": "manual_review",
            "label": "Manual review",
            "reason": "Critical source failures were detected; check the blocking guidance before sending the report externally.",
        }
    if fact_status == "warning" or has_failed or has_caveat:
        return {
            "action": "send_with_caveats",
            "label": "Send with caveats",
            "reason": "The report is usable, but the advisory guidance should travel with it so readers understand the data gaps.",
        }
    return {
        "action": "send",
        "label": "Send",
        "reason": "All monitored sources were healthy, so the report is fit for normal distribution.",
    }


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
    source_warnings = fact_check.get("source_warnings", []) or []
    critical = sum(1 for item in mismatches if item.get("severity", "critical") == "critical")
    review = (
        sum(1 for item in mismatches if item.get("severity", "critical") == "review")
        + sum(1 for item in warnings if item.get("severity", "review") == "review")
        + sum(1 for item in source_warnings if item.get("severity", "critical") == "review")
    )
    critical += sum(1 for item in source_warnings if item.get("severity", "critical") == "critical")
    info = sum(1 for item in warnings + source_warnings if item.get("severity") == "info")
    penalty = critical * 25.0 + review * 12.5 + info * 4.0
    score = max(0.0, 100.0 - penalty)
    return score, fact_check.get("summary", "Fact-check diagnostics were available.")


def _provenance_score(bundle: Dict[str, Any]) -> Tuple[float, str]:
    audit = bundle.get("provenance_audit", {}) or {}
    if not audit:
        return 75.0, "Source provenance validation was not attached to this compatibility fixture."
    if audit.get("status") != "ok":
        return 0.0, f"Source provenance failed with {len(audit.get('errors', []) or [])} error(s)."
    checked = int(audit.get("checked_records", 0) or 0)
    unavailable = int(audit.get("unavailable_records", 0) or 0)
    score = max(60.0, 100.0 - unavailable * 4.0)
    return score, f"{checked} provenance record(s) validated; {unavailable} unavailable source record(s)."


def _source_health_score(bundle: Dict[str, Any]) -> Tuple[float, str]:
    health = bundle.get("source_health", {}) or {}
    if not health:
        return 75.0, "Source-health scoring was not attached to this compatibility fixture."
    rows = health.get("sources", []) or []
    scores = [float(item.get("score", 0.0) or 0.0) for item in rows]
    score = sum(scores) / len(scores) if scores else 0.0
    coverage = health.get("coverage", {}) or {}
    return score, (
        f"{coverage.get('healthy', 0)} healthy, {coverage.get('degraded', 0)} degraded, "
        f"{coverage.get('unavailable', 0)} unavailable source group(s)."
    )


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
    provenance_score, provenance_read = _provenance_score(bundle)
    source_health_score, source_health_read = _source_health_score(bundle)
    runtime_summary, runtime_rows = _runtime_summary(bundle)
    runtime_guidance = _runtime_guidance(runtime_rows)
    provenance_audit = bundle.get("provenance_audit", {}) or {}
    fact_check = bundle.get("fact_check", {}) or {}
    provenance_status = str(provenance_audit.get("status", "") or "")
    fact_status = str(fact_check.get("status", "") or "")
    fact_blocking = bool(fact_check.get("release_blocking", fact_status == "error"))
    source_health_status = str((bundle.get("source_health", {}) or {}).get("status", "") or "")
    if provenance_status == "error":
        runtime_guidance.insert(0, _guidance("blocking", "Source provenance is incomplete or invalid; do not distribute automatically."))
    if fact_status == "error" or fact_blocking:
        runtime_guidance.insert(0, _guidance("blocking", "Fact validation has a release-blocking finding; review it before distribution."))
    elif fact_status == "warning":
        runtime_guidance.insert(0, _guidance("advisory", "Questionable narrative fields were removed; use the deterministic fallback copy and review the audit trail."))
    if source_health_status == "failed":
        failures = ", ".join((bundle.get("source_health", {}) or {}).get("critical_failures", []) or [])
        runtime_guidance.insert(0, _guidance("blocking", f"Critical source freshness or availability failed: {failures or 'unspecified source'}."))
    release_recommendation = _release_recommendation(
        runtime_rows,
        runtime_guidance,
        provenance_status=provenance_status,
        fact_status=fact_status,
        fact_blocking=fact_blocking,
        source_health_status=source_health_status,
    )
    blocking_guidance = sum(1 for item in runtime_guidance if item.get("level") == "blocking")
    advisory_guidance = sum(1 for item in runtime_guidance if item.get("level") == "advisory")

    components = [
        _component("Market data coverage", market_score, 0.20, market_read),
        _component("Hong Kong local metrics", local_score, 0.20, local_read),
        _component("Key public adapters", adapter_score, 0.20, adapter_read),
        _component("Narrative overlay health", llm_score, 0.10, llm_read),
        _component("Fact-check guardrail", fact_score, 0.15, fact_read),
        _component("Source provenance", provenance_score, 0.05, provenance_read),
        _component("Source health and freshness", source_health_score, 0.10, source_health_read),
    ]
    score = sum(item["weighted_score"] for item in components)

    warnings: List[str] = []
    for item in components:
        if item["score"] < 60:
            warnings.append(f"{item['name']} is weak: {item['read']}")
    if fact_check.get("status") == "warning":
        warnings.append("Narrative fact-check guardrail produced warnings; review the validation table before relying on narrative sections.")
    if provenance_status == "error":
        warnings.append("Source provenance validation failed; automatic distribution must remain blocked.")

    quality_status = "manual_review" if release_recommendation.get("action") == "manual_review" else _score_to_status(score)

    return {
        "score": round(score, 1),
        "grade": _score_to_grade(score),
        "status": quality_status,
        "components": components,
        "adapter_status": adapter_rows,
        "runtime_summary": runtime_summary,
        "runtime_status": runtime_rows,
        "runtime_guidance": runtime_guidance,
        "runtime_guidance_summary": f"{blocking_guidance} blocking | {advisory_guidance} advisory",
        "release_recommendation": release_recommendation,
        "warnings": warnings[:8],
    }
