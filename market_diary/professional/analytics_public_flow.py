from __future__ import annotations

from typing import Any, Dict, Optional

from market_diary.professional.analytics_market import _parse_float


def _format_hkd_billions(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    return f"HK${number / 1_000_000_000:.1f}bn"


def _format_rmb_billions(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    return f"RMB{number / 1_000_000_000:.1f}bn"


def _metric_unavailable(metric: Dict[str, Any]) -> bool:
    return not isinstance(metric, dict) or str(metric.get("status", "unavailable")) == "unavailable"


def _public_metric(
    *,
    value: Any,
    display_value: str,
    source: str,
    as_of: str,
    note: str,
    status: str = "live_public",
) -> Dict[str, Any]:
    return {
        "value": value,
        "display_value": display_value,
        "status": status,
        "source": source,
        "as_of": as_of,
        "freshness_days": None,
        "quality": "public_adapter",
        "fallback_used": True,
        "note": note,
    }


def enrich_hk_local_with_public_flow(
    report_date: str,
    hk_local_metrics: Dict[str, Any],
    stock_connect_data: Optional[Dict[str, Any]],
    ah_premium_data: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use already-fetched public adapters if the local rollup timed out."""
    metrics = dict(hk_local_metrics or {})

    if isinstance(stock_connect_data, dict) and stock_connect_data.get("status") in {"ok", "partial"}:
        data = stock_connect_data.get("data", {}) or {}
        meta = stock_connect_data.get("meta", {}) or {}
        source = meta.get("source", "HKEX Stock Connect Historical Daily")
        as_of = meta.get("effective_date", report_date)

        southbound = data.get("southbound", {}) or {}
        if _metric_unavailable(metrics.get("southbound_net_flow", {})):
            net_buy = southbound.get("net_buy")
            total_turnover = southbound.get("total_turnover")
            if net_buy is not None:
                net_hkd = float(net_buy) * 1_000_000.0
                turnover_hkd = float(total_turnover or 0) * 1_000_000.0
                metrics["southbound_net_flow"] = _public_metric(
                    value=net_hkd,
                    display_value=f"Net {_format_hkd_billions(net_hkd)} | turnover {_format_hkd_billions(turnover_hkd)}",
                    source=source,
                    as_of=as_of,
                    note="Derived directly from the already-fetched HKEX Stock Connect public adapter.",
                )
            elif total_turnover is not None:
                turnover_hkd = float(total_turnover) * 1_000_000.0
                metrics["southbound_net_flow"] = _public_metric(
                    value=turnover_hkd,
                    display_value=f"Turnover {_format_hkd_billions(turnover_hkd)} | net unavailable",
                    source=source,
                    as_of=as_of,
                    note="HKEX public file provided Southbound turnover, but not a comparable full-day net-buy figure.",
                    status="partial_public",
                )

        northbound = data.get("northbound", {}) or {}
        if _metric_unavailable(metrics.get("northbound_net_flow", {})):
            net_buy = northbound.get("net_buy")
            total_turnover = northbound.get("total_turnover")
            if net_buy is not None:
                net_rmb = float(net_buy) * 1_000_000.0
                turnover_rmb = float(total_turnover or 0) * 1_000_000.0
                metrics["northbound_net_flow"] = _public_metric(
                    value=net_rmb,
                    display_value=f"Net {_format_rmb_billions(net_rmb)} | turnover {_format_rmb_billions(turnover_rmb)}",
                    source=source,
                    as_of=as_of,
                    note="Derived directly from the already-fetched HKEX Stock Connect public adapter.",
                )
            elif total_turnover is not None:
                turnover_rmb = float(total_turnover) * 1_000_000.0
                metrics["northbound_net_flow"] = _public_metric(
                    value=turnover_rmb,
                    display_value=f"Turnover {_format_rmb_billions(turnover_rmb)} | net unavailable",
                    source=source,
                    as_of=as_of,
                    note="HKEX public file provided Northbound turnover, but not a comparable full-day net-buy figure.",
                    status="partial_public",
                )

    if isinstance(ah_premium_data, dict) and ah_premium_data.get("status") in {"ok", "partial"}:
        data = ah_premium_data.get("data", {}) or {}
        meta = ah_premium_data.get("meta", {}) or {}
        average = data.get("average_premium")
        rows = data.get("rows", []) or data.get("top_premium", []) or []
        if average is not None and _metric_unavailable(metrics.get("ah_premium_index", {})):
            metrics["ah_premium_index"] = _public_metric(
                value=float(average),
                display_value=f"{float(average):.2f}%",
                source=meta.get("source", "Public Yahoo Finance quotes - calculated A/H premium"),
                as_of=meta.get("effective_date", report_date),
                note=f"Simple covered-pair average across {len(rows)} public A/H observations; use dispersion rather than the average alone.",
            )

    return metrics
