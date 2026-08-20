from __future__ import annotations

from typing import Any, Dict, List, Optional

from market_diary.professional.analytics_public_flow import _format_hkd_billions, _format_rmb_billions


def _classify_catalyst(catalyst: str) -> str:
    lowered = (catalyst or "").lower()
    if any(token in lowered for token in ("earnings", "guidance", "upgrade", "downgrade")):
        return "Announcement / expectations"
    if any(token in lowered for token in ("policy", "regulation", "approval")):
        return "Policy catalyst"
    if any(token in lowered for token in ("volume", "flow")):
        return "Flow-driven"
    return "Event-driven"


def build_movers_and_flows(movers_data: Dict[str, Any], risk_data: Dict[str, Any]) -> Dict[str, Any]:
    movers: List[Dict[str, Any]] = []
    premarket = (movers_data or {}).get("premarket_movers", {}) or {}
    for side in ("gainers", "losers"):
        for stock in (premarket.get(side, []) or [])[:3]:
            change_pct = stock.get("change_pct", 0)
            movers.append(
                {
                    "ticker": stock.get("ticker", ""),
                    "move": stock.get("change_pct"),
                    "title": f"{stock.get('ticker', '')} {change_pct:+.2f}%",
                    "summary": stock.get("catalyst", ""),
                    "attribution": _classify_catalyst(stock.get("catalyst", "")),
                    "score": abs(change_pct) + 1.5,
                }
            )

    flow_bullets: List[str] = []
    etf_flows = (movers_data or {}).get("etf_flows", []) or []
    if etf_flows:
        top = etf_flows[:3]
        summary = "; ".join(
            f"{item['ticker']} {item['change_pct']:+.2f}% / volume ratio {item['volume_ratio']:.2f}x / {item['estimated_flow_direction']}"
            for item in top
        )
        flow_bullets.append(f"ETF flow anomalies were concentrated in: {summary}.")

    short_sell = (movers_data or {}).get("short_sell", {}) or {}
    short_sell_data = short_sell.get("data", {}) if isinstance(short_sell, dict) else {}
    short_sell_meta = short_sell.get("meta", {}) if isinstance(short_sell, dict) else {}
    short_market = (short_sell_data or {}).get("market", {}) or {}
    if short_market.get("short_ratio_pct") is not None:
        flow_bullets.append(
            "HKEX short selling: "
            f"{short_market.get('short_ratio_pct'):.2f}% of market turnover "
            f"({_format_hkd_billions(short_market.get('short_turnover_hkd'))} short turnover, "
            f"as of {short_sell_meta.get('effective_date', 'N/A')})."
        )

    top_short_ratio = (short_sell_data or {}).get("top_short_ratio", []) or []
    if top_short_ratio:
        leaders = "; ".join(
            f"{item.get('ticker') or str(item.get('code', '')) + '.HK'} {item.get('short_ratio_pct'):.2f}%"
            for item in top_short_ratio[:3]
        )
        flow_bullets.append(f"Highest stock-level short ratios were concentrated in: {leaders}.")

    options = (movers_data or {}).get("unusual_options", []) or []
    if options:
        opt = options[0]
        flow_bullets.append(
            f"Options activity centered on {opt.get('ticker')} {opt.get('option_type')}, with Vol/OI at {opt.get('volume_oi_ratio')}x and a {opt.get('sentiment')} bias."
        )

    sentiment = (risk_data or {}).get("sentiment_indicators", {}) or {}
    put_call = sentiment.get("put_call_ratio", {}) or {}
    if put_call:
        flow_bullets.append(
            f"Put/Call structure shows equity {put_call.get('equity')} / index {put_call.get('index')}, implying {put_call.get('interpretation', '')}."
        )

    return {
        "movers": movers,
        "flow_bullets": flow_bullets,
        "etf_flows": etf_flows[:8],
        "short_sell": short_sell,
        "short_sell_top_ratio": top_short_ratio[:8],
        "short_sell_top_value": ((short_sell_data or {}).get("top_short_value", []) or [])[:8],
        "short_sell_watchlist_hits": ((short_sell_data or {}).get("watchlist_hits", []) or [])[:8],
    }


def _stock_connect_bullet(stock_connect_data: Dict[str, Any]) -> List[str]:
    if not stock_connect_data or stock_connect_data.get("status") not in {"ok", "partial"}:
        return []
    data = stock_connect_data.get("data", {}) or {}
    meta = stock_connect_data.get("meta", {}) or {}
    bullets: List[str] = []
    southbound = data.get("southbound", {}) or {}
    northbound = data.get("northbound", {}) or {}
    if southbound.get("net_buy") is not None:
        bullets.append(
            "Stock Connect Southbound: "
            f"net {_format_hkd_billions(float(southbound.get('net_buy', 0)) * 1_000_000.0)} "
            f"on turnover {_format_hkd_billions(float(southbound.get('total_turnover') or 0) * 1_000_000.0)} "
            f"(as of {meta.get('effective_date', 'N/A')})."
        )
    elif southbound.get("total_turnover") is not None:
        bullets.append(
            "Stock Connect Southbound: "
            f"turnover {_format_hkd_billions(float(southbound.get('total_turnover') or 0) * 1_000_000.0)}; "
            "net-buy is not available in the public daily file."
        )
    if northbound.get("total_turnover") is not None:
        bullets.append(
            "Stock Connect Northbound: "
            f"turnover {_format_rmb_billions(float(northbound.get('total_turnover') or 0) * 1_000_000.0)}; "
            "public file provides total turnover/top active names, not full net-buy."
        )
    return bullets


def _ah_premium_bullet(ah_premium_data: Dict[str, Any]) -> List[str]:
    if not ah_premium_data or ah_premium_data.get("status") not in {"ok", "partial"}:
        return []
    data = ah_premium_data.get("data", {}) or {}
    average = data.get("average_premium")
    top = data.get("top_premium", []) or []
    if average is None:
        return []
    leader = top[0] if top else {}
    suffix = (
        f"; widest premium: {leader.get('name', '')} {leader.get('premium_pct', 'N/A')}%"
        if leader
        else ""
    )

    basket = data.get("fixed_basket_premium")
    rows = data.get("rows", []) or []
    if basket is not None and data.get("fixed_basket_complete"):
        headline = (
            f"AH premium: fixed {data.get('fixed_basket_size')}-name basket {basket:+.2f}% "
            f"(comparable across dates); covered-pair average {average:+.2f}% over {len(rows)} pairs{suffix}."
        )
    else:
        headline = (
            f"AH premium: covered-pair average {average:+.2f}% over {len(rows)} pairs. "
            f"Composition varies by day, so this level is not comparable with prior reports{suffix}."
        )
    return [headline]


def build_flow_tracker(
    hk_quick_checks: List[Dict[str, str]],
    movers_digest: Dict[str, Any],
    attribution: Dict[str, Any],
    stock_connect_data: Optional[Dict[str, Any]] = None,
    ah_premium_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    check_map = {item.get("metric", ""): item for item in hk_quick_checks or []}
    key_metrics = [
        check_map.get("Main Board turnover vs 20D", {}),
        check_map.get("Short-selling ratio", {}),
        check_map.get("Southbound / Northbound net flow", {}),
        check_map.get("AH premium index", {}),
        check_map.get("HIBOR 1M", {}),
        check_map.get("Aggregate Balance", {}),
        check_map.get("HK ETF flow proxy", {}),
    ]
    key_metrics = [item for item in key_metrics if item]

    return {
        "summary": attribution.get("flow_summary", "Flow evidence was not conclusive."),
        "key_metrics": key_metrics,
        "flow_bullets": (
            _stock_connect_bullet(stock_connect_data or {})
            + _ah_premium_bullet(ah_premium_data or {})
            + ((movers_digest or {}).get("flow_bullets", []) or [])
        ),
        "stock_connect": stock_connect_data or {},
        "ah_premium": ah_premium_data or {},
        "short_sell_top_ratio": (movers_digest or {}).get("short_sell_top_ratio", []) or [],
        "short_sell_top_value": (movers_digest or {}).get("short_sell_top_value", []) or [],
        "short_sell_watchlist_hits": (movers_digest or {}).get("short_sell_watchlist_hits", []) or [],
        "etf_flows": (movers_digest or {}).get("etf_flows", []) or [],
    }
