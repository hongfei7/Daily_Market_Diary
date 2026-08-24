from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from market_diary.professional.analytics_briefing import (
    build_catalyst_calendar,
    build_company_event_digest,
    build_must_watch,
    build_source_links,
)
from market_diary.professional.ai_tmt_chain import build_ai_tmt_chain
from market_diary.professional.analytics_flows import build_flow_tracker, build_movers_and_flows
from market_diary.professional.analytics_hk_checks import build_hk_quick_checks
from market_diary.professional.analytics_macro import build_macro_agenda
from market_diary.professional.analytics_market import (
    build_hk_desk_view,
    build_market_overview,
)
from market_diary.professional.analytics_narrative import (
    build_non_trading_focus,
    build_reflection_prompts,
    build_theme_deep_dive,
    build_today_forward,
    build_week_ahead,
    build_weekly_review,
)
from market_diary.professional.analytics_public_flow import enrich_hk_local_with_public_flow
from market_diary.professional.analytics_sector import build_sector_news_digest
from market_diary.professional.analytics_trackers import build_high_frequency_trackers
from market_diary.professional.analytics_watchlist import build_watchlist_digest
from market_diary.professional.attribution import build_attribution
from market_diary.professional.date_policy import build_date_semantics, build_report_mode


def _default_source_status(payload: Optional[Dict[str, Any]], fallback_ok: bool = False) -> str:
    if isinstance(payload, dict):
        status = str(payload.get("status", "") or "").strip()
        if status:
            return status
    return "ok" if fallback_ok else "unavailable"


def _market_data_status(market_data: Dict[str, Any]) -> str:
    status = str((market_data or {}).get("status", "") or "").strip()
    if status:
        return status
    quality = ((market_data or {}).get("quality", {}) or {})
    quality_status = str(quality.get("status", "") or "").strip()
    if quality_status:
        return quality_status
    summary = ((market_data or {}).get("summary", {}) or {})
    return "ok" if summary else "unavailable"


def _sector_data_status(sector_data: Dict[str, Any]) -> str:
    explicit_status = str((sector_data or {}).get("status", "") or "").strip()
    if explicit_status:
        return explicit_status
    hkex_status = str((((sector_data or {}).get("hkex_announcements", {}) or {}).get("status", "")) or "").strip()
    if hkex_status in {"error", "timeout", "partial"}:
        return hkex_status
    if (
        (sector_data or {}).get("sector_news")
        or (sector_data or {}).get("earnings_calendar")
        or (sector_data or {}).get("analyst_changes")
        or hkex_status == "ok"
    ):
        return "ok"
    return "unavailable"


def _movers_data_status(movers_data: Dict[str, Any]) -> str:
    explicit_status = str((movers_data or {}).get("status", "") or "").strip()
    if explicit_status:
        return explicit_status
    short_sell_status = str((((movers_data or {}).get("short_sell", {}) or {}).get("status", "")) or "").strip()
    if short_sell_status in {"error", "timeout", "partial"}:
        return short_sell_status
    if (
        ((movers_data or {}).get("premarket_movers", {}) or {}).get("gainers")
        or ((movers_data or {}).get("premarket_movers", {}) or {}).get("losers")
        or (movers_data or {}).get("etf_flows")
        or (movers_data or {}).get("unusual_options")
        or short_sell_status == "ok"
    ):
        return "ok"
    return "unavailable"


def build_professional_bundle(
    report_date: str,
    config: Dict[str, Any],
    market_data: Dict[str, Any],
    chart_features: Dict[str, Any],
    macro_data: Dict[str, Any],
    sector_data: Dict[str, Any],
    movers_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    news_headlines: List[str],
    stock_connect_data: Optional[Dict[str, Any]] = None,
    ah_premium_data: Optional[Dict[str, Any]] = None,
    briefing_date: Optional[str] = None,
    global_market_date: Optional[str] = None,
    hk_data_date: Optional[str] = None,
    cn_data_date: Optional[str] = None,
    hk_local_data: Optional[Dict[str, Any]] = None,
    china_rates_data: Optional[Dict[str, Any]] = None,
    metric_history: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary = (market_data or {}).get("summary", {}) or {}
    market_meta = ((market_data or {}).get("meta", {}) or {})
    report_config = (config.get("report", {}) or {}).copy()
    morning_date = briefing_date or report_date
    global_date = global_market_date or market_meta.get("requested_date", report_date)
    local_date = hk_data_date or report_date
    hk_local_metrics = enrich_hk_local_with_public_flow(
        report_date=local_date,
        hk_local_metrics=((hk_local_data or {}).get("data", {}) or {}),
        stock_connect_data=stock_connect_data,
        ah_premium_data=ah_premium_data,
    )
    china_rate_metrics = ((china_rates_data or {}).get("data", {}) or {})

    overview = build_market_overview(summary, chart_features)
    hk_desk_view = build_hk_desk_view(summary, hk_local_metrics, metric_history, local_date)
    ai_tmt_chain = build_ai_tmt_chain(summary)
    day_mode = build_report_mode(morning_date, config)
    date_semantics = build_date_semantics(
        report_date=report_date,
        briefing_date=morning_date,
        global_market_date=global_date,
        hk_data_date=local_date,
        market_meta=market_meta,
        day_mode=day_mode,
        cn_data_date=cn_data_date or "",
    )
    macro_agenda = build_macro_agenda(morning_date, macro_data, config)
    sector_digest = build_sector_news_digest(sector_data, config)
    high_frequency = build_high_frequency_trackers(summary, chart_features)
    movers_digest = build_movers_and_flows(movers_data, risk_data)
    watchlists = build_watchlist_digest(config, report_date)
    catalysts = build_catalyst_calendar(morning_date, macro_agenda, sector_data, risk_data, watchlists, config)
    hk_quick_checks = build_hk_quick_checks(summary, movers_data, hk_desk_view, hk_local_metrics)
    company_events = build_company_event_digest(sector_data, sector_digest)
    attribution = build_attribution(summary, hk_local_metrics, movers_digest, overview)

    # One regime label, not two. The overview derives its own vote-count regime
    # while the composite score derives a weighted one, and both were printed:
    # the theme could read "Risk-On" while the Decision Board directly below
    # read "Mixed". The composite score wins because it is weighted, carries its
    # evidence components, and now includes the semiconductor term.
    composite_bucket = ((attribution.get("risk_dashboard", {}) or {}).get("bucket") or "").strip()
    if composite_bucket:
        overview["risk_regime"] = composite_bucket
        tail = overview.get("theme_tail")
        if tail:
            overview["theme"] = f"{composite_bucket} backdrop with {tail}"
    flow_tracker = build_flow_tracker(hk_quick_checks, movers_digest, attribution, stock_connect_data, ah_premium_data)
    theme_deep_dive = build_theme_deep_dive(morning_date, config, sector_digest, watchlists, high_frequency, catalysts)
    today_forward = build_today_forward(morning_date, macro_agenda, catalysts, day_mode=day_mode)
    non_trading_focus = build_non_trading_focus(
        day_mode=day_mode,
        date_semantics=date_semantics,
        overview=overview,
        macro_agenda=macro_agenda,
        sector_digest=sector_digest,
        high_frequency=high_frequency,
        catalysts=catalysts,
        risk_data=risk_data,
    )
    weekly_review = build_weekly_review(
        day_mode=day_mode,
        date_semantics=date_semantics,
        overview=overview,
        summary=summary,
        hk_desk_view=hk_desk_view,
        high_frequency=high_frequency,
        sector_digest=sector_digest,
        macro_agenda=macro_agenda,
        catalysts=catalysts,
        flow_tracker=flow_tracker,
        attribution=attribution,
    )
    week_ahead = build_week_ahead(
        day_mode=day_mode,
        date_semantics=date_semantics,
        overview=overview,
        summary=summary,
        hk_desk_view=hk_desk_view,
        macro_agenda=macro_agenda,
        sector_digest=sector_digest,
        high_frequency=high_frequency,
        catalysts=catalysts,
        risk_data=risk_data,
        flow_tracker=flow_tracker,
        attribution=attribution,
        config=config,
    )
    reflection_prompts = build_reflection_prompts(config, overview, hk_desk_view)
    source_links = build_source_links(sector_digest, watchlists, report_config, company_events=company_events)
    must_watch = build_must_watch(
        overview=overview,
        macro_agenda=macro_agenda,
        sector_digest=sector_digest,
        high_frequency=high_frequency,
        movers_digest=movers_digest,
        catalysts=catalysts,
        report_config=report_config,
        day_mode=day_mode,
    )

    return {
        "meta": {
            "report_date": report_date,
            "review_date": report_date,
            "briefing_date": morning_date,
            "data_through": local_date,
            "global_market_date": global_date,
            "hk_data_date": local_date,
            "cn_data_date": cn_data_date or "",
            "requested_date": market_meta.get("requested_date", global_date),
            "effective_date": market_meta.get("effective_date", global_date),
            "summary_date": market_meta.get("summary_date", market_meta.get("effective_date", global_date)),
            "market_quality": market_meta.get("market_quality", {}),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": config.get("config_path", ""),
        },
        "date_semantics": date_semantics,
        "overview": overview,
        "day_mode": day_mode,
        "hk_desk_view": hk_desk_view,
        "ai_tmt_chain": ai_tmt_chain,
        "market_summary": summary,
        "macro_agenda": macro_agenda,
        "sector_digest": sector_digest,
        "high_frequency": high_frequency,
        "movers_digest": movers_digest,
        "watchlists": watchlists,
        "catalysts": catalysts,
        "hk_local": hk_local_metrics,
        "hk_local_meta": (hk_local_data or {}).get("meta", {}) or {},
        "china_rates": china_rate_metrics,
        "china_rates_meta": (china_rates_data or {}).get("meta", {}) or {},
        "hk_quick_checks": hk_quick_checks,
        "company_events": company_events,
        "attribution": attribution,
        "flow_tracker": flow_tracker,
        "stock_connect": stock_connect_data or {},
        "ah_premium": ah_premium_data or {},
        "theme_deep_dive": theme_deep_dive,
        "today_forward": today_forward,
        "non_trading_focus": non_trading_focus,
        "weekly_review": weekly_review,
        "week_ahead": week_ahead,
        "reflection_prompts": reflection_prompts,
        "source_links": source_links,
        "must_watch": must_watch,
        "chart_features": chart_features,
        "raw_news_headlines": news_headlines[:20],
        "risk": risk_data,
        "report_config": report_config,
        "source_health_inputs": {
            "market_data": {"status": _market_data_status(market_data)},
            "sector_news": {"status": _sector_data_status(sector_data)},
            "movers": {"status": _movers_data_status(movers_data)},
            "stock_connect": {"status": _default_source_status(stock_connect_data)},
            "ah_premium": {"status": _default_source_status(ah_premium_data)},
            "hk_local": {"status": _default_source_status(hk_local_data, fallback_ok=bool(hk_local_metrics))},
            "china_rates": {"status": _default_source_status(china_rates_data, fallback_ok=bool(china_rate_metrics))},
            "macro_calendar": {"status": _default_source_status(macro_data)},
            "risk_feed": {"status": _default_source_status(risk_data)},
        },
    }
