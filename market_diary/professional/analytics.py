from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from market_diary.professional.analytics_briefing import (
    build_catalyst_calendar,
    build_company_event_digest,
    build_must_watch,
    build_source_links,
)
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
    build_weekly_review,
)
from market_diary.professional.analytics_public_flow import enrich_hk_local_with_public_flow
from market_diary.professional.analytics_sector import build_sector_news_digest
from market_diary.professional.analytics_trackers import build_high_frequency_trackers
from market_diary.professional.analytics_watchlist import build_watchlist_digest
from market_diary.professional.attribution import build_attribution
from market_diary.professional.date_policy import build_date_semantics, build_report_mode


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
    hk_local_data: Optional[Dict[str, Any]] = None,
    china_rates_data: Optional[Dict[str, Any]] = None,
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
    hk_desk_view = build_hk_desk_view(summary)
    day_mode = build_report_mode(report_date, config, briefing_date=morning_date)
    date_semantics = build_date_semantics(
        report_date=report_date,
        briefing_date=morning_date,
        global_market_date=global_date,
        hk_data_date=local_date,
        market_meta=market_meta,
        day_mode=day_mode,
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
        "reflection_prompts": reflection_prompts,
        "source_links": source_links,
        "must_watch": must_watch,
        "chart_features": chart_features,
        "raw_news_headlines": news_headlines[:20],
        "risk": risk_data,
        "report_config": report_config,
    }
