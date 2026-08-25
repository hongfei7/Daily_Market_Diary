from __future__ import annotations

from typing import Any, Dict

from market_diary.professional.report_blocks import (
    _render_ai_tmt_chain,
    _render_attribution,
    _render_call_scorecard,
    _render_company_events,
    _render_catalyst_radar,
    _render_daily_one_chart,
    _render_executive_summary,
    _render_flow_tracker,
    _render_flows,
    _render_hk_review_block,
    _render_internal_notes,
    _render_macro_table,
    _render_macro_takeaway,
    _render_macro_watchpoints,
    _render_md_questions,
    _render_news_table,
    _render_overseas_review_block,
    _render_performance,
    _render_report_quality,
    _render_session_playbook,
    _render_sources,
    _render_theme_deep_dive,
    _render_trend_pack,
    _render_watchlists,
)
from market_diary.professional.report_layout import build_report_layout
from market_diary.professional.report_sections import (
    _render_global_asset_dashboard,
    _render_hk_quick_checks,
    _render_non_trading_focus,
    _render_risk_dashboard,
    _render_selected_news,
    _render_top_items,
    _render_week_ahead,
    _render_weekly_review,
)
from market_diary.professional.report_text import _clean_report_spacing


def render_professional_report(
    bundle: Dict[str, Any],
    charts_section: str,
    dashboard_rel_path: str = "",
    daily_chart_rel_path: str = "",
    trend_pack_rel_path: str = "",
    catalyst_radar_rel_path: str = "",
    ai_tmt_chart_rel_path: str = "",
) -> str:
    layout = build_report_layout(bundle, dashboard_rel_path=dashboard_rel_path)
    meta = layout["meta"]
    day_mode = layout["day_mode"]
    llm_sections = layout["llm_sections"]
    dashboard_md = layout["dashboard_md"]
    appendix_meta_block = layout["appendix_meta_block"]
    pulse = layout["pulse"]
    macro_takeaway = layout["macro_takeaway"]
    is_week_ahead = layout["is_week_ahead"]
    layer_one_title = layout["layer_one_title"]
    checklist_title = layout["checklist_title"]
    call_title = layout["call_title"]
    non_trading_lens = layout["non_trading_lens"]
    briefing_date = layout["briefing_date"]
    global_date = layout["global_date"]
    hk_date = layout["hk_date"]
    cn_date = layout["cn_date"]
    internal_notes = _render_internal_notes(bundle)
    catalyst_radar_block = _render_catalyst_radar(bundle, catalyst_radar_rel_path)
    catalyst_radar_section = f"\n{catalyst_radar_block}\n" if catalyst_radar_block else ""
    daily_chart_block = _render_daily_one_chart(bundle, daily_chart_rel_path)
    ai_tmt_chart_block = (
        f"![AI / TMT read-through]({ai_tmt_chart_rel_path})\n\n" if ai_tmt_chart_rel_path else ""
    )
    trend_pack_block = _render_trend_pack(bundle, trend_pack_rel_path)

    week_ahead_block = _render_week_ahead(bundle)
    # Keep optional sections evidence-gated.  Empty templates and low-attention
    # calendar rows do not earn commute or print space.
    macro_rows = bundle.get("macro_agenda", []) or []
    macro_material = any(int(item.get("attention", 0) or 0) >= 3 for item in macro_rows)
    theme = bundle.get("theme_deep_dive", {}) or {}
    theme_evidence = len(theme.get("signals", []) or []) + len(theme.get("news", []) or [])
    theme_ready = bool(theme_evidence >= 2 and (theme.get("related_names", []) or []) and (theme.get("upcoming", []) or []))

    if is_week_ahead:
        transmission_section = f"### 2.1 This Week at a Glance\n{week_ahead_block}\n\n" if week_ahead_block else ""
        local_section = ""
    else:
        transmission_section = (
            "### 2.1 Global-to-Hong Kong Transmission\n"
            f"{non_trading_lens}{_render_overseas_review_block(bundle)}\n\n"
            f"{_render_non_trading_focus(bundle)}\n\n{_render_weekly_review(bundle)}\n\n"
            f"{_render_selected_news(bundle)}\n\n"
            "**AI / TMT hand-off**\n\n"
            f"{ai_tmt_chart_block}{_render_ai_tmt_chain(bundle)}\n\n"
        )
        local_section = (
            "### 2.2 Hong Kong Local Tape and Flow\n"
            f"{_render_hk_review_block(bundle)}\n\n"
            "**Cross-asset attribution**\n\n"
            f"{_render_attribution(bundle)}\n\n"
            "**Local flow evidence**\n\n"
            f"{_render_flow_tracker(bundle)}\n\n"
        )

    macro_section = (
        "### 2.3 Macro Transmission\n"
        f"{_render_macro_takeaway(macro_takeaway)}\n\n"
        f"{_render_macro_watchpoints(llm_sections)}\n\n"
        f"{_render_macro_table(bundle)}\n\n"
        "**Positioning and risk backdrop**\n\n"
        f"{_render_flows(bundle)}\n"
        + "".join(
            f"- Geopolitics: {item.get('region', '')} | {item.get('event', '')} | Impact: {item.get('impact', '')}\n"
            for item in ((bundle.get('risk', {}) or {}).get('geopolitical_risks', []) or [])[:2]
        )
        + "\n"
        if macro_material
        else (
            "### 2.3 Macro Transmission\n"
            "No verified macro event cleared the decision-materiality threshold for the core edition. "
            "Rule-derived monitoring windows remain in the source bundle and catalyst ladder.\n\n"
        )
    )

    company_section = (
        "### 2.4 Company Micro Research and Catalysts\n"
        f"{_render_news_table(bundle)}\n\n"
        f"{_render_company_events(bundle)}\n\n"
        "**Core coverage requiring follow-up**\n\n"
        f"{_render_watchlists(bundle, item_limit=2, story_limit=1, bucket_order=['Core coverage', 'Priority follow-up'])}\n\n"
    )

    questions_section = f"### 2.5 Morning Meeting Questions\n{_render_md_questions(bundle)}\n\n"

    theme_section = f"### 3.2 Conditional Theme Deep Dive\n{_render_theme_deep_dive(bundle)}\n\n" if theme_ready else ""
    chart_head = "3.3" if theme_ready else "3.2"
    trend_head = "3.4" if theme_ready else "3.3"

    daily_chart_section = (
        f"### {chart_head} Daily One Chart\n{daily_chart_block}\n\n" if daily_chart_block else ""
    )
    trend_pack_section = (
        f"### {trend_head} Hong Kong Trend Pack\n{trend_pack_block}\n\n" if trend_pack_block else ""
    )

    report = f"""# Morning Research Workbench | {briefing_date}

> **Commute reading route:** `Layer 1 scan (5 min)` → `Layer 2 deep read` → `Layer 3 and appendix if time allows`
>
> Mode: `{day_mode.get('label', 'Trading day')}` | {day_mode.get('note', '')}

_Data through: US/global `{global_date}` | HK `{hk_date}` | A-share `{cn_date}` | Generated `{meta.get('generated_at', '')}`_

## Executive Summary

{_render_executive_summary(bundle, pulse)}

## {layer_one_title}

### 1.1 {call_title}
{_render_call_scorecard(bundle)}

### 1.2 Opening Decision Board

**Global-to-Hong Kong signals**

{_render_global_asset_dashboard(bundle)}

**Hong Kong local confirmation**

{_render_hk_quick_checks(bundle)}

**Risk state and evidence**

{_render_risk_dashboard(bundle)}

**{checklist_title}**

{_render_top_items(bundle.get('must_watch', []) or [], limit=4)}

## Visual Dashboard

{dashboard_md}{catalyst_radar_section}## Layer 2 | Core Research (20-30 min)

{transmission_section}{local_section}{macro_section}{company_section}{questions_section}## Layer 3 | Session Playbook (8-12 min)

### 3.1 Base Case and Scenario Map
{_render_session_playbook(bundle)}

{theme_section}{daily_chart_section}{trend_pack_section}## Optional Appendix | Traceability and Performance (10-15 min)

### Report Metadata
{appendix_meta_block}

### Report Quality and Validation
{_render_report_quality(bundle)}

### Historical Signal Performance
{_render_performance(bundle)}

### Source Links
{_render_sources(bundle)}

{"### Desk Notes (Internal)\n" + internal_notes + "\n" if internal_notes else ""}
"""
    return _clean_report_spacing(report)
