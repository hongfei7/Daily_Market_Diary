from __future__ import annotations

from typing import Any, Dict

from market_diary.professional.report_blocks import (
    _render_attribution,
    _render_company_events,
    _render_daily_one_chart,
    _render_executive_summary,
    _render_flow_tracker,
    _render_flows,
    _render_hk_review_block,
    _render_internal_notes,
    _render_macro_table,
    _render_macro_takeaway,
    _render_macro_watchpoints,
    _render_news_table,
    _render_overseas_review_block,
    _render_performance,
    _render_report_quality,
    _render_sources,
    _render_theme_deep_dive,
    _render_today_forward,
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
    _render_weekly_review,
)
from market_diary.professional.report_text import _clean_report_spacing


def render_professional_report(
    bundle: Dict[str, Any],
    charts_section: str,
    dashboard_rel_path: str = "",
    daily_chart_rel_path: str = "",
    trend_pack_rel_path: str = "",
) -> str:
    layout = build_report_layout(bundle, dashboard_rel_path=dashboard_rel_path)
    meta = layout["meta"]
    overview = layout["overview"]
    day_mode = layout["day_mode"]
    llm_sections = layout["llm_sections"]
    dashboard_md = layout["dashboard_md"]
    appendix_meta_block = layout["appendix_meta_block"]
    pulse = layout["pulse"]
    macro_takeaway = layout["macro_takeaway"]
    is_trading_day = layout["is_trading_day"]
    layer_one_title = layout["layer_one_title"]
    checklist_title = layout["checklist_title"]
    today_ahead_title = layout["today_ahead_title"]
    overseas_title = layout["overseas_title"]
    hk_quick_title = layout["hk_quick_title"]
    hk_review_title = layout["hk_review_title"]
    non_trading_lens = layout["non_trading_lens"]
    briefing_date = layout["briefing_date"]
    review_date = layout["review_date"]
    global_date = layout["global_date"]
    hk_date = layout["hk_date"]
    internal_notes = _render_internal_notes(bundle)
    daily_chart_block = _render_daily_one_chart(bundle, daily_chart_rel_path)
    trend_pack_block = _render_trend_pack(bundle, trend_pack_rel_path)
    daily_chart_section = f"### 3.3 Daily One Chart\n{daily_chart_block}\n\n" if daily_chart_block else ""
    trend_pack_section = f"### 3.4 Hong Kong Trend Pack\n{trend_pack_block}\n\n" if trend_pack_block else ""

    report = f"""# Morning Research Workbench | {briefing_date}

> **Commute reading route:** `5-minute decision scan` → `25-30 minute deep read` → `10-15 minute optional appendix`
>
> Mode: `{day_mode.get('label', 'Trading day')}` | {day_mode.get('note', '')}

_Data through: global `{global_date}` | HK/China `{hk_date}` | Market effective `{meta.get('effective_date', '')}` | Generated `{meta.get('generated_at', '')}`_

## Executive Summary

{_render_executive_summary(bundle, pulse)}

## Visual Dashboard

{dashboard_md}## {layer_one_title}

### 1.2 Global Asset Price Dashboard
{_render_global_asset_dashboard(bundle)}

### 1.3 {hk_quick_title}
{_render_hk_quick_checks(bundle)}

### 1.4 Decision Board
{_render_risk_dashboard(bundle)}

**{checklist_title}**

{_render_top_items(bundle.get('must_watch', []) or [], limit=4)}

## Layer 2 | Deep Read (20-30 min)

### 2.1 {overseas_title}
{non_trading_lens}{_render_overseas_review_block(bundle)}

{_render_non_trading_focus(bundle)}

{_render_weekly_review(bundle)}

{_render_selected_news(bundle)}

### 2.2 {hk_review_title}
{non_trading_lens if not is_trading_day else ""}{_render_hk_review_block(bundle)}

### 2.3 Flow Tracker and Attribution
**Cross-Asset Attribution**

{_render_attribution(bundle)}

**Local Flow Tracker**

{_render_flow_tracker(bundle)}

### 2.4 Macro and Policy Tracking
{_render_macro_takeaway(macro_takeaway)}

{_render_macro_watchpoints(llm_sections)}

{_render_macro_table(bundle)}

**Positioning and Risk Backdrop**

{_render_flows(bundle)}
{"".join(f"- Geopolitics: {item.get('region', '')} | {item.get('event', '')} | Impact: {item.get('impact', '')}\n" for item in ((bundle.get('risk', {}) or {}).get('geopolitical_risks', []) or [])[:3])}

### 2.5 Key Company and Sector Events
{_render_news_table(bundle)}

{_render_company_events(bundle)}

### 2.6 Today's Core Names
{_render_watchlists(bundle, item_limit=2, story_limit=1, bucket_order=["Core coverage", "Priority follow-up"])}

## Layer 3 | Decision Deepening (10-15 min)

### 3.1 Rotating Theme Deep Dive
{_render_theme_deep_dive(bundle)}

### 3.2 {today_ahead_title}
{_render_today_forward(bundle)}

{daily_chart_section}{trend_pack_section}## Optional Appendix | Traceability and Performance (10-15 min)

### Report Metadata
{appendix_meta_block}

### Key Questions to Keep in Mind
{"".join(f"- {line}\n" for line in (overview.get('questions', []) or []))}

### Report Quality and Validation
{_render_report_quality(bundle)}

### Historical Signal Performance
{_render_performance(bundle)}

### Source Links
{_render_sources(bundle)}

{"### Desk Notes (Internal)\n" + internal_notes + "\n" if internal_notes else ""}

## Supplementary Visual Appendix

{charts_section}
"""
    return _clean_report_spacing(report)
