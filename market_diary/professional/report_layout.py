from __future__ import annotations

from typing import Any, Dict


def _render_market_data_quality(meta: Dict[str, Any]) -> str:
    quality = meta.get("market_quality", {}) or {}
    if not quality:
        return ""

    available = quality.get("available")
    total = quality.get("total")
    fallback_count = len(quality.get("fallback", []) or [])
    stale_count = len(quality.get("stale", []) or [])
    missing_count = len(quality.get("missing", []) or [])

    parts = []
    if isinstance(available, int) and isinstance(total, int) and total > 0:
        parts.append(f"Coverage: `{available}/{total}`")
    if fallback_count:
        parts.append(f"Fallbacks used: `{fallback_count}`")
    if stale_count:
        parts.append(f"Stale items (>1d): `{stale_count}`")
    if missing_count:
        parts.append(f"Missing: `{missing_count}`")

    if not parts:
        return ""
    return "> Market data quality: " + " | ".join(parts)


def _render_date_policy(bundle: Dict[str, Any]) -> str:
    semantics = bundle.get("date_semantics", {}) or {}
    lines = semantics.get("lines", []) or []
    if not lines:
        return ""
    joined = " | ".join(str(line) for line in lines[:3])
    return f"> Date policy: {joined}"


def build_report_layout(bundle: Dict[str, Any], dashboard_rel_path: str = "") -> Dict[str, Any]:
    """Prepare report-wide rendering context and mode-specific section titles."""

    meta = bundle.get("meta", {}) or {}
    overview = bundle.get("overview", {}) or {}
    day_mode = bundle.get("day_mode", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    quality = bundle.get("report_quality", {}) or {}

    is_trading_day = bool(day_mode.get("is_trading_day", True))
    report_mode = day_mode.get("mode", "trading_daily")
    is_weekly_review = report_mode == "weekly_review"
    dashboard_md = f"![Research Dashboard]({dashboard_rel_path})\n" if dashboard_rel_path else ""
    market_quality_line = _render_market_data_quality(meta)
    date_policy_line = _render_date_policy(bundle)
    quality_line = (
        f"> Report quality: `{quality.get('score')}/100` | Grade `{quality.get('grade')}` | Status `{str(quality.get('status', '')).replace('_', ' ')}`\n"
        if quality
        else ""
    )
    pulse = llm_sections.get("one_line_market_pulse") or overview.get("theme", "")
    deep_read_setup = llm_sections.get("deep_read_setup") or (
        f"The overnight tape still looks like a `{overview.get('risk_regime', 'Neutral')}` setup. "
        + " ".join((overview.get("notes", []) or [])[:3])
    )

    if is_weekly_review:
        layer_one_title = "Layer 1 | Weekly Scan (5 min)"
        checklist_title = "Next Week Checklist"
        today_ahead_title = "Next Week Calendar and Reopen Prep"
        overseas_title = "Weekly Cross-Asset Review"
        hk_quick_title = "Hong Kong Weekly Tape Quick Check"
        hk_review_title = "Hong Kong / A-share Weekly Review"
        mode_lens = (
            "> Weekly review mode: synthesize the completed Hong Kong week, retain the main cross-asset and policy lessons, "
            "and convert them into next-week preparation rather than treating Saturday as a fresh cash-market session.\n\n"
        )
    elif is_trading_day:
        layer_one_title = "Layer 1 | Scan (5 min)"
        checklist_title = "Morning Checklist"
        today_ahead_title = "Today Ahead and Trading Calendar"
        overseas_title = "Overnight Overseas Market Review"
        hk_quick_title = "Hong Kong Key Data Quick Check"
        hk_review_title = "Hong Kong / A-share Previous-Day Review"
        mode_lens = ""
    else:
        layer_one_title = "Layer 1 | Reset (5 min)"
        checklist_title = "Next Open Checklist"
        today_ahead_title = "Next Session Outlook and Calendar"
        overseas_title = "Still-Moving Global Financial Actions"
        hk_quick_title = "Hong Kong Last Cash-Tape Quick Check (Reference)"
        hk_review_title = "Last Available Hong Kong / A-share Tape (Reference Only)"
        mode_lens = (
            "> Non-trading mode: treat the cash-market tape below as last-available reference, not a fresh trading signal. "
            "Priority shifts to policy headlines, geopolitics, central-bank repricing, FX/commodities/crypto, corporate actions, and next-open preparation.\n\n"
        )

    return {
        "meta": meta,
        "overview": overview,
        "day_mode": day_mode,
        "llm_sections": llm_sections,
        "dashboard_md": dashboard_md,
        "appendix_meta_block": "".join(
            block for block in (f"{date_policy_line}\n" if date_policy_line else "", f"{market_quality_line}\n" if market_quality_line else "", quality_line) if block
        ),
        "pulse": pulse,
        "deep_read_setup": deep_read_setup,
        "macro_takeaway": llm_sections.get("macro_takeaway", ""),
        "is_trading_day": is_trading_day,
        "report_mode": report_mode,
        "is_weekly_review": is_weekly_review,
        "layer_one_title": layer_one_title,
        "checklist_title": checklist_title,
        "today_ahead_title": today_ahead_title,
        "overseas_title": overseas_title,
        "hk_quick_title": hk_quick_title,
        "hk_review_title": hk_review_title,
        "non_trading_lens": mode_lens,
        "briefing_date": meta.get("briefing_date", meta.get("report_date", "")),
        "review_date": meta.get("review_date", meta.get("report_date", "")),
        "data_through": meta.get("data_through", meta.get("report_date", "")),
        "global_date": meta.get("global_market_date", meta.get("effective_date", meta.get("data_through", meta.get("report_date", "")))),
        "hk_date": meta.get("hk_data_date", meta.get("data_through", meta.get("report_date", ""))),
    }
