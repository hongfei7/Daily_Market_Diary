from __future__ import annotations

import re
from typing import Any, Dict, List

from professional.report_formatting import (
    _fmt_hkd_bn,
    _fmt_millions,
    _fmt_pct,
    _fmt_price,
    _make_table,
    _report_setting,
    _source_as_of,
    _status_label,
    _truncate,
)
from professional.report_layout import build_report_layout
from professional.report_sections import (
    _pick_metrics_by_name,
    _render_global_asset_dashboard,
    _render_hk_quick_checks,
    _render_non_trading_focus,
    _render_risk_dashboard,
    _render_selected_news,
    _render_top_items,
    _render_weekly_review,
)


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`$])", normalized)
    return [part.strip() for part in parts if part.strip()]


def _paragraph_chunks(text: str, max_sentences: int = 2, max_chars: int = 420, limit: int = 3) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        projected_len = current_len + len(sentence) + (1 if current else 0)
        if current and (len(current) >= max_sentences or projected_len > max_chars):
            chunks.append(" ".join(current))
            current = []
            current_len = 0
            if len(chunks) >= limit:
                break
        current.append(sentence)
        current_len += len(sentence) + 1

    if current and len(chunks) < limit:
        chunks.append(" ".join(current))
    return chunks[:limit]


def _compact_bullets(items: List[str], limit: int = 4, width: int = 150) -> List[str]:
    bullets: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            bullets.append(_truncate(text, width))
        if len(bullets) >= limit:
            break
    return bullets


def _clean_report_spacing(text: str) -> str:
    """Keep generated markdown readable without changing table rows."""
    cleaned = re.sub(r"\n{4,}", "\n\n\n", text)
    return cleaned.strip() + "\n"


def _render_macro_table(bundle: Dict[str, Any], limit: int | None = None) -> str:
    rows = (bundle.get("macro_agenda", []) or [])[: limit or _report_setting(bundle, "top_macro_events", 6)]
    table_rows = [
        (
            item.get("time", ""),
            item.get("country", ""),
            item.get("event", ""),
            item.get("status", ""),
            item.get("impact", ""),
            ", ".join(item.get("affected_industries", [])),
            item.get("attention", ""),
        )
        for item in rows
    ]
    return _make_table(["Time", "Country", "Event", "Status", "Impact", "Industries", "Attention"], table_rows)


def _render_news_table(bundle: Dict[str, Any], limit: int | None = None) -> str:
    rows = (bundle.get("sector_digest", {}) or {}).get("graded_news", [])[ : limit or _report_setting(bundle, "top_news_items", 8)]
    table_rows = [
        (
            item.get("grade", ""),
            item.get("sector", ""),
            _truncate(item.get("title", ""), 70),
            _truncate(item.get("why", ""), 74),
            item.get("horizon", ""),
        )
        for item in rows
    ]
    return _make_table(["Grade", "Sector", "Headline", "Why it matters", "Horizon"], table_rows)


def _render_watchlists(bundle: Dict[str, Any], item_limit: int | None = None, story_limit: int | None = None) -> str:
    sections: List[str] = []
    effective_item_limit = item_limit if item_limit is not None else 0
    effective_story_limit = story_limit if story_limit is not None else _report_setting(bundle, "watchlist_story_limit", 2)
    for bucket, items in (bundle.get("watchlists", {}) or {}).items():
        sections.append(f"### {bucket}")
        if not items:
            sections.append("No items were available.\n")
            continue
        visible_items = items[:effective_item_limit] if effective_item_limit > 0 else items
        table_rows = [
            (
                item.get("name", ""),
                item.get("ticker", ""),
                _fmt_price(item.get("last_price")),
                _fmt_pct(item.get("daily_change_pct")),
                item.get("range_label", "N/A"),
                _truncate(item.get("note", ""), 84),
            )
            for item in visible_items
        ]
        sections.append(_make_table(["Name", "Ticker", "Last", "1D", "Range position", "Morning note"], table_rows))
        for item in visible_items:
            news = item.get("recent_news", []) or []
            if news:
                sections.append(f"**Recent headlines for {item.get('name', '')}:**")
                for story in news[:effective_story_limit]:
                    title = story.get("title", "")
                    url = story.get("url", "")
                    source = story.get("source", "")
                    if url:
                        sections.append(f"- [{title}]({url}) ({source})")
                    else:
                        sections.append(f"- {title} ({source})")
        sections.append("")
    return "\n".join(sections).strip()


def _render_flows(bundle: Dict[str, Any]) -> str:
    bullets = (bundle.get("movers_digest", {}) or {}).get("flow_bullets", []) or []
    if not bullets:
        return "- No flow or positioning signals were available."
    flow_tracker_keywords = (
        "Stock Connect",
        "AH premium",
        "HKEX short selling",
        "Highest stock-level short ratios",
        "ETF flow anomalies",
    )
    filtered = [
        item
        for item in bullets
        if not any(keyword.lower() in str(item).lower() for keyword in flow_tracker_keywords)
    ]
    if not filtered:
        return "- Detailed local flow evidence is covered in Section 2.3; keep this section focused on options, hedging, and macro-positioning risk."
    return "\n".join(f"- {item}" for item in filtered[:5])


def _render_attribution(bundle: Dict[str, Any]) -> str:
    attribution = bundle.get("attribution", {}) or {}
    drivers = attribution.get("dominant_drivers", []) or []
    if not drivers:
        return "- No cross-asset attribution drivers were available."
    rows = [
        (
            item.get("name", ""),
            item.get("direction", ""),
            item.get("score", ""),
            _truncate(item.get("evidence", ""), 72),
            _truncate(item.get("implication", ""), 90),
        )
        for item in drivers[:6]
    ]
    return _make_table(["Driver", "Direction", "Score", "Evidence", "HK implication"], rows)


def _render_flow_tracker(bundle: Dict[str, Any]) -> str:
    tracker = bundle.get("flow_tracker", {}) or {}
    if not tracker:
        return "No dedicated flow tracker data were available."

    lines: List[str] = ["##### Flow Takeaways"]
    summary = tracker.get("summary", "Flow evidence was not conclusive.")
    if summary:
        lines.append(f"- {summary}")

    flow_bullets = tracker.get("flow_bullets", []) or []
    if flow_bullets:
        lines.extend(f"- {item}" for item in flow_bullets[:3])
    lines.append("")

    metrics = tracker.get("key_metrics", []) or []
    metrics = _pick_metrics_by_name(
        metrics,
        (
            "Southbound / Northbound net flow",
            "Main Board turnover vs 20D",
            "Short-selling ratio",
            "AH premium index",
            "HIBOR 1M",
            "Aggregate Balance",
        ),
    )
    if metrics:
        rows = [
            (
                item.get("metric", ""),
                item.get("value", ""),
                _status_label(str(item.get("status", ""))),
                _source_as_of(item),
                _truncate(item.get("note", ""), 76),
            )
            for item in metrics
        ]
        lines.append("##### Key Flow / Funding Metrics")
        lines.append(_make_table(["Metric", "Value", "Status", "Source / as of", "Read"], rows))
        lines.append("")

    stock_connect = (tracker.get("stock_connect", {}) or {}).get("data", {}) or {}
    southbound_active = ((stock_connect.get("southbound", {}) or {}).get("top_active", []) or [])[:5]
    lines.append("##### Stock Connect Southbound Active Names")
    if southbound_active:
        rows = [
            (
                item.get("rank", ""),
                item.get("ticker", ""),
                item.get("name", ""),
                _fmt_millions(item.get("buy_turnover"), "HK$"),
                _fmt_millions(item.get("sell_turnover"), "HK$"),
                _fmt_millions(item.get("net_buy"), "HK$") if item.get("net_buy") is not None else "N/A",
            )
            for item in southbound_active
        ]
        lines.append(_make_table(["Rank", "Ticker", "Name", "Buy", "Sell", "Net buy"], rows))
        lines.append("")
    else:
        status = (tracker.get("stock_connect", {}) or {}).get("status", "unavailable")
        lines.append(f"- Southbound active-name detail was not available from the public adapter. Status: `{status}`.")
        lines.append("")

    ah_premium = (tracker.get("ah_premium", {}) or {}).get("data", {}) or {}
    ah_rows = ah_premium.get("top_premium", []) or []
    lines.append("##### AH Premium Dispersion")
    if ah_rows:
        rows = [
            (
                item.get("name", ""),
                item.get("a_ticker", ""),
                item.get("h_ticker", ""),
                f"{item.get('premium_pct', 0):+.2f}%",
                item.get("as_of", ""),
            )
            for item in ah_rows[:5]
        ]
        lines.append(_make_table(["Name", "A ticker", "H ticker", "Premium", "As of"], rows))
        lines.append("")
    else:
        status = (tracker.get("ah_premium", {}) or {}).get("status", "unavailable")
        lines.append(f"- A/H premium dispersion was unavailable from the public quote model. Status: `{status}`.")
        lines.append("")

    watch_hits = tracker.get("short_sell_watchlist_hits", []) or []
    short_ratio = tracker.get("short_sell_top_ratio", []) or []
    short_value = tracker.get("short_sell_top_value", []) or []
    short_rows = watch_hits or short_ratio
    if short_rows:
        lines.append("##### HKEX Short-Selling Watch")
        rows = [
            (
                item.get("ticker") or f"{item.get('code', '')}.HK",
                item.get("name", ""),
                f"{item.get('short_ratio_pct', 0):.2f}%",
                _fmt_hkd_bn(item.get("short_turnover_hkd")),
                _fmt_hkd_bn(item.get("total_turnover_hkd")),
            )
            for item in short_rows[:5]
        ]
        lines.append(_make_table(["Ticker", "Name", "Short ratio", "Short turnover", "Total turnover"], rows))
        lines.append("")

    if short_value:
        leaders = "; ".join(
            f"{item.get('ticker') or item.get('code', '') + '.HK'} {item.get('name', '')} ({_fmt_hkd_bn(item.get('short_turnover_hkd'))})"
            for item in short_value[:4]
        )
        lines.append(f"- **Short-value leaders:** {leaders}.")

    if not southbound_active:
        proxy_table = _render_hk_etf_proxy_table(bundle)
        if "No Hong Kong or offshore-China ETF proxy data" not in proxy_table:
            lines.append("##### ETF Proxy Fallback")
            lines.append("_Use this only when official Stock Connect / local-flow data are incomplete._")
            lines.append(proxy_table)
    return "\n".join(lines).strip()


def _render_hk_etf_proxy_table(bundle: Dict[str, Any]) -> str:
    rows = []
    for item in ((bundle.get("movers_digest", {}) or {}).get("etf_flows", []) or []):
        if item.get("ticker") in {"2800.HK", "2828.HK", "3033.HK", "FXI", "KWEB"}:
            rows.append(
                (
                    item.get("ticker", ""),
                    _fmt_price(item.get("price")),
                    _fmt_pct(item.get("change_pct")),
                    f"{item.get('volume_ratio', 1):.2f}x",
                    item.get("estimated_flow_direction", ""),
                )
            )
    if not rows:
        return "No Hong Kong or offshore-China ETF proxy data were available."
    return _make_table(["Ticker", "Last", "1D", "Volume ratio", "Flow bias"], rows)


def _has_official_stock_connect_flow(bundle: Dict[str, Any]) -> bool:
    tracker = bundle.get("flow_tracker", {}) or {}
    stock_connect = (tracker.get("stock_connect", {}) or {}).get("data", {}) or {}
    southbound = stock_connect.get("southbound", {}) or {}
    return bool(southbound.get("top_active")) or southbound.get("net_buy") is not None


def _render_overseas_review_block(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    overview = bundle.get("overview", {}) or {}
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}

    lines: List[str] = []
    setup = str(llm_sections.get("deep_read_setup", "") or "").strip()
    setup_chunks = _paragraph_chunks(setup, max_sentences=2, max_chars=360, limit=2)
    if setup_chunks:
        lines.append("#### Market Setup")
        lines.extend(setup_chunks)
    else:
        lines.append("#### Market Setup")
        lines.append(str(overview.get("theme", "") or "No LLM overnight setup was available; rely on the dashboard and attribution tables below."))

    drivers = _compact_bullets(llm_sections.get("overnight_drivers", []) or [], limit=4, width=140)
    residual_sentences = _split_sentences(setup)[len(_split_sentences(" ".join(setup_chunks))):]
    if drivers or residual_sentences:
        lines.append("")
        lines.append("#### Key Drivers")
        if drivers:
            lines.extend(f"- {item}" for item in drivers)
        else:
            lines.extend(f"- {_truncate(item, 140)}" for item in residual_sentences[:4])

    hk_implication = str(llm_sections.get("overnight_hk_implication", "") or "").strip()
    hk_lines = _compact_bullets(hk_desk_view.get("lines", []) or [], limit=3, width=140)
    lines.append("")
    lines.append("#### Hong Kong Read-Through")
    leadership = str(hk_desk_view.get("leadership", "") or "").strip()
    if leadership:
        lines.append(f"- **Desk lens:** {leadership}.")
    if hk_implication:
        lines.append(f"- **Opening implication:** {_truncate(hk_implication, 170)}")
    for item in hk_lines:
        lines.append(f"- **Cross-market read:** {item}")
    if not leadership and not hk_implication and not hk_lines:
        lines.append("- Hong Kong read-through was not conclusive from the available data.")

    chart_read = (overview.get("chart_read", {}) or {})
    watch_points = _compact_bullets((chart_read.get("fx", []) or []) + (chart_read.get("assets", []) or []), limit=4, width=150)
    if watch_points:
        lines.append("")
        lines.append("#### Watch Points")
        lines.extend(f"- {item}" for item in watch_points)

    return "\n".join(lines)


def _render_hk_review_block(bundle: Dict[str, Any]) -> str:
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}

    lines: List[str] = []
    setup = str(llm_sections.get("hk_review_setup", "") or "").strip()
    if setup:
        lines.append("#### Local Tape Setup")
        lines.extend(_paragraph_chunks(setup, max_sentences=2, max_chars=360, limit=2))
        lines.append("")

    leadership = hk_desk_view.get("leadership", "")
    lines.append("#### Style and Local Leadership")
    if leadership:
        lines.append(f"- **Style leadership:** {leadership}.")
    else:
        lines.append("- **Style leadership:** Check HSI, HSCEI, and HSTECH first to separate broad-beta, old-economy, and growth leadership.")

    for line in (hk_desk_view.get("lines", []) or [])[:3]:
        lines.append(f"- **Cross-market read:** {line}")

    local_leadership = str(llm_sections.get("hk_local_leadership", "") or "").strip()
    if local_leadership:
        lines.append(f"- **LLM local leadership read:** {local_leadership}")

    lines.append("")
    lines.append("#### Flow Confirmation")
    if _has_official_stock_connect_flow(bundle):
        lines.append("- Official Stock Connect evidence is available; use Section 2.3 to confirm whether local money supports the price action.")
    else:
        lines.append("- Official Stock Connect confirmation is incomplete; treat ETF proxies and price leadership as fallback evidence only.")

    follow_through = str(llm_sections.get("hk_follow_through", "") or "").strip()
    if not follow_through:
        follow_through = "Confirm the opening read through Southbound active names, short-selling concentration, USD/CNH, and USD/HKD funding pressure."
    lines.append("")
    lines.append("#### Follow-Through Checklist")
    lines.append(f"- **Follow-through check:** {follow_through}")

    if not _has_official_stock_connect_flow(bundle):
        proxy_table = _render_hk_etf_proxy_table(bundle)
        if "No Hong Kong or offshore-China ETF proxy data" not in proxy_table:
            lines.append("")
            lines.append("**ETF proxy fallback, only if official local-flow data are incomplete:**")
            lines.append(proxy_table)

    return "\n".join(lines)


def _render_company_events(bundle: Dict[str, Any]) -> str:
    company_events = bundle.get("company_events", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    sections: List[str] = []

    if llm_sections.get("company_takeaway"):
        sections.append(llm_sections.get("company_takeaway", ""))
        sections.append("")

    company_notes = llm_sections.get("company_notes", []) or []
    if company_notes:
        sections.append("#### LLM Quick Takes")
        for item in company_notes[:6]:
            sections.append(f"- **{item.get('ticker', '')}** | {item.get('commentary', '')}")
        sections.append("")

    announcements = company_events.get("announcements", []) or []
    sections.append("#### HKEX Announcements")
    if announcements:
        rows = [
            (
                item.get("grade", ""),
                item.get("ticker", ""),
                item.get("event_type", ""),
                item.get("release_time", ""),
                _truncate(item.get("title", ""), 88),
            )
            for item in announcements[:8]
        ]
        sections.append(_make_table(["Grade", "Ticker", "Type", "Release time", "Title"], rows))
    else:
        sections.append("No HKEX announcement items were available from the public adapter.")
    sections.append("")

    earnings = company_events.get("earnings", []) or []
    sections.append("#### Earnings / Results Watch")
    if earnings:
        rows = [
            (
                item.get("ticker", ""),
                item.get("company", ""),
                item.get("time", ""),
                _truncate(item.get("comparison", ""), 84),
            )
            for item in earnings[:6]
        ]
        sections.append(_make_table(["Ticker", "Company", "Timing", "Expectation framing"], rows))
    else:
        sections.append("No earnings items were available.")

    ratings = company_events.get("ratings", []) or []
    sections.append("\n#### Rating Changes")
    if ratings:
        sections.extend(
            f"- **{item.get('ticker', '')}** | {item.get('firm', '')} | {item.get('action', '')} | {item.get('summary', '')} | PT {item.get('target_change', '')}"
            for item in ratings[:6]
        )
    else:
        sections.append("- No rating-change items were available.")

    sections.append("\n#### IPO Watch")
    sections.append(f"- {company_events.get('ipo_watch', 'No IPO watch items were available.')}")
    return "\n".join(sections)


def _render_theme_deep_dive(bundle: Dict[str, Any]) -> str:
    section = bundle.get("theme_deep_dive", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    rows = []
    for item in section.get("related_names", []) or []:
        rows.append(
            (
                item.get("name", ""),
                item.get("ticker", ""),
                item.get("bucket", ""),
                _truncate(item.get("note", ""), 86),
            )
        )

    blocks = [
        f"- **Theme:** {section.get('theme', '')}",
        f"- **Angle for the day:** {section.get('angle', '')}",
    ]
    if llm_sections.get("theme_paragraph"):
        blocks.append("")
        blocks.append(llm_sections.get("theme_paragraph", ""))
    blocks.extend(
        [
        "- **Signals to keep in mind:**",
        ]
    )
    blocks.extend(f"  - {line}" for line in (section.get("signals", []) or []))
    if llm_sections.get("theme_watch_items"):
        blocks.append("\n**LLM watch items:**")
        blocks.extend(f"- {line}" for line in (llm_sections.get("theme_watch_items", []) or [])[:5])

    if rows:
        blocks.append("\n**Related names to keep close:**")
        blocks.append(_make_table(["Name", "Ticker", "Bucket", "Morning read"], rows))

    upcoming = section.get("upcoming", []) or []
    if upcoming:
        blocks.append("\n**Upcoming catalysts tied to the theme:**")
        blocks.extend(
            f"- {item.get('date', '')} | {item.get('event', '')} | {_truncate(item.get('impact', ''), 100)}"
            for item in upcoming[:4]
        )

    theme_news = section.get("news", []) or []
    if theme_news:
        blocks.append("\n**Relevant headlines:**")
        blocks.extend(
            f"- [{item.get('title', '')}]({item.get('url', '')})" if item.get("url") else f"- {item.get('title', '')}"
            for item in theme_news[:3]
        )

    return "\n".join(blocks)


def _render_today_forward(bundle: Dict[str, Any]) -> str:
    today_forward = bundle.get("today_forward", {}) or {}
    lines = [f"- {line}" for line in (today_forward.get("focus_lines", []) or [])]
    if not lines:
        lines = ["- No same-day focus lines were available."]

    macro_rows = [
        (
            item.get("time", ""),
            item.get("country", ""),
            item.get("event", ""),
            item.get("status", ""),
            _truncate(item.get("detail", ""), 64),
        )
        for item in (today_forward.get("today_macro", []) or [])[:5]
    ]
    catalyst_rows = [
        (
            item.get("date", ""),
            item.get("time", ""),
            item.get("category", ""),
            _truncate(item.get("event", ""), 60),
            item.get("importance", ""),
        )
        for item in (today_forward.get("today_catalysts", []) or [])[:6]
    ]

    next_rows = [
        (
            item.get("date", ""),
            item.get("category", ""),
            _truncate(item.get("event", ""), 62),
            _truncate(item.get("impact", ""), 72),
        )
        for item in (today_forward.get("next_catalysts", []) or [])[:8]
    ]

    output = ["\n".join(lines), "\n**Same-day macro docket**"]
    output.append(_make_table(["Time", "Country", "Event", "Status", "Detail"], macro_rows) if macro_rows else "No same-day macro items were available.")
    output.append("\n**Same-day catalyst list**")
    output.append(_make_table(["Date", "Time", "Category", "Event", "Importance"], catalyst_rows) if catalyst_rows else "No same-day catalysts were available.")
    output.append("\n**Next few sessions**")
    output.append(_make_table(["Date", "Category", "Event", "Impact"], next_rows) if next_rows else "No forward catalysts were available.")
    return "\n".join(output)


def _render_daily_one_chart(bundle: Dict[str, Any], daily_chart_rel_path: str) -> str:
    chart_meta = bundle.get("daily_one_chart", {}) or {}
    chart_path = daily_chart_rel_path or chart_meta.get("rel_path", "")
    if not chart_path:
        return "No dedicated Daily One Chart image was available. This section should not reuse the Visual Dashboard."

    title = chart_meta.get("title", "Daily One Chart")
    caption = chart_meta.get("caption", "")
    source = chart_meta.get("source", "")
    source_line = f"\n\n_Source: {source}_" if source else ""
    return f"**{title}**\n\n![Daily One Chart]({chart_path})\n\n{caption}{source_line}"


def _render_trend_pack(bundle: Dict[str, Any], trend_pack_rel_path: str) -> str:
    chart_meta = bundle.get("trend_pack", {}) or {}
    chart_path = trend_pack_rel_path or chart_meta.get("rel_path", "")
    if not chart_path:
        return "No dedicated Hong Kong Trend Pack image was available."

    title = chart_meta.get("title", "Hong Kong Trend Pack")
    caption = chart_meta.get("caption", "")
    source = chart_meta.get("source", "")
    source_line = f"\n\n_Source: {source}_" if source else ""
    return f"**{title}**\n\n![Hong Kong Trend Pack]({chart_path})\n\n{caption}{source_line}"


def _render_reflection_area(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    prompts = bundle.get("reflection_prompts", []) or []
    interview_answer = llm_sections.get("interview_answer", "")
    risk_check = llm_sections.get("risk_check", "")

    lines = []
    if llm_sections.get("thinking_note"):
        lines.append(f"- **Thinking note:** {llm_sections.get('thinking_note', '')}")
    if interview_answer:
        lines.append(f"- **Suggested two-sentence market answer:** {interview_answer}")
    if risk_check:
        lines.append(f"- **What could break the view:** {risk_check}")
    for prompt in prompts[:5]:
        lines.append(f"- {prompt}")
    lines.append("- My own view after the commute:")
    lines.append("- What changed versus yesterday:")
    lines.append("- Which name / sector deserves extra prep before the open:")
    return "\n".join(lines)


def _render_sources(bundle: Dict[str, Any], limit: int | None = None) -> str:
    links = (bundle.get("source_links", []) or [])[: limit or _report_setting(bundle, "top_source_links", 15)]
    if not links:
        return "- No traceable links were available."
    lines = []
    for item in links:
        if item.get("url"):
            lines.append(f"- [{item.get('label', '')}]({item.get('url')}) ({item.get('source', '')})")
        else:
            lines.append(f"- {item.get('label', '')} ({item.get('source', '')})")
    return "\n".join(lines)


def _render_report_quality(bundle: Dict[str, Any]) -> str:
    quality = bundle.get("report_quality", {}) or {}
    fact_check = bundle.get("fact_check", {}) or {}
    if not quality:
        return "Report-quality diagnostics were not available."

    lines = [
        f"- **Quality score:** {quality.get('score', 'N/A')}/100 | **Grade:** {quality.get('grade', 'N/A')} | **Status:** {str(quality.get('status', 'N/A')).replace('_', ' ')}",
    ]

    components = quality.get("components", []) or []
    if components:
        rows = [
            (
                item.get("name", ""),
                item.get("score", ""),
                item.get("weight", ""),
                _truncate(str(item.get("read", "")).replace("|", "/"), 86),
            )
            for item in components
        ]
        lines.append(_make_table(["Component", "Score", "Weight", "Read"], rows))

    warnings = quality.get("warnings", []) or []
    if warnings:
        lines.append("\n**Quality warnings**")
        lines.extend(f"- {item}" for item in warnings[:6])

    if fact_check:
        lines.append("\n**LLM fact-check guardrail**")
        lines.append(f"- {fact_check.get('summary', 'No fact-check summary was available.')}")
        mismatches = fact_check.get("numeric_mismatches", []) or []
        logic_warnings = fact_check.get("logic_warnings", []) or []
        if mismatches:
            rows = [
                (
                    item.get("field", ""),
                    item.get("label", ""),
                    item.get("claimed", ""),
                    item.get("expected", ""),
                    _truncate(item.get("snippet", ""), 70),
                )
                for item in mismatches[:6]
            ]
            lines.append(_make_table(["Field", "Claim", "Claimed", "Expected", "Snippet"], rows))
        if logic_warnings:
            lines.extend(f"- Logic warning: {item.get('message', '')}" for item in logic_warnings[:6])

    adapter_rows = quality.get("adapter_status", []) or []
    if adapter_rows:
        lines.append("\n**Adapter status**")
        lines.append(_make_table(["Adapter", "Status"], [(item.get("name", ""), item.get("status", "")) for item in adapter_rows]))

    return "\n".join(lines)


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
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    llm_sections = layout["llm_sections"]
    dashboard_md = layout["dashboard_md"]
    market_quality_block = layout["market_quality_block"]
    date_policy_block = layout["date_policy_block"]
    quality_line = layout["quality_line"]
    pulse = layout["pulse"]
    deep_read_setup = layout["deep_read_setup"]
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

    report = f"""# Morning Research Workbench | {briefing_date}

> Designed for a Hong Kong sell-side commute: Layer 1 `Scan`, Layer 2 `Deep Read`, Layer 3 `Thinking`  
> Mode: `{day_mode.get('label', 'Trading day')}` | {day_mode.get('note', '')}
> Briefing date: `{briefing_date}` | Review date: `{review_date}` | Global request: `{global_date}` | HK/China request: `{hk_date}` | Market effective date: `{meta.get('effective_date', '')}` | Generated at: `{meta.get('generated_at', '')}`
{date_policy_block}
{market_quality_block}
{quality_line}

## Visual Dashboard

{dashboard_md}## {layer_one_title}

### 1.1 One-Line Market Pulse
{pulse}

### 1.2 Global Asset Price Dashboard
{_render_global_asset_dashboard(bundle)}

### 1.3 {hk_quick_title}
{_render_hk_quick_checks(bundle)}

### 1.4 Risk Dashboard
{_render_risk_dashboard(bundle)}

### 1.5 {checklist_title}
{_render_top_items(bundle.get('must_watch', []) or [], limit=5)}

## Layer 2 | Deep Read (20-30 min)

### 2.1 {overseas_title}
{non_trading_lens}{_render_overseas_review_block(bundle)}

{_render_non_trading_focus(bundle)}

{_render_weekly_review(bundle)}

{_render_selected_news(bundle)}

### 2.2 {hk_review_title}
{non_trading_lens if not is_trading_day else ""}{_render_hk_review_block(bundle)}

### 2.3 Flow Tracker and Attribution
#### Cross-Asset Attribution v1
{_render_attribution(bundle)}

#### Flow Tracker
{_render_flow_tracker(bundle)}

### 2.4 Macro and Policy Tracking
{macro_takeaway}

{"".join(f"- Watchpoint: {line}\n" for line in (llm_sections.get('macro_watchpoints', []) or []))}
{_render_macro_table(bundle)}

#### Positioning and Risk Backdrop
{_render_flows(bundle)}
{"".join(f"- Geopolitics: {item.get('region', '')} | {item.get('event', '')} | Impact: {item.get('impact', '')}\n" for item in ((bundle.get('risk', {}) or {}).get('geopolitical_risks', []) or [])[:3])}

### 2.5 Key Company and Sector Events
{_render_news_table(bundle)}

{_render_company_events(bundle)}

### 2.6 Coverage Pools
{_render_watchlists(bundle)}

## Layer 3 | Thinking (10-15 min)

### 3.1 Rotating Theme Deep Dive
{_render_theme_deep_dive(bundle)}

### 3.2 {today_ahead_title}
{_render_today_forward(bundle)}

### 3.3 Daily One Chart
{_render_daily_one_chart(bundle, daily_chart_rel_path)}

### 3.4 Hong Kong Trend Pack
{_render_trend_pack(bundle, trend_pack_rel_path)}

### 3.5 Personal View Pad
{_render_reflection_area(bundle)}

## Traceable Appendix

### Key Questions to Keep in Mind
{"".join(f"- {line}\n" for line in (overview.get('questions', []) or []))}

### Report Quality and Validation
{_render_report_quality(bundle)}

### Source Links
{_render_sources(bundle)}

## Supplementary Visual Appendix

{charts_section}
"""
    return _clean_report_spacing(report)
