from __future__ import annotations

from typing import Any, Dict, List

from market_diary.professional.report_formatting import (
    _compact_source_as_of,
    _fmt_hkd_bn,
    _fmt_millions,
    _fmt_pct,
    _fmt_price,
    _make_table,
    _report_flag,
    _report_setting,
    _status_label,
    _truncate,
)
from market_diary.professional.report_sections import _pick_metrics_by_name, _resolved_hk_leadership, _safe_sentence_clip
from market_diary.professional.report_text import (
    _compact_bullets,
    _condense_sentence,
    _render_labeled_paragraphs,
    _render_labeled_points,
)


def _render_macro_table(bundle: Dict[str, Any], limit: int | None = None) -> str:
    rows = (bundle.get("macro_agenda", []) or [])[: limit or _report_setting(bundle, "top_macro_events", 6)]
    if not rows:
        return "The macro calendar was light for this run."
    table_rows = [
        (
            item.get("time", ""),
            item.get("country", ""),
            _truncate(item.get("event", ""), 74),
            item.get("status", ""),
            _truncate(
                " | ".join(
                    str(part)
                    for part in (
                        f"Impact: {item.get('impact', '')}" if item.get("impact") else "",
                        f"Attention: {item.get('attention')}/5" if item.get("attention") not in (None, "") else "",
                        f"Industries: {', '.join(item.get('affected_industries', []))}" if item.get("affected_industries") else "",
                    )
                    if part
                ),
                170,
            ),
        )
        for item in rows
    ]
    return _make_table(["Time", "Region", "Event", "Status", "Desk read"], table_rows)


def _render_executive_summary(bundle: Dict[str, Any], pulse: str) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    must_watch = bundle.get("must_watch", []) or []
    leadership = _resolved_hk_leadership(bundle)
    lines: List[str] = []

    pulse_text = _safe_sentence_clip(pulse, 190)
    if pulse_text:
        lines.append(f"- **Market pulse:** {pulse_text}")

    hk_implication = _safe_sentence_clip(llm_sections.get("overnight_hk_implication", ""), 180)
    if hk_implication:
        lines.append(f"- **Hong Kong lens:** {leadership}. {hk_implication}")
    elif leadership:
        lines.append(f"- **Hong Kong lens:** {leadership}.")

    confirmation = _safe_sentence_clip(llm_sections.get("hk_follow_through", ""), 175)
    if not confirmation:
        focus_lines = ((bundle.get("today_forward", {}) or {}).get("focus_lines", []) or [])
        confirmation = _safe_sentence_clip(focus_lines[0] if focus_lines else "", 175)
    if confirmation:
        lines.append(f"- **What would confirm it:** {confirmation}")

    invalidation = _safe_sentence_clip(llm_sections.get("risk_check", ""), 175)
    if invalidation:
        lines.append(f"- **What could break it:** {invalidation}")

    if len(lines) < 3:
        for item in must_watch:
            title = _safe_sentence_clip(item.get("title", ""), 90)
            summary = _safe_sentence_clip(item.get("summary", ""), 130)
            if title and summary:
                lines.append(f"- **Priority evidence:** {title} — {summary}")
                break

    if not lines:
        return "- **Executive summary pending:** rely on the section headlines and key tables below."
    return "\n".join(lines)


def _render_news_table(bundle: Dict[str, Any], limit: int | None = None) -> str:
    raw_rows = (bundle.get("sector_digest", {}) or {}).get("graded_news", []) or []
    rows = [
        item
        for item in raw_rows
        if item.get("grade") in {"A", "B"}
        and (str(item.get("sector", "")).lower() != "other" or float(item.get("score", 0) or 0) >= 4.0)
    ][: limit or _report_setting(bundle, "top_news_items", 8)]
    if not rows:
        return "No high-conviction sector story cleared the main-report relevance gate for this run."
    table_rows = [
        (
            item.get("grade", ""),
            item.get("sector", ""),
            _truncate(item.get("title", ""), 70, suffix=""),
            _truncate(item.get("why", ""), 74, suffix=""),
            item.get("horizon", ""),
        )
        for item in rows
    ]
    return _make_table(["Grade", "Sector", "Headline", "Why it matters", "Horizon"], table_rows)


def _render_watchlists(
    bundle: Dict[str, Any],
    item_limit: int | None = None,
    story_limit: int | None = None,
    bucket_order: List[str] | None = None,
) -> str:
    sections: List[str] = []
    effective_item_limit = item_limit if item_limit is not None else _report_setting(bundle, "quick_watchlist_items_per_bucket", 2)
    effective_story_limit = story_limit if story_limit is not None else _report_setting(bundle, "watchlist_story_limit", 2)
    watchlists = bundle.get("watchlists", {}) or {}
    ordered_buckets = bucket_order or list(watchlists.keys())
    for bucket in ordered_buckets:
        items = watchlists.get(bucket, []) or []
        sections.append(f"#### {bucket}")
        if not items:
            sections.append("No names were highlighted in this bucket for the current run.\n")
            continue
        visible_items = items[:effective_item_limit] if effective_item_limit > 0 else items
        has_quote_detail = any(item.get("last_price") is not None or item.get("daily_change_pct") is not None for item in visible_items)
        if has_quote_detail:
            table_rows = [
                (
                    item.get("name", ""),
                    item.get("ticker", ""),
                    _fmt_price(item.get("last_price")),
                    _fmt_pct(item.get("daily_change_pct")),
                    item.get("range_label", "N/A"),
                    _safe_sentence_clip(item.get("note", ""), 110),
                )
                for item in visible_items
            ]
            sections.append(_make_table(["Name", "Ticker", "Last", "1D", "Range position", "Morning note"], table_rows))
        else:
            table_rows = [
                (
                    item.get("name", ""),
                    item.get("ticker", ""),
                    _safe_sentence_clip(item.get("note", ""), 120),
                )
                for item in visible_items
            ]
            sections.append("_Quote fields were not refreshed for this bucket; the table keeps only coverage and action notes._")
            sections.append("")
            sections.append(_make_table(["Name", "Ticker", "Morning note"], table_rows))
        inserted_headline_gap = False
        for item in visible_items:
            news = item.get("recent_news", []) or []
            if news:
                if not inserted_headline_gap:
                    sections.append("")
                    inserted_headline_gap = True
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
        return "- Flow and positioning detail was limited in this run."
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
        return "- Cross-asset attribution did not surface a dominant driver set for this run."
    rows = [
        (
            item.get("name", ""),
            item.get("direction", ""),
            item.get("score", ""),
            _truncate(item.get("evidence", ""), 72, suffix=""),
            _truncate(item.get("implication", ""), 90, suffix=""),
        )
        for item in drivers[:6]
    ]
    return _make_table(["Driver", "Direction", "Score", "Evidence", "HK implication"], rows)


def _render_flow_tracker(bundle: Dict[str, Any]) -> str:
    tracker = bundle.get("flow_tracker", {}) or {}
    if not tracker:
        return "Dedicated local-flow detail was limited for this run."

    lines: List[str] = ["##### Flow Takeaways"]
    summary = tracker.get("summary", "Flow evidence was mixed rather than decisive.")
    if summary:
        lines.append(f"- {summary}")

    flow_bullets = tracker.get("flow_bullets", []) or []
    if flow_bullets:
        lines.extend(f"- {item}" for item in flow_bullets[:3])
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
        lines.append(f"- Southbound active-name detail was not disclosed in the current public feed. Status: `{status}`.")
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
        lines.append(f"- A/H premium dispersion was not refreshed in the current public quote set. Status: `{status}`.")
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
            lines.append("##### ETF Proxy Read")
            lines.append("_Use this only when official Stock Connect / local-flow detail is incomplete._")
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
        return "ETF proxy detail was not available for this run."
    return _make_table(["Ticker", "Last", "1D", "Volume ratio", "Flow bias"], rows)


def _has_official_stock_connect_flow(bundle: Dict[str, Any]) -> bool:
    tracker = bundle.get("flow_tracker", {}) or {}
    stock_connect = (tracker.get("stock_connect", {}) or {}).get("data", {}) or {}
    southbound = stock_connect.get("southbound", {}) or {}
    return bool(southbound.get("top_active")) or southbound.get("net_buy") is not None


def _is_low_signal_market_line(line: str) -> bool:
    text = str(line or "").strip()
    if not text:
        return True
    na_count = text.count("N/A")
    return na_count >= 2 or "last traded around N/A" in text


def _compact_hk_read_lines(lines: List[str], limit: int = 3) -> tuple[List[str], int]:
    visible = [line for line in lines if not _is_low_signal_market_line(line)]
    suppressed = max(0, len(lines) - len(visible))
    return visible[:limit], suppressed


def _render_overseas_review_block(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    overview = bundle.get("overview", {}) or {}
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}

    lines: List[str] = []
    setup = str(llm_sections.get("deep_read_setup", "") or "").strip()
    lines.append("#### Market Setup")
    lines.extend(
        _render_labeled_paragraphs(
            setup,
            ["Core tape", "Main driver", "HK relevance"],
            fallback=str(overview.get("theme", "") or "Use the dashboard and attribution tables below as the primary read."),
            limit=3,
            width=225,
        )
    )

    drivers = _compact_bullets(llm_sections.get("overnight_drivers", []) or [], limit=4, width=140)
    if drivers:
        lines.append("")
        lines.append("#### Key Drivers")
        lines.extend(f"- {item}" for item in drivers)

    hk_implication = str(llm_sections.get("overnight_hk_implication", "") or "").strip()
    hk_lines, suppressed_hk_lines = _compact_hk_read_lines(_compact_bullets(hk_desk_view.get("lines", []) or [], limit=5, width=140), limit=3)
    lines.append("")
    lines.append("#### Hong Kong Read-Through")
    leadership = str(hk_desk_view.get("leadership", "") or "").strip()
    if leadership:
        lines.append(f"**Desk lens.** {leadership}.")
    if hk_implication:
        lines.append(f"**Opening implication.** {_condense_sentence(hk_implication, 260)}")
    for item in hk_lines:
        lines.append(f"- **Cross-market read:** {item}")
    if suppressed_hk_lines:
        lines.append(
            "- **Coverage note:** Hong Kong quote coverage was limited, so low-signal index / FX lines are suppressed; use Stock Connect and A/H premium as the firmer evidence."
        )
    if not leadership and not hk_implication and not hk_lines:
        lines.append("Hong Kong read-through remains tentative on the available evidence.")

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
        lines.extend(
            _render_labeled_paragraphs(
                setup,
                ["Local read", "Verification point", "Risk to watch"],
                limit=3,
                width=220,
            )
        )
        lines.append("")

    leadership = _resolved_hk_leadership(bundle)
    lines.append("#### Style and Local Leadership")
    if leadership:
        lines.append(f"**Style leadership.** {leadership}.")
    else:
        lines.append("**Style leadership.** Check HSI, HSCEI, and HSTECH first to separate broad-beta, old-economy, and growth leadership.")

    hk_lines, suppressed_hk_lines = _compact_hk_read_lines(hk_desk_view.get("lines", []) or [], limit=3)
    for line in hk_lines:
        lines.append(f"- **Cross-market read:** {line}")
    if suppressed_hk_lines:
        lines.append(
            "- **Coverage note:** Low-signal index / FX lines were suppressed because quote coverage was incomplete; flow confirmation below carries more weight for this run."
        )

    lines.append("")
    lines.append("#### Flow Confirmation")
    if _has_official_stock_connect_flow(bundle):
        lines.append("Official Stock Connect evidence is available; use Section 2.3 to confirm whether local money supports the price action.")
    else:
        lines.append("Official Stock Connect confirmation is incomplete; use ETF proxies and price leadership only as secondary evidence.")

    follow_through = str(llm_sections.get("hk_follow_through", "") or "").strip()
    if not follow_through:
        follow_through = "Confirm the opening read through Southbound active names, short-selling concentration, USD/CNH, and USD/HKD funding pressure."
    lines.append("")
    lines.append("#### Follow-Through Checklist")
    lines.append(f"**Follow-through check.** {_condense_sentence(follow_through, 260)}")

    if not _has_official_stock_connect_flow(bundle):
        proxy_table = _render_hk_etf_proxy_table(bundle)
        if "No Hong Kong or offshore-China ETF proxy data" not in proxy_table:
            lines.append("")
            lines.append("**ETF proxy read, only if official local-flow data are incomplete:**")
            lines.append(proxy_table)

    return "\n".join(lines)


def _render_company_events(bundle: Dict[str, Any]) -> str:
    company_events = bundle.get("company_events", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    sections: List[str] = []

    if llm_sections.get("company_takeaway"):
        sections.append("#### Company Event Takeaway")
        sections.extend(
            _render_labeled_paragraphs(
                llm_sections.get("company_takeaway", ""),
                ["Desk read", "Why it matters", "Follow-up"],
                limit=3,
                width=220,
            )
        )
        sections.append("")

    company_notes = llm_sections.get("company_notes", []) or []
    if company_notes:
        sections.append("#### LLM Quick Takes")
        for item in company_notes[:6]:
            sections.append(f"- **{item.get('ticker', '')}** | {_truncate(item.get('commentary', ''), 170, suffix='')}")
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
                _truncate(item.get("title", ""), 88, suffix=""),
            )
            for item in announcements[:8]
        ]
        sections.append(_make_table(["Grade", "Ticker", "Type", "Release time", "Title"], rows))
    else:
        sections.append("No material HKEX announcement items were captured in the current public feed.")
    sections.append("")

    earnings = company_events.get("earnings", []) or []
    sections.append("#### Earnings / Results Watch")
    if earnings:
        rows = [
            (
                item.get("ticker", ""),
                item.get("company", ""),
                item.get("time", ""),
                _truncate(item.get("comparison", ""), 84, suffix=""),
            )
            for item in earnings[:6]
        ]
        sections.append(_make_table(["Ticker", "Company", "Timing", "Expectation framing"], rows))
    else:
        sections.append("No earnings items were scheduled in the current window.")

    ratings = company_events.get("ratings", []) or []
    sections.append("\n#### Rating Changes")
    if ratings:
        sections.extend(
            f"- **{item.get('ticker', '')}** | {item.get('firm', '')} | {item.get('action', '')} | {item.get('summary', '')} | PT {item.get('target_change', '')}"
            for item in ratings[:6]
        )
    else:
        sections.append("- No rating-change items were highlighted in the current sell-side feed.")

    sections.append("\n#### IPO Watch")
    sections.append(f"- {company_events.get('ipo_watch', 'IPO monitoring was not part of this run.')}")
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
                _safe_sentence_clip(item.get("note", ""), 105),
            )
        )

    blocks = [
        _make_table(
            ["Field", "Read"],
            [
                ("Theme", section.get("theme", "") or "No rotating theme was set for this run"),
                ("Angle for the day", _truncate(section.get("angle", "") or "No dedicated theme angle was set for this run.", 170, suffix="")),
            ],
        )
    ]
    if llm_sections.get("theme_paragraph"):
        blocks.append("\n**Theme desk read**")
        blocks.extend(
            _render_labeled_paragraphs(
                llm_sections.get("theme_paragraph", ""),
                ["Core thesis", "Evidence", "What to test"],
                limit=3,
                width=225,
            )
        )

    signals = section.get("signals", []) or []
    if signals:
        blocks.append("\n**Signals to keep in mind**")
        blocks.extend(f"- {_truncate(line, 140, suffix='')}" for line in signals[:5])
    if llm_sections.get("theme_watch_items"):
        blocks.append("\n**LLM watch items:**")
        blocks.extend(f"- {_truncate(line, 150, suffix='')}" for line in (llm_sections.get("theme_watch_items", []) or [])[:5])

    if rows:
        blocks.append("\n**Related names to keep close:**")
        blocks.append(_make_table(["Name", "Ticker", "Bucket", "Morning read"], rows))

    upcoming = section.get("upcoming", []) or []
    if upcoming:
        blocks.append("\n**Upcoming catalysts tied to the theme:**")
        blocks.extend(
            f"- {item.get('date', '')} | {item.get('event', '')} | {_truncate(item.get('impact', ''), 100, suffix='')}"
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
        lines = ["- The session does not carry a separate same-day focus overlay beyond the catalyst tables below."]

    macro_rows = [
        (
            item.get("time", ""),
            item.get("country", ""),
            item.get("event", ""),
            item.get("status", ""),
            _truncate(item.get("detail", ""), 64, suffix=""),
        )
        for item in (today_forward.get("today_macro", []) or [])[:5]
    ]
    seen_events = set()
    catalyst_rows = [
        (
            item.get("date", ""),
            item.get("time", ""),
            item.get("category", ""),
            _truncate(item.get("event", ""), 60, suffix=""),
            item.get("importance", ""),
        )
        for item in (today_forward.get("today_catalysts", []) or [])
        if item.get("event") and not (item.get("event") in seen_events or seen_events.add(item.get("event")))
    ]
    catalyst_rows = catalyst_rows[:4]

    seen_next = set()
    next_rows = [
        (
            item.get("date", ""),
            item.get("category", ""),
            _truncate(item.get("event", ""), 62, suffix=""),
            _truncate(item.get("impact", ""), 72, suffix=""),
        )
        for item in (today_forward.get("next_catalysts", []) or [])
        if item.get("event") and not (item.get("event") in seen_next or seen_next.add(item.get("event")))
    ]
    next_rows = next_rows[:4]

    output = ["\n".join(lines), "\n**Same-day macro docket**"]
    output.append(_make_table(["Time", "Country", "Event", "Status", "Detail"], macro_rows) if macro_rows else "No same-day macro items were scheduled.")
    output.append("\n**Same-day catalyst list**")
    output.append(_make_table(["Date", "Time", "Category", "Event", "Importance"], catalyst_rows) if catalyst_rows else "No same-day catalysts were highlighted.")
    output.append("\n**Next few sessions**")
    output.append(_make_table(["Date", "Category", "Event", "Impact"], next_rows) if next_rows else "No forward catalyst list was populated.")
    return "\n".join(output)


def _render_macro_takeaway(text: str) -> str:
    points = _render_labeled_paragraphs(
        text,
        ["Desk read", "Market sensitivity", "HK implication"],
        fallback="Use the calendar table and released data below as the primary macro read.",
        limit=3,
        width=230,
    )
    return "\n\n".join(points)


def _render_macro_watchpoints(llm_sections: Dict[str, Any]) -> str:
    watchpoints = llm_sections.get("macro_watchpoints", []) or []
    if not watchpoints:
        return ""
    lines = ["**Macro watchpoints**"]
    lines.extend(f"- {_truncate(line, 145, suffix='')}" for line in watchpoints[:4])
    return "\n".join(lines)


def _render_daily_one_chart(bundle: Dict[str, Any], daily_chart_rel_path: str) -> str:
    chart_meta = bundle.get("daily_one_chart", {}) or {}
    chart_path = daily_chart_rel_path or chart_meta.get("rel_path", "")
    if not chart_path:
        return ""

    title = chart_meta.get("title", "Daily One Chart")
    caption = chart_meta.get("caption", "")
    source = chart_meta.get("source", "")
    source_line = f"\n\n_Source: {source}_" if source else ""
    caption_lines = "\n".join(
        _render_labeled_paragraphs(
            caption,
            ["Chart read", "Why it matters"],
            limit=2,
            width=220,
        )
    )
    return f"**{title}**\n\n![Daily One Chart]({chart_path})\n\n{caption_lines}{source_line}"


def _render_trend_pack(bundle: Dict[str, Any], trend_pack_rel_path: str) -> str:
    chart_meta = bundle.get("trend_pack", {}) or {}
    chart_path = trend_pack_rel_path or chart_meta.get("rel_path", "")
    if not chart_path:
        return ""

    title = chart_meta.get("title", "Hong Kong Trend Pack")
    caption = chart_meta.get("caption", "")
    source = chart_meta.get("source", "")
    source_line = f"\n\n_Source: {source}_" if source else ""
    caption_lines = "\n".join(
        _render_labeled_paragraphs(
            caption,
            ["Trend read", "Why it matters"],
            limit=2,
            width=220,
        )
    )
    return f"**{title}**\n\n![Hong Kong Trend Pack]({chart_path})\n\n{caption_lines}{source_line}"


def _render_reflection_area(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    prompts = bundle.get("reflection_prompts", []) or []
    interview_answer = llm_sections.get("interview_answer", "")
    risk_check = llm_sections.get("risk_check", "")

    lines = []
    if llm_sections.get("thinking_note"):
        lines.extend(
            _render_labeled_points(
                llm_sections.get("thinking_note", ""),
                ["Thinking note", "Action"],
                limit=2,
                width=170,
            )
        )
    if interview_answer:
        lines.extend(
            _render_labeled_points(
                interview_answer,
                ["Suggested market answer", "Second sentence"],
                limit=2,
                width=170,
            )
        )
    if risk_check:
        lines.extend(
            _render_labeled_points(
                risk_check,
                ["What could break the view", "Risk trigger"],
                limit=2,
                width=170,
            )
        )
    for prompt in prompts[:5]:
        lines.append(f"- {prompt}")
    lines.append("- My own view after the commute:")
    lines.append("- What changed versus yesterday:")
    lines.append("- Which name / sector deserves extra prep before the open:")
    return "\n".join(lines)


def _render_internal_notes(bundle: Dict[str, Any]) -> str:
    if not _report_flag(bundle, "show_internal_reflection", False):
        return ""
    return _render_reflection_area(bundle)


def _render_sources(bundle: Dict[str, Any], limit: int | None = None) -> str:
    links = (bundle.get("source_links", []) or [])[: limit or _report_setting(bundle, "top_source_links", 15)]
    if not links:
        return "- Source links were not attached for this run."
    lines = []
    for item in links:
        if item.get("url"):
            lines.append(f"- [{item.get('label', '')}]({item.get('url')}) ({item.get('source', '')})")
        else:
            lines.append(f"- {item.get('label', '')} ({item.get('source', '')})")
    return "\n".join(lines)


def _percent(value: Any, digits: int = 1) -> str:
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return "N/A"


def _render_performance(bundle: Dict[str, Any]) -> str:
    performance = bundle.get("performance", {}) or {}
    if not performance or performance.get("status") in {"disabled", "error"}:
        return "Historical signal diagnostics were unavailable for this run."

    data_quality = performance.get("data_quality", {}) or {}
    lines = [
        f"- **Readiness:** {str(performance.get('status', 'unknown')).replace('_', ' ')} | "
        f"{data_quality.get('observations', 0)} market observations | "
        f"{data_quality.get('active_signal_dates', 0)} active signal dates",
        "- **Execution rule:** use only the next available close after publication; 10 bps turnover cost by default. Current-day signals never receive same-day returns.",
    ]
    rows = []
    for name, payload in (performance.get("benchmarks", {}) or {}).items():
        metrics = payload.get("metrics", {}) or {}
        rows.append(
            (
                name,
                metrics.get("sessions", 0),
                _percent(metrics.get("cumulative_return_net")),
                _percent(metrics.get("benchmark_return")),
                _percent(metrics.get("excess_return")),
                _percent(metrics.get("max_drawdown")),
                _percent(metrics.get("hit_rate_active_sessions")),
            )
        )
    if rows:
        lines.append(
            _make_table(
                ["Benchmark", "Sessions", "Signal net", "Buy & hold", "Excess", "Max drawdown", "Hit rate"],
                rows,
            )
        )

    hsi_events = ((performance.get("benchmarks", {}) or {}).get("Hang Seng Index", {}) or {}).get("event_horizons", {}) or {}
    if hsi_events:
        event_rows = [
            (
                f"{horizon} sessions",
                item.get("resolved_signals", 0),
                _percent(item.get("hit_rate")),
                _percent(item.get("average_directional_return_net")),
            )
            for horizon, item in hsi_events.items()
        ]
        lines.append("\n**Hang Seng event-horizon diagnostics**")
        lines.append(_make_table(["Horizon", "Resolved", "Hit rate", "Average net return"], event_rows))

    rel_path = str(performance.get("rel_path", "") or "")
    if rel_path:
        lines.append(f"\n![Published signal performance]({rel_path})")
    conflicts = data_quality.get("conflicts", []) or []
    exclusions = data_quality.get("excluded_non_session_observations", []) or []
    if conflicts or exclusions:
        lines.append(
            f"- **Data-quality caveat:** {len(conflicts)} conflicting historical price revision(s) and "
            f"{len(exclusions)} weekend pseudo-session observation(s) were handled explicitly; inspect the ledger before relying on the result."
        )
    lines.append(
        "- **Use boundary:** this is a close-to-close research diagnostic, not an executable strategy or investment recommendation; dividends, financing, borrow and market impact are excluded."
    )
    return "\n".join(lines)


def _render_source_health(bundle: Dict[str, Any]) -> str:
    health = bundle.get("source_health", {}) or {}
    if not health:
        return "- Source-health diagnostics were not attached."
    coverage = health.get("coverage", {}) or {}
    lines = [
        f"- **Source health:** {str(health.get('status', 'unknown')).replace('_', ' ')} | "
        f"{coverage.get('healthy', 0)} healthy | {coverage.get('degraded', 0)} degraded | "
        f"{coverage.get('unavailable', 0)} unavailable"
    ]
    attention = [item for item in (health.get("sources", []) or []) if item.get("critical") or item.get("status") != "healthy"]
    if attention:
        lines.append(
            _make_table(
                ["Source", "Critical", "Status", "Score", "Freshest age", "Policy"],
                [
                    (
                        item.get("source", ""),
                        "Yes" if item.get("critical") else "No",
                        item.get("status", ""),
                        item.get("score", ""),
                        f"{item.get('freshest_age_days')}d" if item.get("freshest_age_days") is not None else "Unknown",
                        f"≤{item.get('max_age_days', '')}d",
                    )
                    for item in attention[:8]
                ],
            )
        )
    return "\n".join(lines)


def _render_report_quality(bundle: Dict[str, Any]) -> str:
    quality = bundle.get("report_quality", {}) or {}
    fact_check = bundle.get("fact_check", {}) or {}
    provenance_audit = bundle.get("provenance_audit", {}) or {}
    if not quality:
        return "Report-quality diagnostics were not available."

    lines = [
        f"- **Quality score:** {quality.get('score', 'N/A')}/100 | **Grade:** {quality.get('grade', 'N/A')} | **Status:** {str(quality.get('status', 'N/A')).replace('_', ' ')}",
    ]

    runtime_summary = str(quality.get("runtime_summary", "") or "").strip()
    runtime_rows = quality.get("runtime_status", []) or []
    runtime_guidance = quality.get("runtime_guidance", []) or []
    runtime_guidance_summary = str(quality.get("runtime_guidance_summary", "") or "").strip()
    release_recommendation = quality.get("release_recommendation", {}) or {}
    if runtime_summary:
        lines.append(f"- **Run summary:** {runtime_summary}")
    if release_recommendation:
        lines.append(
            f"- **Release recommendation:** {release_recommendation.get('label', 'N/A')} | {release_recommendation.get('reason', '')}"
        )
    if runtime_rows:
        lines.append(_make_table(["Source", "Status", "Bucket"], [(item.get("name", ""), item.get("status", ""), item.get("bucket", "")) for item in runtime_rows]))
    if runtime_guidance:
        if runtime_guidance_summary:
            lines.append(f"\n**Desk-use guidance summary:** {runtime_guidance_summary}")
        lines.append("\n**Desk-use guidance**")
        lines.extend(
            f"- **{str(item.get('level', 'advisory')).capitalize()}:** {item.get('message', '')}"
            for item in runtime_guidance[:4]
        )

    components = quality.get("components", []) or []
    if components:
        rows = [
            (
                item.get("name", ""),
                item.get("score", ""),
                item.get("weight", ""),
                _condense_sentence(str(item.get("read", "")).replace("|", "/"), 120),
            )
            for item in components
        ]
        lines.append(_make_table(["Component", "Score", "Weight", "Read"], rows))

    warnings = quality.get("warnings", []) or []
    if warnings:
        lines.append("\n**Quality warnings**")
        lines.extend(f"- {item}" for item in warnings[:6])

    if fact_check:
        lines.append("\n**Narrative fact-check guardrail**")
        lines.append(f"- {fact_check.get('summary', 'Fact-check summary was not attached for this run.')}")
        mismatches = fact_check.get("numeric_mismatches", []) or []
        logic_warnings = fact_check.get("logic_warnings", []) or []
        if mismatches:
            rows = [
                (
                    item.get("severity", "critical"),
                    item.get("field", ""),
                    item.get("label", ""),
                    item.get("claim_type", ""),
                    item.get("claimed", ""),
                    item.get("expected", ""),
                    item.get("snippet", ""),
                )
                for item in mismatches[:6]
            ]
            lines.append(_make_table(["Severity", "Field", "Claim", "Type", "Claimed", "Expected", "Snippet"], rows))
        if logic_warnings:
            lines.extend(f"- Logic warning: {item.get('message', '')}" for item in logic_warnings[:6])
        source_warnings = fact_check.get("source_warnings", []) or []
        if source_warnings:
            lines.extend(f"- Source/text warning: {item.get('message', '')}" for item in source_warnings[:6])

    if provenance_audit:
        lines.append("\n**Source provenance validation**")
        lines.append(
            f"- Status: {provenance_audit.get('status', 'unknown')} | "
            f"records checked: {provenance_audit.get('checked_records', 0)} | "
            f"unavailable: {provenance_audit.get('unavailable_records', 0)}"
        )
        lines.extend(f"- Provenance error: {item}" for item in (provenance_audit.get("errors", []) or [])[:6])

    lines.append("\n**Source health and freshness**")
    lines.append(_render_source_health(bundle))

    adapter_rows = quality.get("adapter_status", []) or []
    if adapter_rows:
        lines.append("\n**Adapter status**")
        lines.append(_make_table(["Adapter", "Status"], [(item.get("name", ""), item.get("status", "")) for item in adapter_rows]))

    return "\n".join(lines)
