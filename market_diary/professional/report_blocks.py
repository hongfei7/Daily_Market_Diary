from __future__ import annotations

import html
import re
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
from market_diary.professional.analytics_market import _fresh_change_pct, build_market_snapshot
from market_diary.professional.report_sections import (
    _pick_metrics_by_name,
    _resolved_hk_leadership,
    _resolved_hk_lens,
    _safe_sentence_clip,
)
from market_diary.professional.report_text import (
    _compact_bullets,
    _condense_sentence,
    _render_labeled_paragraphs,
    _render_labeled_points,
)


def _macro_source_configured(bundle: Dict[str, Any]) -> bool:
    """Whether a macro calendar source actually reported for this run."""
    status = ((bundle.get("source_health_inputs", {}) or {}).get("macro_calendar", {}) or {}).get("status", "")
    return str(status).lower() not in {"", "unavailable", "error", "disabled"}


def _render_macro_table(bundle: Dict[str, Any], limit: int | None = None) -> str:
    rows = (bundle.get("macro_agenda", []) or [])[: limit or _report_setting(bundle, "top_macro_events", 6)]
    if not rows:
        # An empty table means "no source" far more often than "quiet calendar".
        # Saying the calendar was light implies evidence of absence that the
        # pipeline does not have.
        if not _macro_source_configured(bundle):
            return (
                "No macro calendar source reported for this run, so this section is empty. "
                "That is an absence of data, not evidence that the calendar was quiet."
            )
        return "The macro calendar source returned no events for this run."
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


def _restates_evidence(line: str, evidence: str) -> bool:
    """Whether a cross-market line only repeats figures already in Evidence.

    The style call cites HSCEI and 3033.HK, then the first cross-market bullet
    listed the same two levels again. Repeating a number does not add a fact.
    """
    if not evidence:
        return False
    figures = set(re.findall(r"[-+]?\d+\.\d+%", line))
    if not figures:
        return False
    already = set(re.findall(r"[-+]?\d+\.\d+%", evidence))
    return bool(figures) and figures.issubset(already)


def _lens_confidence(hk_desk_view: Dict[str, Any]) -> str:
    """Grade the style call by the evidence actually behind it.

    A reader cannot otherwise tell a call backed by fresh prices and confirming
    flow from one the pipeline could not verify at all.
    """
    if hk_desk_view.get("stale_inputs") or hk_desk_view.get("style") == "unconfirmed":
        return "unconfirmed"
    confirmations = len(hk_desk_view.get("confirmation_flags", []) or [])
    participation = len(hk_desk_view.get("participation_flags", []) or [])
    if confirmations >= 2 and participation == 0:
        return "high confidence"
    if participation and not confirmations:
        return "low confidence"
    return "medium confidence"


def _summary_yesterday(bundle: Dict[str, Any]) -> str:
    card = bundle.get("call_scorecard", {}) or {}
    verdict = str(card.get("verdict", "") or "")
    if not verdict:
        return "No previously published call is on file yet."
    # Plain text, not bold. The HTML template styles every <strong> inside a
    # summary bullet as an uppercase block label, so a second bold here rendered
    # the verdict on its own shouting line and orphaned the dash after it.
    headline = _safe_sentence_clip(card.get("headline", ""), 220)
    return f"{verdict} — {headline}" if headline else verdict


def _summary_overnight(bundle: Dict[str, Any]) -> str:
    """Lead with the semis complex, which drives Hong Kong tech most directly."""
    chain = bundle.get("ai_tmt_chain", {}) or {}
    parts: List[str] = []

    legs = [item for item in (chain.get("overnight_leg", []) or []) if item.get("available")]
    if legs:
        moves = ", ".join(f"{item['label']} {item['display']}" for item in legs[:3])
        direction = "lower" if (chain.get("overnight_avg_pct") or 0) < 0 else "higher"
        parts.append(f"Semis led {direction}: {moves}.")

    rows = build_market_snapshot(bundle.get("market_summary", {}) or {})
    hsi, _ = _fresh_change_pct(rows, "Hang Seng Index")
    tech, _ = _fresh_change_pct(rows, "3033.HK ETF")
    if hsi is not None and tech is not None:
        joiner = "but" if (hsi >= 0) != (tech >= 0) else "and"
        parts.append(f"HSI {hsi:+.2f}% {joiner} 3033.HK {tech:+.2f}%.")

    return " ".join(parts) if parts else "Overnight coverage was insufficient to summarise the tape."


def _summary_ai_tmt(bundle: Dict[str, Any]) -> str:
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    chain = bundle.get("ai_tmt_chain", {}) or {}

    stale = hk_desk_view.get("stale_inputs", []) or []
    if stale:
        return f"Style call withheld: {'; '.join(stale)}. The stale input is excluded from the risk score."

    spread = hk_desk_view.get("style_spread_pp")
    headline = str(hk_desk_view.get("headline", "") or "")
    parts: List[str] = []
    if spread is not None and headline:
        # Do not lowercase the headline: it contains proper terms such as
        # "H-share" that must keep their casing.
        lead = "Style, not beta" if abs(spread) >= 0.5 else "No decisive style winner"
        parts.append(f"{lead} — {headline}, a {abs(spread):.2f}pp spread.")
    elif headline:
        parts.append(f"{headline}.")

    if chain.get("divergence_note"):
        parts.append(_safe_sentence_clip(chain["divergence_note"], 190))
    else:
        implication = _safe_sentence_clip(hk_desk_view.get("implication", ""), 170)
        if implication:
            parts.append(implication)

    return " ".join(parts) if parts else "The Hong Kong style read could not be formed from available data."


def _summary_watch(bundle: Dict[str, Any]) -> str:
    parts: List[str] = []

    # "What to watch today" must be a same-day event, not the nearest upcoming
    # one: on a Friday a Monday print is not what to watch *today*.
    briefing_date = str((bundle.get("meta", {}) or {}).get("briefing_date", "") or "")
    agenda = bundle.get("macro_agenda", []) or []
    today_items = [
        item for item in agenda
        if str(item.get("date", "") or "") == briefing_date and str(item.get("status", "")).lower() == "upcoming"
    ]
    if today_items:
        item = today_items[0]
        event = str(item.get("event", "") or "").strip()
        when = str(item.get("time") or "").strip()
        if event:
            parts.append(f"{event}{f' ({when})' if when else ''}.")
    else:
        upcoming = [item for item in agenda if str(item.get("status", "")).lower() == "upcoming"]
        if upcoming:
            item = upcoming[0]
            event = str(item.get("event", "") or "").strip()
            on_date = str(item.get("date", "") or "").strip()
            if event:
                parts.append(f"Next scheduled catalyst: {event}{f' on {on_date}' if on_date else ''}.")

    chain = bundle.get("ai_tmt_chain", {}) or {}
    test = _safe_sentence_clip(chain.get("test", ""), 200)
    if not test:
        test = _safe_sentence_clip((bundle.get("hk_desk_view", {}) or {}).get("confirmation", ""), 200)
    if test:
        parts.append(test)

    return " ".join(parts) if parts else "No scheduled catalyst; trade the overnight tape and local flow."


def _render_executive_summary(bundle: Dict[str, Any], pulse: str) -> str:
    """Answer the same four questions every day, one sentence each.

    The previous format put the style call, its evidence, a conviction caveat, a
    partial-support clause and the portfolio implication into a single 70-word
    bullet. Fixed questions in a fixed order can be scanned in seconds and
    compared across days, which a free-form paragraph cannot.
    """
    del pulse  # The overnight line is built from the AI/TMT chain instead.

    questions = [
        ("Did yesterday's call work?", _summary_yesterday(bundle)),
        ("What changed overnight?", _summary_overnight(bundle)),
        ("What it means for AI/TMT", _summary_ai_tmt(bundle)),
        ("What to watch today", _summary_watch(bundle)),
    ]
    return "\n".join(f"- **{label}** {answer}" for label, answer in questions if answer)


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
        sections.append(f"**{bucket}**")
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
        return "- Detailed local flow evidence is covered under Flow Tracker and Attribution; keep this section focused on options, hedging, and macro-positioning risk."
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

    lines: List[str] = ["**Flow Takeaways**"]
    summary = tracker.get("summary", "Flow evidence was mixed rather than decisive.")
    if summary:
        lines.append(f"- {summary}")

    flow_bullets = tracker.get("flow_bullets", []) or []
    if flow_bullets:
        lines.extend(f"- {item}" for item in flow_bullets[:3])
    lines.append("")

    stock_connect = (tracker.get("stock_connect", {}) or {}).get("data", {}) or {}
    southbound_active = ((stock_connect.get("southbound", {}) or {}).get("top_active", []) or [])[:5]
    lines.append("**Stock Connect Southbound Active Names**")
    if southbound_active:
        # The table is ordered by net buy, so a HKEX turnover rank in the first
        # column read as a broken sequence (7, 2, 3, 9, 10). Number the rows by
        # the order actually shown and keep the exchange rank in its own column.
        rows = [
            (
                position,
                item.get("ticker", ""),
                item.get("name", ""),
                _fmt_millions(item.get("buy_turnover"), "HK$"),
                _fmt_millions(item.get("sell_turnover"), "HK$"),
                _fmt_millions(item.get("net_buy"), "HK$") if item.get("net_buy") is not None else "N/A",
                item.get("rank", "") or "N/A",
            )
            for position, item in enumerate(southbound_active, start=1)
        ]
        lines.append(
            _make_table(
                ["#", "Ticker", "Name", "Buy", "Sell", "Net buy", "HKEX turnover rank"],
                rows,
            )
        )
        lines.append("")
    else:
        status = (tracker.get("stock_connect", {}) or {}).get("status", "unavailable")
        lines.append(f"- Southbound active-name detail was not disclosed in the current public feed. Status: `{status}`.")
        lines.append("")

    ah_premium = (tracker.get("ah_premium", {}) or {}).get("data", {}) or {}
    ah_rows = ah_premium.get("top_premium", []) or []
    lines.append("**AH Premium Dispersion**")
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
        lines.append("**HKEX Short-Selling Watch**")
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

    if not _has_official_stock_connect_flow(bundle):
        proxy_table = _render_hk_etf_proxy_table(bundle)
        if "No Hong Kong or offshore-China ETF proxy data" not in proxy_table:
            lines.append("**ETF Proxy Read**")
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
    lines.append("**Market Setup**")
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
        lines.append("**Key Drivers**")
        lines.extend(f"- {item}" for item in drivers)

    # This section covers the overnight overseas tape. The Hong Kong style call
    # belongs to Section 2.2 and is deliberately not repeated here: rendering the
    # same hk_desk_view fields in both places produced word-for-word duplicate
    # paragraphs plus three identical "Cross-market read" bullets every day.
    hk_implication = str(llm_sections.get("overnight_hk_implication", "") or "").strip()
    lines.append("")
    lines.append("**Hong Kong Read-Through**")
    if hk_implication:
        lines.append(f"**Opening implication.** {_condense_sentence(hk_implication, 260)}")
    else:
        lines.append(
            "**Opening implication.** Carry the overnight tape into the Hong Kong open using the style and "
            "flow evidence in the Hong Kong review rather than the headline index alone."
        )
    lines.append(
        "_Hong Kong style leadership and flow confirmation are set out in the Hong Kong review below._"
    )

    chart_read = (overview.get("chart_read", {}) or {})
    watch_points = _compact_bullets((chart_read.get("fx", []) or []) + (chart_read.get("assets", []) or []), limit=4, width=150)
    if watch_points:
        lines.append("")
        lines.append("**Watch Points**")
        lines.extend(f"- {item}" for item in watch_points)

    return "\n".join(lines)


def _render_hk_review_block(bundle: Dict[str, Any]) -> str:
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}

    lines: List[str] = []
    setup = str(llm_sections.get("hk_review_setup", "") or "").strip()
    if setup:
        lines.append("**Local Tape Setup**")
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
    headline = str(hk_desk_view.get("headline", "") or "").strip()
    evidence = str(hk_desk_view.get("evidence", "") or "").strip()
    implication = str(hk_desk_view.get("implication", "") or "").strip()
    lines.append("**Style and Local Leadership**")
    if headline:
        lines.append(f"**Style call.** {headline}.")
        if evidence:
            lines.append(f"- **Evidence:** {_condense_sentence(evidence, 220)}")
        if implication:
            lines.append(f"- **Portfolio meaning:** {_condense_sentence(implication, 240)}")
    elif leadership:
        lines.append(f"**Style leadership.** {leadership}.")
    else:
        lines.append("**Style leadership.** Check HSI, HSCEI, and the 3033.HK ETF proxy first to separate broad-beta, old-economy, and growth leadership.")

    # One label, one line. Three consecutive bullets all called "Cross-market
    # read" repeated the label without distinguishing the facts, and the first of
    # them restated the index levels already given in Evidence directly above.
    hk_lines, suppressed_hk_lines = _compact_hk_read_lines(hk_desk_view.get("lines", []) or [], limit=3)
    hk_lines = [line for line in hk_lines if not _restates_evidence(line, evidence)]
    if hk_lines:
        lines.append(f"- **Cross-market read:** {' '.join(hk_lines)}")
    if suppressed_hk_lines:
        lines.append(
            "- **Coverage note:** Low-signal index / FX lines were suppressed because quote coverage was incomplete; flow confirmation below carries more weight for this run."
        )

    lines.append("")
    lines.append("**Flow Confirmation**")
    if _has_official_stock_connect_flow(bundle):
        lines.append("Official Stock Connect evidence is available; the Flow Tracker confirms whether local money supports the price action.")
    else:
        lines.append("Official Stock Connect confirmation is incomplete; use ETF proxies and price leadership only as secondary evidence.")

    follow_through = str(llm_sections.get("hk_follow_through", "") or "").strip()
    deterministic_confirmation = str(hk_desk_view.get("confirmation", "") or "").strip()
    deterministic_invalidation = str(hk_desk_view.get("invalidation", "") or "").strip()
    if not follow_through:
        follow_through = deterministic_confirmation or "Confirm the opening read through Southbound active names, short-selling concentration, USD/CNH, and USD/HKD funding pressure."
    lines.append("")
    lines.append("**Follow-Through Checklist**")
    lines.append(f"**Follow-through check.** {_condense_sentence(follow_through, 260)}")
    if deterministic_invalidation:
        lines.append(f"**Failure condition.** {_condense_sentence(deterministic_invalidation, 260)}")

    # The ETF proxy fallback lives in the Flow Tracker (2.4); rendering it here
    # too produced the same table twice on every low-flow day.

    return "\n".join(lines)


def _event_card_html(item: Dict[str, Any]) -> str:
    priority = str(item.get("priority", "Monitor") or "Monitor")
    priority_slug = priority.lower().replace(" ", "-")
    company = str(item.get("company", "") or "").strip()
    ticker = str(item.get("ticker", "") or "").strip()
    identity = " · ".join(part for part in (company, ticker) if part) or "Market event"
    event_type = str(item.get("event_type", "Company event") or "Company event")
    release_time = str(item.get("release_time", item.get("time", "")) or "Timing not supplied")
    fact = str(item.get("filing_extract") or item.get("what_changed") or item.get("title") or "Primary detail pending review.")
    investor_read = str(item.get("investor_read") or "Assess whether the event changes estimates, valuation or thesis confidence.")
    next_check = str(item.get("next_check") or "Open the primary source and identify the next dated confirmation point.")
    drivers = str(item.get("filing_drivers", "") or "").strip()
    source = str(item.get("source", "") or "Primary source")
    url = str(item.get("source_url", item.get("url", "")) or "").strip()
    source_link = (
        f'<a class="event-source" href="{html.escape(url, quote=True)}">{html.escape(source)} filing ↗</a>'
        if url.startswith(("http://", "https://"))
        else f'<span class="event-source event-source-muted">{html.escape(source)}</span>'
    )
    drivers_html = (
        f'<p class="event-drivers"><span>Drivers</span>{html.escape(_truncate(drivers, 260, suffix=""))}</p>'
        if drivers
        else ""
    )
    return f"""<article class="event-card priority-{priority_slug}">
<div class="event-card-meta"><span class="event-priority">{html.escape(priority)}</span><span>{html.escape(event_type)}</span><time>{html.escape(release_time)}</time></div>
<h5>{html.escape(identity)}</h5>
<p class="event-fact">{html.escape(_truncate(fact, 330, suffix=""))}</p>
{drivers_html}<div class="event-read-grid">
<div><span>Investor read</span><p>{html.escape(_truncate(investor_read, 230, suffix=""))}</p></div>
<div><span>Next check</span><p>{html.escape(_truncate(next_check, 230, suffix=""))}</p></div>
</div>
{source_link}
</article>"""


def _coverage_boundary(company_events: Dict[str, Any]) -> str:
    missing: List[str] = []
    if not (company_events.get("earnings", []) or []) and company_events.get("earnings_status") != "ok":
        missing.append("Earnings calendar")
    if not (company_events.get("ratings", []) or []) and company_events.get("ratings_status") != "ok":
        missing.append("sell-side rating changes")
    if company_events.get("ipo_status") == "not_covered":
        missing.append("IPO / grey-market monitor")
    if not missing:
        return ""
    return (
        '<p class="event-coverage-note"><strong>Coverage boundary.</strong> '
        + html.escape(", ".join(missing))
        + " were not decision-grade in this run; their absence is not treated as confirmation that no event exists.</p>"
    )


def _render_company_events(bundle: Dict[str, Any]) -> str:
    company_events = bundle.get("company_events", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    sections: List[str] = []

    if llm_sections.get("company_takeaway"):
        sections.append("**Company Event Takeaway**")
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
        sections.append("**LLM Quick Takes**")
        for item in company_notes[:6]:
            sections.append(f"- **{item.get('ticker', '')}** | {_truncate(item.get('commentary', ''), 170, suffix='')}")
        sections.append("")

    summary = company_events.get("event_summary", {}) or {}
    filings = int(summary.get("official_filings", 0) or 0)
    watchlist_hits = int(summary.get("watchlist_hits", 0) or 0)
    announced_actionable = int(summary.get("actionable_events", 0) or 0)
    type_counts = summary.get("type_counts", {}) or {}
    hkex_status = str((company_events.get("hkex_meta", {}) or {}).get("status", "unavailable") or "unavailable")

    cards: List[Dict[str, Any]] = []
    for item in company_events.get("announcements", []) or []:
        if item.get("priority") in {"Portfolio", "High", "Review"}:
            cards.append(dict(item))

    for item in company_events.get("earnings", []) or []:
        cards.append(
            {
                **item,
                "priority": "High",
                "event_type": "Earnings",
                "what_changed": item.get("comparison", "Expectation frame pending"),
                "investor_read": "Potential estimate reset: compare actual KPIs and guidance with the stated expectation bar.",
                "next_check": "Prepare the KPI and valuation bridge before the release window.",
                "release_time": item.get("time", ""),
                "url": item.get("source_url", ""),
            }
        )

    for item in company_events.get("ratings", []) or []:
        cards.append(
            {
                **item,
                "priority": "Review",
                "company": item.get("firm", ""),
                "event_type": "Rating change",
                "what_changed": " | ".join(
                    part for part in (str(item.get("action", "")), str(item.get("summary", "")), f"PT {item.get('target_change')}" if item.get("target_change") else "") if part
                ),
                "investor_read": "Treat as a sentiment and estimate-change signal, not as primary company evidence.",
                "next_check": "Check the estimate revisions and thesis logic behind the rating action.",
                "release_time": item.get("as_of", ""),
                "url": item.get("source_url", ""),
            }
        )

    priority_rank = {"Portfolio": 0, "High": 1, "Review": 2, "Monitor": 3}
    cards.sort(key=lambda item: (priority_rank.get(str(item.get("priority", "Monitor")), 4), str(item.get("release_time", ""))), reverse=False)
    cards = cards[:4]

    if watchlist_hits:
        verdict = f"Portfolio attention required: {watchlist_hits} official filing{'s' if watchlist_hits != 1 else ''} matched the active coverage list."
    elif announced_actionable:
        verdict = f"No watchlist filing hit; {announced_actionable} market event{'s' if announced_actionable != 1 else ''} cleared the decision filter."
    elif filings:
        verdict = f"No immediate portfolio catalyst: {filings} official filings were screened and the low-signal market set stays aggregated."
    elif hkex_status in {"ok", "partial"}:
        verdict = "No portfolio-relevant HKEX filing was identified in the screened window."
    else:
        verdict = "Official HKEX filing coverage was unavailable; do not interpret the empty event set as a clean calendar."

    hygiene_parts = []
    for label, key in (("profit warnings", "profit_warnings"), ("results", "results"), ("trading-status notices", "trading_status")):
        count = int(type_counts.get(key, 0) or 0)
        if count:
            hygiene_parts.append(f"{count} {label}")
    hygiene = " · ".join(hygiene_parts) or "No categorized official filing count was available."

    card_html = "\n".join(_event_card_html(item) for item in cards)
    if not card_html:
        card_html = """<div class="event-monitor-empty">
<strong>No event cleared the portfolio decision filter.</strong>
<p>Broad-market filings remain traceable in the source appendix; they are not expanded here without a watchlist, estimate or liquidity read-through.</p>
</div>"""

    sections.append(
        f"""<div class="company-event-monitor">
<div class="event-monitor-summary">
<div class="event-summary-copy"><span class="event-kicker">Decision filter</span><h4>{html.escape(verdict)}</h4><p>{html.escape(hygiene)}</p></div>
<div class="event-stats" aria-label="Company event summary">
<div><strong>{filings}</strong><span>Official filings</span></div>
<div><strong>{watchlist_hits}</strong><span>Watchlist hits</span></div>
<div><strong>{announced_actionable}</strong><span>Actionable events</span></div>
</div>
</div>
<div class="event-card-list">{card_html}</div>
{_coverage_boundary(company_events)}
</div>"""
    )
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

    output = ["\n".join(lines)]

    # Rendering three empty sub-blocks every day made an unimplemented source
    # look like a genuinely quiet calendar. Collapse to one honest statement
    # when nothing at all is populated.
    if not macro_rows and not catalyst_rows and not next_rows:
        if not _macro_source_configured(bundle):
            output.append(
                "\n_No macro calendar or catalyst source reported for this run, so no same-day or forward "
                "events can be listed. This is missing coverage, not confirmation that the calendar is empty._"
            )
        else:
            output.append(
                "\n_The configured calendar and catalyst sources returned no same-day or forward events._"
            )
        return "\n".join(output)

    output.append("\n**Same-day macro docket**")
    output.append(
        _make_table(["Time", "Country", "Event", "Status", "Detail"], macro_rows)
        if macro_rows
        else "No same-day macro items were scheduled."
    )
    if catalyst_rows:
        output.append("\n**Same-day catalyst list**")
        output.append(_make_table(["Date", "Time", "Category", "Event", "Importance"], catalyst_rows))
    if next_rows:
        output.append("\n**Next few sessions**")
        output.append(_make_table(["Date", "Category", "Event", "Impact"], next_rows))
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


def _render_catalyst_radar(bundle: Dict[str, Any], catalyst_radar_rel_path: str) -> str:
    chart_meta = bundle.get("catalyst_radar", {}) or {}
    chart_path = catalyst_radar_rel_path or chart_meta.get("rel_path", "")
    if not chart_path:
        return ""

    title = chart_meta.get("title", "Catalyst & Event Radar")
    caption = chart_meta.get("caption", "")
    source = chart_meta.get("source", "")
    source_line = f"\n\n_Source: {source}_" if source else ""
    chart_read = f"\n\n**Chart read:** {caption}" if caption else ""
    return f"**{title}**\n\n![Catalyst & Event Radar]({chart_path}){chart_read}{source_line}"


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

    # Per-day scoring now lives in Section 1.1, so this stays a compact
    # aggregate: readiness, the headline table, and the boundary on how far the
    # result can be pushed.
    data_quality = performance.get("data_quality", {}) or {}
    lines = [
        f"- **Readiness:** {str(performance.get('status', 'unknown')).replace('_', ' ')} | "
        f"{data_quality.get('observations', 0)} observations | "
        f"{data_quality.get('active_signal_dates', 0)} active signal dates | "
        "next-close entry, 10bps cost, no same-day returns.",
        "- **Interpretation boundary:** exploratory until a benchmark reaches 252 sessions and 100 active-signal sessions.",
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

    # Event-horizon diagnostics are research depth rather than commute reading;
    # the full breakdown is archived in audit/performance_summary.json.
    hsi_events = ((performance.get("benchmarks", {}) or {}).get("Hang Seng Index", {}) or {}).get("event_horizons", {}) or {}
    one_session = hsi_events.get("1") or hsi_events.get(1) or {}
    if one_session:
        lines.append(
            f"- **Next-session diagnostic:** {_percent(one_session.get('hit_rate'))} hit rate over "
            f"{one_session.get('resolved_signals', 0)} resolved signals; 5- and 20-session horizons are in "
            "`audit/performance_summary.json`."
        )

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
                ["Source", "Critical", "Status", "Score", "Fresh coverage", "Age range", "Policy"],
                [
                    (
                        item.get("source", ""),
                        "Yes" if item.get("critical") else "No",
                        item.get("status", ""),
                        item.get("score", ""),
                        f"{item.get('fresh_records', 0)}/{item.get('active_records', 0)}",
                        (
                            f"{item.get('freshest_age_days')}–{item.get('oldest_age_days')}d"
                            if item.get("freshest_age_days") is not None
                            else "Unknown"
                        ),
                        f"≤{item.get('max_age_days', '')}d / ≥{float(item.get('min_fresh_ratio', 0) or 0):.0%}",
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
    # Only unhealthy sources are listed. A full roster of every source and its
    # bucket ran to ~10 rows of process detail in a report read on a commute;
    # the complete table is archived in audit/source_health.json.
    degraded_rows = [item for item in runtime_rows if str(item.get("bucket", "")).lower() not in {"healthy", "ok", ""}]
    if degraded_rows:
        lines.append(
            _make_table(
                ["Source", "Status", "Bucket"],
                [(item.get("name", ""), item.get("status", ""), item.get("bucket", "")) for item in degraded_rows],
            )
        )
    elif runtime_rows:
        lines.append(f"- All {len(runtime_rows)} sources were healthy on this run.")
    if runtime_guidance:
        if runtime_guidance_summary:
            lines.append(f"\n**Desk-use guidance summary:** {runtime_guidance_summary}")
        lines.append("\n**Desk-use guidance**")
        lines.extend(
            f"- **{str(item.get('level', 'advisory')).capitalize()}:** {item.get('message', '')}"
            for item in runtime_guidance[:4]
        )

    # Grade ceilings change how much weight to put on the report, so they stay.
    for cap in (quality.get("grade_caps", []) or [])[:3]:
        lines.append(f"- **Grade capped:** {cap}")

    # Only components that actually dragged the score are worth a commute
    # reader's attention; the full weighted breakdown is archived alongside the
    # report rather than printed in it.
    components = quality.get("components", []) or []
    weak = [item for item in components if float(item.get("score", 100) or 100) < 70]
    if weak:
        rows = [
            (
                item.get("name", ""),
                item.get("score", ""),
                _condense_sentence(str(item.get("read", "")).replace("|", "/"), 120),
            )
            for item in weak
        ]
        lines.append(_make_table(["Weak component", "Score", "Read"], rows))

    warnings = quality.get("warnings", []) or []
    if warnings:
        lines.append("\n**Quality warnings**")
        lines.extend(f"- {item}" for item in warnings[:6])

    if fact_check:
        lines.append("\n**Narrative fact-check guardrail**")
        lines.append(f"- {fact_check.get('summary', 'Fact-check summary was not attached for this run.')}")
        degraded_fields = fact_check.get("degraded_fields", []) or []
        if degraded_fields:
            lines.append(f"- **Deterministic fallback fields:** {', '.join(str(item) for item in degraded_fields[:8])}")
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

    # Provenance, per-source freshness and adapter status are pipeline
    # diagnostics rather than market content. They are written to audit/*.json
    # on every run, so the report carries only the exceptions.
    if provenance_audit and str(provenance_audit.get("status", "")).lower() != "ok":
        lines.append(
            f"\n- **Source provenance:** {provenance_audit.get('status', 'unknown')} | "
            f"{provenance_audit.get('unavailable_records', 0)} unavailable of "
            f"{provenance_audit.get('checked_records', 0)} records checked."
        )
        lines.extend(f"- Provenance error: {item}" for item in (provenance_audit.get("errors", []) or [])[:3])

    health = bundle.get("source_health", {}) or {}
    if str(health.get("status", "")).lower() not in {"", "healthy", "ok"}:
        coverage = health.get("coverage", {}) or {}
        lines.append(
            f"- **Source health:** {health.get('status')} | {coverage.get('healthy', 0)} healthy, "
            f"{coverage.get('degraded', 0)} degraded, {coverage.get('unavailable', 0)} unavailable."
        )

    failed_adapters = [
        item for item in (quality.get("adapter_status", []) or [])
        if str(item.get("status", "")).lower() not in {"ok", "healthy"}
    ]
    if failed_adapters:
        lines.append(
            "- **Adapters not OK:** "
            + ", ".join(f"{item.get('name', '')} ({item.get('status', '')})" for item in failed_adapters)
        )

    lines.append(
        "\n_Full component weights, per-source freshness, adapter status and provenance records are archived "
        "with this report under `audit/` rather than printed here._"
    )

    return "\n".join(lines)


def _render_ai_tmt_chain(bundle: Dict[str, Any]) -> str:
    """Render the overnight-semis to Hong Kong-tech read-through.

    Stated as an explicit chain rather than a score table so the reasoning can
    be repeated in a morning meeting, not just the numbers.
    """
    chain = bundle.get("ai_tmt_chain", {}) or {}
    if not chain or chain.get("status") == "unavailable":
        return (
            "Semiconductor coverage was unavailable for this run, so no AI/TMT read-through is offered. "
            "This is missing coverage, not evidence that the complex was quiet."
        )

    lines: List[str] = [
        f"**Overnight leg.** {chain.get('headline', '')}",
        f"**Hong Kong expression.** {chain.get('expression', '')}",
        f"**Observable test.** {chain.get('test', '')}",
        "",
    ]

    rows = [
        (item["label"], item["role"], item["display"], "overnight")
        for item in (chain.get("overnight_leg", []) or [])
    ] + [
        (item["label"], item["role"], item["display"], "Hong Kong")
        for item in (chain.get("hk_leg", []) or [])
    ]
    if rows:
        lines.append(_make_table(["Name", "Role in the chain", "1D", "Leg"], rows))
        lines.append("")

    if chain.get("divergence_note"):
        lines.append(f"**Read with care.** {chain['divergence_note']}")
    elif chain.get("hk_followed_overnight") is True:
        lines.append(
            f"**Coherence.** Hong Kong tech moved with the overnight leg "
            f"(semis {chain.get('overnight_avg_pct'):+.2f}% versus HK tech {chain.get('hk_avg_pct'):+.2f}%), "
            "so the global cycle is a sufficient explanation without invoking local flow."
        )

    if chain.get("single_name_outliers"):
        lines.append(
            f"**Single-name outlier.** {', '.join(chain['single_name_outliers'])} moved far outside the rest "
            "of the Hong Kong leg. Treat as company-specific until checked; do not read it as a cycle signal."
        )

    if chain.get("stale_inputs"):
        lines.append(
            f"**Coverage caveat.** {'; '.join(chain['stale_inputs'])}. "
            "Those names are excluded from the averages above."
        )

    return "\n".join(lines)


def _render_call_scorecard(bundle: Dict[str, Any]) -> str:
    """Score the previously published call before presenting a new one.

    A desk's first question in the morning is whether yesterday's read worked.
    The ledger has always held the answer; it was only ever surfaced as an
    aggregate hit rate in the appendix.
    """
    card = bundle.get("call_scorecard", {}) or {}
    record = bundle.get("call_record", {}) or {}
    if not card or card.get("status") == "error":
        return "The previous call could not be scored for this run."

    verdict = card.get("verdict", "UNRESOLVED")
    lines: List[str] = [f"**Verdict: {verdict}.** {card.get('headline', '')}"]

    moves = [item for item in (card.get("moves", []) or []) if item.get("move_pct") is not None]
    if moves:
        lines.append("")
        lines.append(
            _make_table(
                ["Benchmark", "From", "To", "Realised move"],
                [
                    (
                        item["label"],
                        item.get("from_date", ""),
                        item.get("to_date", ""),
                        f"{item['move_pct']:+.2f}%",
                    )
                    for item in moves
                ],
            )
        )

    if record.get("scored"):
        lines.append("")
        lines.append(
            f"**Recent record.** {record['confirmed']} confirmed / {record['broken']} broken over the last "
            f"{record['scored']} scoreable calls ({record.get('hit_rate_pct')}% hit rate). "
            "This is a directional close-to-close diagnostic, not a track record."
        )

    if verdict == "BROKEN":
        lines.append("")
        lines.append(
            "**Carry-forward.** State the miss before restating today's view; a thesis that just failed "
            "needs new evidence, not a repeat."
        )

    return "\n".join(lines)


def _render_md_questions(bundle: Dict[str, Any]) -> str:
    """Render the questions a senior is most likely to ask, with answers.

    Deterministic by design: the narrative overlay is too unreliable to carry
    morning-meeting preparation.
    """
    questions = bundle.get("md_questions", []) or []
    if not questions:
        return (
            "No divergence, tail reading or coverage gap stood out today, so there is no obvious "
            "follow-up question beyond the base case above."
        )

    lines: List[str] = []
    for idx, item in enumerate(questions, start=1):
        lines.append(f"**Q{idx}. {item.get('question', '')}**")
        lines.append(f"- **Evidence:** {item.get('evidence', '')}")
        lines.append(f"- **How to answer:** {item.get('answer', '')}")
        lines.append("")
    return "\n".join(lines).rstrip()
