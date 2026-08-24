from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from market_diary.professional.analytics_market import _format_signed, _get_row, build_market_snapshot


def _theme_rotation_entry(report_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    weekday = datetime.strptime(report_date, "%Y-%m-%d").weekday()
    rotations = ((config.get("thinking", {}) or {}).get("rotation", []) or [])
    for entry in rotations:
        if int(entry.get("weekday", -1)) == weekday:
            return entry
    return rotations[0] if rotations else {
        "theme": "Hong Kong Market Structure and Flows",
        "angle": "Track whether style leadership and cross-border flows remain supportive.",
        "keywords": ["hong kong", "flow", "turnover"],
    }


def build_theme_deep_dive(
    report_date: str,
    config: Dict[str, Any],
    sector_digest: Dict[str, Any],
    watchlists: Dict[str, List[Dict[str, Any]]],
    high_frequency: List[Dict[str, Any]],
    catalysts: List[Dict[str, Any]],
) -> Dict[str, Any]:
    entry = _theme_rotation_entry(report_date, config)
    keywords = [str(keyword).lower() for keyword in entry.get("keywords", [])]

    matched_news: List[Dict[str, Any]] = []
    for item in (sector_digest or {}).get("graded_news", []) or []:
        text = " ".join(
            [
                str(item.get("sector", "")),
                str(item.get("title", "")),
                str(item.get("summary", "")),
                str(item.get("why", "")),
            ]
        ).lower()
        if any(keyword in text for keyword in keywords):
            matched_news.append(item)

    related_names: List[Dict[str, Any]] = []
    focus_buckets = {"Core coverage", "Priority follow-up"}
    for bucket, items in (watchlists or {}).items():
        if bucket not in focus_buckets:
            continue
        for item in items:
            text = " ".join(
                [
                    str(item.get("name", "")),
                    str(item.get("ticker", "")),
                    str(item.get("bucket", bucket)),
                    str(item.get("note", "")),
                    str(item.get("upcoming_catalyst", "")),
                    str(item.get("thesis", "")),
                ]
            ).lower()
            if any(keyword in text for keyword in keywords):
                related_names.append(item)

    if not related_names:
        for bucket, bucket_items in (watchlists or {}).items():
            if bucket not in focus_buckets:
                continue
            related_names.extend(bucket_items[:1])
            if len(related_names) >= 3:
                break

    matched_catalysts: List[Dict[str, Any]] = []
    for item in catalysts:
        text = " ".join([str(item.get("event", "")), str(item.get("impact", "")), str(item.get("category", ""))]).lower()
        if any(keyword in text for keyword in keywords):
            matched_catalysts.append(item)

    signal_lines: List[str] = []
    for news in matched_news[:2]:
        signal_lines.append(f"{news.get('title', '')}: {news.get('why', '')}")
    for tracker in high_frequency[:2]:
        signal_lines.append(
            f"{tracker.get('label', '')} {tracker.get('change_display') or _format_signed(tracker.get('change_pct'))}: {tracker.get('interpretation', '')}"
        )
    if not signal_lines:
        signal_lines.append("No clean thematic signal matched the current rotation, so use the section mainly as a checklist.")

    return {
        "theme": entry.get("theme", ""),
        "angle": entry.get("angle", ""),
        "signals": signal_lines[:4],
        "news": matched_news[:3],
        "related_names": related_names[:4],
        "upcoming": matched_catalysts[:4],
    }


def build_today_forward(
    report_date: str,
    macro_agenda: List[Dict[str, Any]],
    catalysts: List[Dict[str, Any]],
    day_mode: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    today = report_date
    today_macro = [item for item in macro_agenda if item.get("date", today) == today][:6]
    today_catalysts = [item for item in catalysts if item.get("date", today) == today][:8]
    next_catalysts = catalysts[:10]
    is_trading_day = bool((day_mode or {}).get("is_trading_day", True))

    focus_lines = []
    if not is_trading_day:
        focus_lines.append(
            "Non-trading review: use today's calendar to prepare the next Hong Kong open rather than treating the last cash tape as a fresh signal."
        )
    if today_macro:
        focus_lines.append(
            f"Macro: {today_macro[0].get('event', '')} is the first item to anchor the open and the rates/FX response."
        )
    if today_catalysts:
        focus_lines.append(
            f"Corporate / event: {today_catalysts[0].get('event', '')} is the cleanest same-day catalyst to prepare for."
        )
    if not focus_lines:
        # Do not infer a light calendar from an empty feed: no calendar source
        # has reported, so absence of events is absence of coverage.
        focus_lines.append(
            "No same-day macro or catalyst items were supplied for this run, so the session has no "
            "scheduled anchor in this report; trade the overnight tape and positioning, and check an "
            "external calendar before assuming the day is genuinely quiet."
        )

    return {
        "today_macro": today_macro,
        "today_catalysts": today_catalysts,
        "next_catalysts": next_catalysts,
        "focus_lines": focus_lines,
    }


def build_reflection_prompts(config: Dict[str, Any], overview: Dict[str, Any], hk_desk_view: Dict[str, Any]) -> List[str]:
    prompts = ((config.get("thinking", {}) or {}).get("reflection_prompts", []) or [])
    dynamic = [
        f"Does the overnight tape still read as `{overview.get('risk_regime', 'Neutral')}`, or do I expect a different Hong Kong cash-session outcome?",
        f"Is today's Hong Kong setup better described as `{hk_desk_view.get('leadership', 'broad leadership')}`, and does that match my current mental model?",
    ]
    return dynamic + [str(prompt) for prompt in prompts]


def build_non_trading_focus(
    day_mode: Dict[str, Any],
    date_semantics: Dict[str, Any],
    overview: Dict[str, Any],
    macro_agenda: List[Dict[str, Any]],
    sector_digest: Dict[str, Any],
    high_frequency: List[Dict[str, Any]],
    catalysts: List[Dict[str, Any]],
    risk_data: Dict[str, Any],
) -> Dict[str, Any]:
    if (
        bool((day_mode or {}).get("is_trading_day", True))
        or (day_mode or {}).get("mode") in {"weekly_review", "week_ahead"}
    ):
        return {}

    active_categories = {"FX", "Commodities", "Crypto", "Rates", "Vol"}
    still_moving = [
        item
        for item in high_frequency
        if item.get("category") in active_categories and item.get("price") is not None
    ][:6]

    action_items: List[Dict[str, Any]] = []
    for item in ((risk_data or {}).get("geopolitical_risks", []) or [])[:3]:
        action_items.append(
            {
                "bucket": "Geopolitics",
                "item": f"{item.get('region', '')}: {item.get('event', '')}",
                "read": item.get("impact", "Watch risk-premium transmission into oil, gold, FX, and China proxies."),
            }
        )
    for item in (sector_digest.get("graded_news", []) or [])[:3]:
        action_items.append(
            {
                "bucket": "Policy / company tape",
                "item": item.get("title", ""),
                "read": item.get("why", ""),
            }
        )
    for item in (macro_agenda or [])[:3]:
        action_items.append(
            {
                "bucket": item.get("status", "Macro"),
                "item": item.get("event", ""),
                "read": item.get("impact", ""),
            }
        )

    event_watch: List[Dict[str, Any]] = []
    for item in still_moving[:4]:
        event_watch.append(
            {
                "channel": item.get("category", ""),
                "signal": f"{item.get('label', '')} {item.get('change_display') or _format_signed(item.get('change_pct'))}",
                "why": item.get("interpretation", ""),
                "next_check": "Keep this as the bridge signal until the next Hong Kong cash close confirms or rejects it.",
            }
        )
    for item in (macro_agenda or [])[:3]:
        event_watch.append(
            {
                "channel": item.get("status", "Macro"),
                "signal": item.get("event", ""),
                "why": item.get("impact", ""),
                "next_check": "Check the rates, FX, and China-proxy reaction after release or official communication.",
            }
        )
    for item in ((risk_data or {}).get("geopolitical_risks", []) or [])[:2]:
        event_watch.append(
            {
                "channel": "Geopolitics",
                "signal": f"{item.get('region', '')}: {item.get('event', '')}",
                "why": item.get("impact", "Potential risk-premium transmission into oil, gold, FX, and China proxies."),
                "next_check": "Map any escalation first into oil, gold, USD, CNH, and HK growth-beta sensitivity.",
            }
        )
    for item in (catalysts or [])[:3]:
        event_watch.append(
            {
                "channel": item.get("category", "Catalyst"),
                "signal": item.get("event", ""),
                "why": item.get("impact", ""),
                "next_check": "Prepare the base-case and risk-case talking points before the next open.",
            }
        )

    next_open = [
        "Refresh HKEX turnover, Stock Connect, short-selling, and AH dispersion once the next HK cash session closes.",
        "Use USD/CNH, USD/HKD, DXY, oil, gold, and crypto as bridge signals before cash markets reopen.",
        "Prepare one base case and one risk case for the next Hong Kong open instead of over-reading stale cash-market moves.",
    ]
    if catalysts:
        next_open.insert(0, f"First dated catalyst to prepare: {catalysts[0].get('date', '')} | {catalysts[0].get('event', '')}.")

    return {
        "summary": (
            f"No fresh Hong Kong cash-market session is assumed for {date_semantics.get('review_date')}; "
            f"HK local tape is {date_semantics.get('hk_cash_role')} from {date_semantics.get('hk_data_date')}."
        ),
        "still_moving": still_moving,
        "event_watch": event_watch[:8],
        "action_items": action_items[:6],
        "next_open": next_open[:5],
        "market_regime": overview.get("theme", ""),
    }


def build_weekly_review(
    day_mode: Dict[str, Any],
    date_semantics: Dict[str, Any],
    overview: Dict[str, Any],
    summary: Dict[str, Any],
    hk_desk_view: Dict[str, Any],
    high_frequency: List[Dict[str, Any]],
    sector_digest: Dict[str, Any],
    macro_agenda: List[Dict[str, Any]],
    catalysts: List[Dict[str, Any]],
    flow_tracker: Dict[str, Any],
    attribution: Dict[str, Any],
) -> Dict[str, Any]:
    del high_frequency
    if (day_mode or {}).get("mode") != "weekly_review":
        return {}

    rows = build_market_snapshot(summary)
    selected_labels = ["S&P 500", "Nasdaq 100", "Hang Seng Index", "3033.HK ETF", "DXY", "US 10Y", "Gold", "VIX"]
    cross_assets = []
    for label in selected_labels:
        row = _get_row(rows, label)
        if row:
            cross_assets.append(
                {
                    "asset": row.get("label", label),
                    "latest_move": row.get("change_display") or _format_signed(row.get("change_pct")),
                    "read": row.get("question", ""),
                }
            )

    hk_rows = []
    for label in ["Hang Seng Index", "HSCEI", "3033.HK ETF", "China proxy (FXI)", "USD/CNH", "USD/HKD"]:
        row = _get_row(rows, label)
        if row:
            hk_rows.append(
                {
                    "signal": row.get("label", label),
                    "latest_move": row.get("change_display") or _format_signed(row.get("change_pct")),
                    "read": row.get("question", ""),
                }
            )

    developments = []
    for item in (sector_digest.get("graded_news", []) or [])[:4]:
        developments.append(
            {
                "bucket": item.get("grade", "News"),
                "item": item.get("title", ""),
                "read": item.get("why", ""),
            }
        )
    for item in (macro_agenda or [])[:3]:
        developments.append(
            {
                "bucket": item.get("status", "Macro"),
                "item": item.get("event", ""),
                "read": item.get("impact", ""),
            }
        )

    next_week = [
        {
            "date": item.get("date", ""),
            "event": item.get("event", ""),
            "read": item.get("impact", ""),
        }
        for item in (catalysts or [])[:8]
    ]

    flow_lines = []
    conclusion = (flow_tracker or {}).get("conclusion")
    if conclusion:
        flow_lines.append(conclusion)
    for item in ((attribution or {}).get("dominant_drivers", []) or [])[:3]:
        flow_lines.append(f"{item.get('driver', '')}: {item.get('interpretation', '')}")

    desk_questions = [
        "Did Southbound flow confirm the index move, or was the week mainly offshore beta without local money follow-through?",
        "Did HIBOR and Aggregate Balance point to benign funding, or should tighter HKD liquidity be part of next week's risk case?",
        "Was leadership broad enough beyond the 3033.HK ETF / platform beta proxy, or was the week concentrated in a narrow style pocket?",
        "Which dated macro, policy, earnings, or IPO catalyst can realistically change the next-week narrative?",
        "What single data point would invalidate the base-case market pulse by Monday or Tuesday morning?",
    ]

    return {
        "window": {
            "start": day_mode.get("period_start", ""),
            "end": day_mode.get("period_end", ""),
            "review_date": date_semantics.get("review_date", ""),
        },
        "summary": (
            f"Weekly review window: {day_mode.get('period_start', '')} to {day_mode.get('period_end', '')}. "
            f"Core regime: {overview.get('theme', '')}. Hong Kong leadership: {hk_desk_view.get('leadership', '')}."
        ),
        "cross_assets": cross_assets,
        "hk_tape": hk_rows,
        "developments": developments[:6],
        "next_week": next_week,
        "flow_lines": flow_lines[:4],
        "desk_questions": desk_questions,
        "method_note": "Weekly mode uses the last completed cash tape, available cross-asset snapshots, and Hong Kong Trend Pack evidence when generated.",
    }


def build_week_ahead(
    day_mode: Dict[str, Any],
    date_semantics: Dict[str, Any],
    overview: Dict[str, Any],
    summary: Dict[str, Any],
    hk_desk_view: Dict[str, Any],
    macro_agenda: List[Dict[str, Any]],
    sector_digest: Dict[str, Any],
    high_frequency: List[Dict[str, Any]],
    catalysts: List[Dict[str, Any]],
    risk_data: Dict[str, Any],
    flow_tracker: Dict[str, Any],
    attribution: Dict[str, Any],
) -> Dict[str, Any]:
    """Build the Monday 'week ahead' block: calendar, forecast, watch list.

    ``summary``/``hk_desk_view``/``macro_agenda``/``flow_tracker``/``attribution``
    are accepted for signature parity with the other mode builders even when not
    all of them are consumed here.
    """
    del summary, hk_desk_view, macro_agenda, flow_tracker, attribution
    if (day_mode or {}).get("mode") != "week_ahead":
        return {}

    week_start = str((day_mode or {}).get("week_start", "") or "")
    week_end = str((day_mode or {}).get("week_end", "") or "")
    last_close = str((day_mode or {}).get("last_hk_trading_day", "") or "")

    # Week calendar: Mon-Fri, each day's dated catalysts already in `catalysts`.
    by_date: Dict[str, List[Dict[str, Any]]] = {}
    for item in (catalysts or []):
        date_value = str(item.get("date", "") or "")
        if date_value and week_start <= date_value <= week_end:
            by_date.setdefault(date_value, []).append(item)

    week_calendar: List[Dict[str, Any]] = []
    if week_start and week_end:
        start = datetime.strptime(week_start, "%Y-%m-%d").date()
        end = datetime.strptime(week_end, "%Y-%m-%d").date()
        cursor = start
        while cursor <= end:
            iso = cursor.isoformat()
            events = sorted(by_date.get(iso, []), key=lambda it: -float(it.get("score", 0) or 0))
            week_calendar.append(
                {
                    "date": iso,
                    "day": cursor.strftime("%a"),
                    "items": [
                        {"event": it.get("event", ""), "impact": it.get("impact", ""), "time": it.get("time", "")}
                        for it in events[:4]
                    ],
                }
            )
            cursor += timedelta(days=1)

    regime = str((overview or {}).get("risk_regime", "Neutral") or "Neutral").lower()
    if "risk-on" in regime:
        base_case = "Risk-on continuation: leadership stays with growth / platform names and the 3033.HK ETF proxy, supported by flows."
        risk_case = "A hawkish macro surprise or a sharp USD/CNH leg higher unwinds the risk-on bias and rotates into defensives."
    elif "risk-off" in regime:
        base_case = "Cautious tape: defensives and yield proxies lead while growth re-rates; keep sizing small until breadth confirms."
        risk_case = "A dovish policy surprise or strong Southbound buying forces a squeeze higher against the defensive positioning."
    else:
        base_case = "Range-bound / mixed: index chops around Friday's close until a dated catalyst resolves the direction."
        risk_case = "A scheduled print (CPI / FOMC / earnings) resolving hard either way breaks the range and re-rates style."

    still_moving = [
        item
        for item in high_frequency
        if item.get("category") in {"FX", "Commodities", "Crypto", "Rates", "Vol"} and item.get("price") is not None
    ][:4]

    watch_items: List[str] = []
    if last_close:
        watch_items.append(f"Open versus Friday's close ({last_close}): whether the gap holds or fades is the first signal of the week.")
    watch_items.append("Southbound flow: confirm the index move with local money rather than offshore beta.")
    watch_items.append("HSI vs 3033.HK ETF leadership: growth-led or value-led decides the week's style.")
    watch_items.append("USD/CNH and USD/HKD: FX stability is the precondition for a clean risk-on extension.")
    watch_items.append("Turnover vs 20-day: thin participation makes any index move easier to fade.")
    first_catalyst = catalysts[0] if catalysts else None
    if first_catalyst:
        watch_items.append(
            f"First dated catalyst: {first_catalyst.get('date', '')} | {first_catalyst.get('event', '')} — the cleanest early test of the base case."
        )

    weekend_digest: List[Dict[str, str]] = []
    for item in ((sector_digest or {}).get("graded_news", []) or [])[:3]:
        weekend_digest.append({"channel": "Weekend news", "signal": item.get("title", ""), "why": item.get("why", "")})
    for item in ((risk_data or {}).get("geopolitical_risks", []) or [])[:2]:
        weekend_digest.append(
            {"channel": "Geopolitics", "signal": f"{item.get('region', '')}: {item.get('event', '')}", "why": item.get("impact", "Watch risk-premium transmission into oil, gold, FX, and China proxies.")}
        )
    for item in still_moving:
        weekend_digest.append(
            {"channel": item.get("category", "Still-moving"), "signal": f"{item.get('label', '')} {item.get('change_display') or _format_signed(item.get('change_pct'))}", "why": item.get("interpretation", "")}
        )

    return {
        "week_start": week_start,
        "week_end": week_end,
        "last_close": last_close,
        "summary": (
            f"Week ahead ({week_start} to {week_end}) opens against a `{overview.get('risk_regime', 'Neutral')}` backdrop, "
            f"using Friday's close ({last_close}) as the baseline."
        ),
        "week_calendar": week_calendar,
        "forecast": {
            "base_case": base_case,
            "risk_case": risk_case,
        },
        "watch_items": watch_items[:8],
        "weekend_digest": weekend_digest[:8],
        "desk_questions": [
            "Which single dated catalyst this week is most likely to change the base case?",
            "Does Southbound flow confirm or contradict the overnight cross-asset tone?",
            "Is the week likely to be beta-led or driven by a narrow style pocket?",
            "What data point would invalidate the base case by mid-week?",
        ],
    }
