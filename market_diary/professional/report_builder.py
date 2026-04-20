from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence


def _report_setting(bundle: Dict[str, Any], key: str, default: int) -> int:
    report_config = (bundle.get("report_config", {}) or {})
    value = report_config.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _render_market_data_quality(meta: Dict[str, Any]) -> str:
    quality = meta.get("market_quality", {}) or {}
    if not quality:
        return ""

    available = quality.get("available")
    total = quality.get("total")
    fallback_count = len(quality.get("fallback", []) or [])
    stale_count = len(quality.get("stale", []) or [])
    missing_count = len(quality.get("missing", []) or [])

    parts: List[str] = []
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


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:+.2f}%"


def _fmt_price(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) < 10:
        digits = max(digits, 4)
    return f"{number:,.{digits}f}"


def _fmt_hkd_bn(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"HK${number / 1_000_000_000:.1f}bn"


def _fmt_millions(value: Any, currency: str = "HK$") -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{currency}{number:,.1f}mn"


def _status_label(status: str) -> str:
    mapping = {
        "live_local": "Live local",
        "stale_local": "Stale local",
        "live_public": "Live public",
        "stale_public": "Stale public",
        "live_quote": "Live quote",
        "live_hybrid": "Live quote + local",
        "proxy": "Proxy fallback",
        "unavailable": "Unavailable",
    }
    return mapping.get(str(status or ""), str(status or "Unavailable").replace("_", " ").title())


def _source_as_of(item: Dict[str, Any]) -> str:
    source = str(item.get("source", "") or "").strip()
    as_of = str(item.get("as_of", "") or "").strip()
    if source and as_of:
        return f"{source} | {as_of}"
    if source:
        return source
    if as_of:
        return as_of
    return "N/A"


def _bundle_metric(bundle: Dict[str, Any], section: str, key: str) -> Dict[str, Any]:
    section_data = bundle.get(section, {}) or {}
    item = section_data.get(key, {}) if isinstance(section_data, dict) else {}
    return item if isinstance(item, dict) else {}


def _truncate(text: str, limit: int = 110) -> str:
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _make_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def _cell(value: Any) -> str:
        text = str(value)
        text = text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")
        return text

    lines = ["| " + " | ".join(_cell(header) for header in headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _summary_item(bundle: Dict[str, Any], category: str, name: str) -> Dict[str, Any]:
    summary = (bundle.get("market_summary", {}) or {})
    item = (summary.get(category, {}) or {}).get(name, {})
    return item if isinstance(item, dict) else {}


def _summary_price(bundle: Dict[str, Any], category: str, name: str) -> Any:
    return _summary_item(bundle, category, name).get("Price")


def _summary_pct(bundle: Dict[str, Any], category: str, name: str) -> Any:
    item = _summary_item(bundle, category, name)
    value = item.get("Pct Change")
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_alert_pct(value: Any, threshold: float = 1.5) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:+.2f}%"
    return f"**{text}**" if abs(number) >= threshold else text


def _render_top_items(items: List[Dict[str, Any]], limit: int = 6) -> str:
    if not items:
        return "1. No priority items were available."
    lines = []
    for idx, item in enumerate(items[:limit], 1):
        lines.append(f"{idx}. **{item.get('title', '')}**")
        lines.append(f"   {item.get('bucket', '')} | {_truncate(item.get('summary', ''), 110)}")
    return "\n".join(lines)


def _render_selected_news(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    items = llm_sections.get("selected_news", []) or []
    if not items:
        return ""
    lines = ["**Curated overnight stories:**"]
    for item in items[:5]:
        lines.append(
            f"- **{item.get('headline', '')}** | {item.get('why_it_matters', '')} | HK read-through: {item.get('hk_market_impact', '')}"
        )
    return "\n".join(lines)


def _render_global_asset_dashboard(bundle: Dict[str, Any]) -> str:
    rows = [
        ("Equities", "S&P 500", "S&P 500", "US large-cap risk appetite"),
        ("Equities", "Nasdaq 100", "Nasdaq 100", "Growth and duration leadership"),
        ("Equities", "Dow Jones", "Dow Jones", "Old-economy and cyclicals"),
        ("Equities", "Hang Seng Index", "Hang Seng Index", "Hong Kong broad beta"),
        ("Equities", "Hang Seng TECH ETF", "Hang Seng TECH", "Hong Kong growth / platform read-through"),
        ("Equities", "CSI 300", "CSI 300", "Mainland large-cap tone"),
        ("Equities", "ChiNext Index", "ChiNext", "Mainland growth risk appetite"),
        ("Equities", "Nikkei 225", "Nikkei 225", "Asia developed-market leadership"),
        ("Equities", "Euro Stoxx 50", "Euro Stoxx 50", "Europe macro risk"),
        ("Rates", "10Y Treasury", "US 10Y", "Global discount-rate anchor"),
        ("Rates", "China 10Y", "China 10Y", "China local rates anchor"),
        ("Rates", "CN-US 10Y spread", "CN-US 10Y spread", "Relative carry and macro-pressure gauge"),
        ("FX", "DXY", "DXY", "Dollar liquidity impulse"),
        ("FX", "USD/CNH", "USD/CNH", "Offshore China risk appetite"),
        ("FX", "USD/HKD", "USD/HKD", "Linked-exchange regime stress check"),
        ("FX", "USD/JPY", "USD/JPY", "Asia funding and carry read"),
        ("Commodities", "Brent Crude", "Brent crude", "Energy / geopolitics"),
        ("Commodities", "Gold", "COMEX gold", "Hedge demand / real yields"),
        ("Commodities", "Copper", "Copper", "Global growth proxy"),
        ("Vol", "VIX", "VIX", "US stress barometer"),
        ("Vol", "HSI Volatility", "HSI volatility", "No stable public HSI volatility feed is configured"),
    ]

    table_rows = []
    china_10y_metric = _bundle_metric(bundle, "china_rates", "china_10y")
    spread_metric = _bundle_metric(bundle, "china_rates", "cn_us_10y_spread")

    for category, name, label, note in rows:
        if name == "China 10Y":
            price = china_10y_metric.get("display_value", "N/A")
            pct = china_10y_metric.get("change_display", "")
            read = f"{note} | {_status_label(china_10y_metric.get('status', 'unavailable'))} | {_source_as_of(china_10y_metric)}"
        elif name == "CN-US 10Y spread":
            price = spread_metric.get("display_value", "N/A")
            pct = spread_metric.get("change_display", "")
            read = f"{note} | {_status_label(spread_metric.get('status', 'unavailable'))} | {_source_as_of(spread_metric)}"
        else:
            price = _summary_price(bundle, category, name)
            pct = _summary_pct(bundle, category, name)
            read = note

        last_value = price if isinstance(price, str) else _fmt_price(price)
        move_value = pct if isinstance(pct, str) and pct else _fmt_alert_pct(pct)
        table_rows.append((label, last_value, move_value, read))

    return _make_table(["Asset", "Last", "1D move", "Read"], table_rows)


def _render_hk_quick_checks(bundle: Dict[str, Any]) -> str:
    rows = bundle.get("hk_quick_checks", []) or []
    table_rows = [
        (
            item.get("metric", ""),
            item.get("value", ""),
            _status_label(str(item.get("status", ""))),
            _source_as_of(item),
            _truncate(item.get("note", ""), 88),
        )
        for item in rows
    ]
    return _make_table(["Check", "Value", "Status", "Source / as of", "Why it matters"], table_rows)


def _render_risk_dashboard(bundle: Dict[str, Any]) -> str:
    risk = ((bundle.get("attribution", {}) or {}).get("risk_dashboard", {}) or {})
    if not risk:
        return "Risk dashboard was unavailable."

    components = risk.get("components", []) or []
    lines = [
        f"- **Composite risk score:** {risk.get('score', 'N/A')}/100 | **Regime:** {risk.get('bucket', 'Mixed')}",
    ]
    if components:
        rows = [
            (
                item.get("label", ""),
                f"{item.get('delta', 0):+}",
                _truncate(item.get("evidence", ""), 80),
            )
            for item in components[:6]
        ]
        lines.append(_make_table(["Component", "Score impact", "Evidence"], rows))
    return "\n".join(lines)


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
    return "\n".join(f"- {item}" for item in bullets)


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

    lines: List[str] = [tracker.get("summary", "Flow evidence was not conclusive."), ""]

    metrics = tracker.get("key_metrics", []) or []
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
        lines.append("#### Local Flow and Funding Checks")
        lines.append(_make_table(["Metric", "Value", "Status", "Source / as of", "Read"], rows))
        lines.append("")

    flow_bullets = tracker.get("flow_bullets", []) or []
    if flow_bullets:
        lines.append("#### Flow Notes")
        lines.extend(f"- {item}" for item in flow_bullets[:6])
        lines.append("")

    stock_connect = (tracker.get("stock_connect", {}) or {}).get("data", {}) or {}
    southbound_active = ((stock_connect.get("southbound", {}) or {}).get("top_active", []) or [])[:8]
    if southbound_active:
        lines.append("#### Stock Connect Southbound Active Names")
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

    ah_premium = (tracker.get("ah_premium", {}) or {}).get("data", {}) or {}
    ah_rows = ah_premium.get("top_premium", []) or []
    if ah_rows:
        lines.append("#### AH Premium Dispersion")
        rows = [
            (
                item.get("name", ""),
                item.get("a_ticker", ""),
                item.get("h_ticker", ""),
                f"{item.get('premium_pct', 0):+.2f}%",
                item.get("as_of", ""),
            )
            for item in ah_rows[:8]
        ]
        lines.append(_make_table(["Name", "A ticker", "H ticker", "Premium", "As of"], rows))
        lines.append("")

    watch_hits = tracker.get("short_sell_watchlist_hits", []) or []
    short_ratio = tracker.get("short_sell_top_ratio", []) or []
    short_value = tracker.get("short_sell_top_value", []) or []
    short_rows = watch_hits or short_ratio
    if short_rows:
        lines.append("#### HKEX Short-Selling Watch")
        rows = [
            (
                item.get("ticker") or f"{item.get('code', '')}.HK",
                item.get("name", ""),
                f"{item.get('short_ratio_pct', 0):.2f}%",
                _fmt_hkd_bn(item.get("short_turnover_hkd")),
                _fmt_hkd_bn(item.get("total_turnover_hkd")),
            )
            for item in short_rows[:8]
        ]
        lines.append(_make_table(["Ticker", "Name", "Short ratio", "Short turnover", "Total turnover"], rows))
        lines.append("")

    if short_value:
        leaders = "; ".join(
            f"{item.get('ticker') or item.get('code', '') + '.HK'} {item.get('name', '')} ({_fmt_hkd_bn(item.get('short_turnover_hkd'))})"
            for item in short_value[:4]
        )
        lines.append(f"- **Short-value leaders:** {leaders}.")

    lines.append("#### ETF Proxy Flow")
    lines.append(_render_hk_etf_proxy_table(bundle))
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
    meta = bundle.get("meta", {}) or {}
    overview = bundle.get("overview", {}) or {}
    day_mode = bundle.get("day_mode", {}) or {}
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    dashboard_md = f"![Research Dashboard]({dashboard_rel_path})\n" if dashboard_rel_path else ""
    market_quality_line = _render_market_data_quality(meta)
    market_quality_block = f"{market_quality_line}\n" if market_quality_line else ""
    quality = bundle.get("report_quality", {}) or {}
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
    hk_review_setup = llm_sections.get("hk_review_setup", "")
    macro_takeaway = llm_sections.get("macro_takeaway", "")
    layer_one_title = "Layer 1 | Scan (5-10 min)" if day_mode.get("is_trading_day", True) else "Layer 1 | Reset (5-10 min)"
    checklist_title = "Morning Checklist" if day_mode.get("is_trading_day", True) else "Next Open Checklist"
    today_ahead_title = "Today Ahead and Trading Calendar" if day_mode.get("is_trading_day", True) else "Next Session Outlook and Calendar"

    briefing_date = meta.get("briefing_date", meta.get("report_date", ""))
    review_date = meta.get("review_date", meta.get("report_date", ""))
    data_through = meta.get("data_through", meta.get("report_date", ""))
    global_date = meta.get("global_market_date", meta.get("effective_date", data_through))
    hk_date = meta.get("hk_data_date", data_through)

    return f"""# Morning Research Workbench | {briefing_date}

> Designed for a Hong Kong sell-side commute: Layer 1 `Scan`, Layer 2 `Deep Read`, Layer 3 `Thinking`  
> Mode: `{day_mode.get('label', 'Trading day')}` | {day_mode.get('note', '')}
> Briefing date: `{briefing_date}` | Review date: `{review_date}` | Global request: `{global_date}` | HK/China request: `{hk_date}` | Market effective date: `{meta.get('effective_date', '')}` | Generated at: `{meta.get('generated_at', '')}`
{market_quality_block}
{quality_line}

## Visual Dashboard

{dashboard_md}## {layer_one_title}

### 1.1 One-Line Market Pulse
{pulse}

### 1.2 Global Asset Price Dashboard
{_render_global_asset_dashboard(bundle)}

### 1.3 Hong Kong Key Data Quick Check
{_render_hk_quick_checks(bundle)}

### 1.4 Risk Dashboard
{_render_risk_dashboard(bundle)}

### 1.5 {checklist_title}
{_render_top_items(bundle.get('must_watch', []) or [], limit=6)}

## Layer 2 | Deep Read (20-30 min)

### 2.1 Overnight Overseas Market Review
{deep_read_setup}

{_render_selected_news(bundle)}

- **Hong Kong desk lens:** {hk_desk_view.get('leadership', '')}
{"".join(f"- {line}\n" for line in (hk_desk_view.get('lines', []) or []))}
{"".join(f"- Driver: {line}\n" for line in (llm_sections.get('overnight_drivers', []) or []))}
{"".join(f"- Hong Kong implication: {llm_sections.get('overnight_hk_implication', '')}\n" if llm_sections.get('overnight_hk_implication') else "")}
{"".join(f"- {line}\n" for line in ((overview.get('chart_read', {}) or {}).get('fx', []) or []))}
{"".join(f"- {line}\n" for line in ((overview.get('chart_read', {}) or {}).get('assets', []) or []))}

### 2.2 Hong Kong / A-share Previous-Day Review
{hk_review_setup}

- Use Hong Kong style leadership first: HSI / HSCEI / HSTECH tell you whether the market is broad-beta, old-economy, or growth-led.
- For cross-market read-through, keep CSI 300, ChiNext, FXI, and USD/CNH together rather than reading any single market in isolation.
- **LLM local leadership read:** {llm_sections.get('hk_local_leadership', '') or 'No extra leadership read was generated.'}
- **Follow-through check:** {llm_sections.get('hk_follow_through', '') or 'No extra follow-through check was generated.'}
- Hong Kong / offshore-China ETF proxies:
{_render_hk_etf_proxy_table(bundle)}

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
