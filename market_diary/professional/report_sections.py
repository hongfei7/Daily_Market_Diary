from __future__ import annotations

import re
from typing import Any, Dict, List, Sequence

from market_diary.professional.report_formatting import (
    _bundle_metric,
    _compact_source_as_of,
    _fmt_alert_pct,
    _fmt_pct,
    _fmt_price,
    _make_table,
    _status_label,
    _summary_pct,
    _summary_price,
    _truncate,
)
from market_diary.professional.relevance import canonical_hk_leadership, is_relevant_llm_story
from market_diary.professional.instruments import format_summary_change, summary_change


def _safe_sentence_clip(text: Any, limit: int = 160) -> str:
    sentence = " ".join(str(text or "").split()).strip()
    if len(sentence) <= limit:
        return sentence
    phrase = _truncate(sentence, limit, suffix="").strip()
    phrase = re.sub(r"\b(?:could|can|would|may|might|should|will)(?:\s+\w+){0,2}$", "", phrase, flags=re.IGNORECASE).strip()
    phrase = re.sub(
        r"\b(?:after|before|during|into|onto|above|below|around|than|via|through|against|despite)\s+(?:any|the|a|an|this|that|these|those|current|next)?$",
        "",
        phrase,
        flags=re.IGNORECASE,
    ).strip()
    phrase = re.sub(r"\b(?:and|or|but|with|without|to|from|for|of|the|a|an|because|whether|if)$", "", phrase, flags=re.IGNORECASE).strip()
    if phrase and phrase[-1] not in ".!?":
        phrase = phrase.rstrip(" ,;:-|") + "."
    return phrase


def _render_top_items(items: List[Dict[str, Any]], limit: int = 6) -> str:
    if not items:
        return "1. **Priority list pending**\n   Checklist items were not populated for this run."
    lines = []
    for idx, item in enumerate(items[:limit], 1):
        lines.append(f"{idx}. **{item.get('title', '')}**")
        lines.append(f"   {item.get('bucket', '')} | {_safe_sentence_clip(item.get('summary', ''), 175)}")
    return "\n".join(lines)


def _render_selected_news(bundle: Dict[str, Any]) -> str:
    llm_sections = bundle.get("llm_sections", {}) or {}
    items = [
        item
        for item in (llm_sections.get("selected_news", []) or [])
        if is_relevant_llm_story(item.get("headline", ""), item.get("hk_market_impact", ""))
    ]
    if not items:
        return ""
    lines = ["**Curated overnight stories**"]
    for idx, item in enumerate(items[:5], 1):
        lines.append(f"{idx}. **{_truncate(item.get('headline', ''), 90, suffix='')}**")
        lines.append(f"   Why it matters: {_safe_sentence_clip(item.get('why_it_matters', ''), 175)}")
        lines.append(f"   HK read-through: {_safe_sentence_clip(item.get('hk_market_impact', ''), 175)}")
    return "\n".join(lines)


def _signal_direction(value: Any, deadband: float = 0.05) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        cleaned = value.replace("**", "").replace("%", "").replace("bp", "").replace(",", "").strip()
        try:
            value = float(cleaned)
        except ValueError:
            return 0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0
    if number > deadband:
        return 1
    if number < -deadband:
        return -1
    return 0


def _cross_asset_move(bundle: Dict[str, Any], category: str, name: str) -> float | None:
    if category == "Rates":
        item = ((bundle.get("market_summary", {}) or {}).get(category, {}) or {}).get(name, {}) or {}
        value, unit = summary_change(item)
        return value if unit == "bp" else None
    return _summary_pct(bundle, category, name)


def _asset_interpretation(bundle: Dict[str, Any], name: str, move: Any) -> tuple[str, str]:
    direction = _signal_direction(move)
    spx = _cross_asset_move(bundle, "Equities", "S&P 500")
    nasdaq = _cross_asset_move(bundle, "Equities", "Nasdaq 100")
    hsi = _cross_asset_move(bundle, "Equities", "Hang Seng Index")
    hstech = _cross_asset_move(bundle, "Equities", "Hang Seng TECH ETF")
    us10y = _cross_asset_move(bundle, "Rates", "10Y Treasury")
    dxy = _cross_asset_move(bundle, "FX", "DXY")
    cnh = _cross_asset_move(bundle, "FX", "USD/CNH")
    copper = _cross_asset_move(bundle, "Commodities", "Copper")
    vix = _cross_asset_move(bundle, "Vol", "VIX")

    if name == "S&P 500":
        interpretation = (
            "US beta improved, raising the global risk floor for Hong Kong."
            if direction > 0
            else "US beta weakened, reducing the external risk cushion for Hong Kong."
            if direction < 0
            else "US beta was flat and offers little directional lead for Hong Kong."
        )
        check = "Confirm with Nasdaq breadth and a same-direction VIX move; invalidate if offshore-China proxies diverge."
    elif name == "Nasdaq 100":
        relative = None if nasdaq is None or spx is None else nasdaq - spx
        if relative is not None and relative > 0.15:
            interpretation = "Growth outperformed broad beta, a supportive style signal for Hong Kong internet and platform names."
        elif relative is not None and relative < -0.15:
            interpretation = "Growth lagged broad beta, warning that headline risk-on may not transmit cleanly to the 3033.HK ETF proxy."
        else:
            interpretation = "Growth tracked broad beta, so the move looks market-wide rather than duration-led."
        check = "Confirm through the 3033.HK ETF versus HSI leadership; higher US yields would weaken the read."
    elif name == "Hang Seng Index":
        interpretation = (
            "Hong Kong broad beta strengthened, but local flow must confirm whether the move is investable."
            if direction > 0
            else "Hong Kong broad beta weakened and needs offshore or policy support to stabilize."
            if direction < 0
            else "The headline index was flat; style leadership and flow matter more than index direction."
        )
        check = "Confirm with Southbound participation and breadth; invalidate if USD/CNH weakens while turnover fades."
    elif name == "Hang Seng TECH ETF":
        relative = None if hstech is None or hsi is None else hstech - hsi
        if relative is not None and relative > 0.1:
            interpretation = "Hong Kong growth led broad beta, improving the platform/internet style signal."
        elif relative is not None and relative < -0.1:
            interpretation = "Hong Kong growth lagged, so broad-index strength is not yet a duration signal."
        else:
            interpretation = "Growth and broad beta moved together; there is no decisive style rotation yet."
        check = "Confirm with stable CNH, Southbound buying and lower yields; invalidate on growth underperformance with rising yields."
    elif name == "10Y Treasury":
        interpretation = (
            "The yield proxy rose, tightening the discount-rate backdrop for long-duration Hong Kong growth."
            if direction > 0
            else "The yield proxy fell, easing the valuation headwind for duration-sensitive equities."
            if direction < 0
            else "The yield proxy was stable and did not materially change the valuation backdrop."
        )
        check = "Confirm through Nasdaq and 3033.HK ETF relative performance; invalidate if growth leads despite the opposing rate move."
    elif name == "China 10Y":
        interpretation = (
            "Higher local yields may indicate firmer growth or tighter financial conditions; price action alone cannot distinguish them."
            if direction > 0
            else "Lower local yields may reflect easing support or softer demand; the signal is ambiguous without growth confirmation."
            if direction < 0
            else "Local yields were stable and provide little incremental macro signal."
        )
        check = "Use CSI 300, copper and official credit/activity data to separate growth optimism from demand weakness."
    elif name == "CN-US 10Y spread":
        interpretation = (
            "The relative yield gap moved toward China, modestly easing the carry disadvantage."
            if direction > 0
            else "The relative yield gap moved further against China, increasing carry and FX sensitivity."
            if direction < 0
            else "The relative yield gap was broadly unchanged."
        )
        check = "Confirm through USD/CNH and foreign-risk proxies; invalidate if FX remains stable despite further spread pressure."
    elif name == "DXY":
        interpretation = (
            "A firmer dollar tightens the external-liquidity backdrop and can pressure offshore-China risk appetite."
            if direction > 0
            else "A softer dollar eases an external-liquidity headwind for Hong Kong risk assets."
            if direction < 0
            else "The dollar was range-bound and adds little directional information."
        )
        check = "Confirm with USD/CNH moving in the same risk direction; divergence reduces conviction."
    elif name == "USD/CNH":
        interpretation = (
            "A higher USD/CNH means a weaker offshore renminbi, a headwind for offshore-China sentiment."
            if direction > 0
            else "A stronger offshore renminbi supports the external-risk backdrop for Hong Kong equities."
            if direction < 0
            else "CNH was stable, removing an immediate FX impulse but not confirming equity direction."
        )
        check = "Confirm with FXI, the 3033.HK ETF proxy and Southbound flow; invalidate if equities move opposite to FX with strong local participation."
    elif name == "USD/HKD":
        interpretation = (
            "A move toward the weak-side convertibility boundary keeps Hong Kong funding sensitivity in focus."
            if direction > 0
            else "A move away from the weak-side boundary modestly eases linked-rate funding pressure."
            if direction < 0
            else "The spot rate was stable; boundary distance matters more than the daily move."
        )
        check = "Confirm with HIBOR and Aggregate Balance; spot alone is not a liquidity conclusion."
    elif name == "Brent Crude":
        interpretation = (
            "Higher oil raises the inflation and margin-cost question; whether it is growth-positive depends on copper and cyclicals."
            if direction > 0
            else "Lower oil eases cost pressure but may also reflect softer demand."
            if direction < 0
            else "Oil was stable and adds little incremental macro pressure."
        )
        check = "Copper moving with oil supports a demand read; divergence toward gold favors a supply/geopolitical explanation."
    elif name == "Gold":
        interpretation = (
            "Gold strength may reflect hedge demand or falling real yields; it is not a standalone risk-off signal."
            if direction > 0
            else "Gold weakness reduces the haven signal unless real yields or the dollar explain the move."
            if direction < 0
            else "Gold was stable and does not alter the hedge-demand read."
        )
        check = "Cross-check DXY, US yields and VIX before assigning a risk-regime interpretation."
    elif name == "Copper":
        interpretation = (
            "Copper strength supports the global-demand and China-cyclical read."
            if direction > 0
            else "Copper weakness challenges a broad growth-recovery narrative."
            if direction < 0
            else "Copper was flat and provides no strong growth confirmation."
        )
        check = "Confirm through China equities and activity data; invalidate if cyclicals and credit indicators disagree."
    elif name == "VIX":
        interpretation = (
            "Higher implied volatility raises the tail-risk premium and weakens risk-on conviction."
            if direction > 0
            else "Lower implied volatility reduces near-term stress, but is supportive only if equity breadth confirms."
            if direction < 0
            else "Volatility was stable and does not change the tail-risk assessment."
        )
        check = "Confirm with equity breadth and credit-sensitive assets; invalidate if lower VIX accompanies narrow or falling equities."
    else:
        interpretation = "The move is a monitoring input, not a conclusion on its own."
        check = "Require confirmation from related prices, local flow and dated fundamental evidence."

    # Keep currently observed cross-asset tensions visible in the most important rate/growth rows.
    if name == "Nasdaq 100" and _signal_direction(us10y) > 0 and _signal_direction(nasdaq) > 0:
        interpretation += " Growth advanced despite higher yields, showing momentum resilience but also higher reversal sensitivity."
    if name == "DXY" and _signal_direction(dxy) < 0 and _signal_direction(cnh) >= 0:
        interpretation += " CNH did not confirm the relief, so the liquidity signal is incomplete."
    if name == "Brent Crude" and _signal_direction(move) > 0 and _signal_direction(copper) < 0:
        interpretation += " Copper divergence leans away from a clean demand-recovery interpretation."
    if name == "S&P 500" and _signal_direction(spx) > 0 and _signal_direction(vix) < 0:
        interpretation += " Falling volatility confirms lower near-term stress."

    return interpretation, check


def _render_global_asset_dashboard(bundle: Dict[str, Any]) -> str:
    rows = [
        ("Equities", "S&P 500", "S&P 500", "US large-cap risk appetite"),
        ("Equities", "Nasdaq 100", "Nasdaq 100", "Growth and duration leadership"),
        ("Equities", "Hang Seng Index", "Hang Seng Index", "Hong Kong broad beta"),
        ("Equities", "Hang Seng TECH ETF", "Hang Seng TECH ETF (3033.HK)", "Listed proxy for Hong Kong growth / platform leadership"),
        ("Equities", "CSI 300", "CSI 300", "Mainland large-cap tone"),
        ("Rates", "10Y Treasury", "US 10Y", "Global discount-rate anchor"),
        ("Rates", "China 10Y", "China 10Y", "China local rates anchor"),
        ("Rates", "CN-US 10Y spread", "CN-US 10Y spread", "Relative carry and macro-pressure gauge"),
        ("FX", "DXY", "DXY", "Dollar liquidity impulse"),
        ("FX", "USD/CNH", "USD/CNH", "Offshore China risk appetite"),
        ("FX", "USD/HKD", "USD/HKD", "Linked-exchange regime stress check"),
        ("Commodities", "Brent Crude", "Brent crude", "Energy / geopolitics"),
        ("Commodities", "Gold", "COMEX gold", "Hedge demand / real yields"),
        ("Commodities", "Copper", "Copper", "Global growth proxy"),
        ("Vol", "VIX", "VIX", "US stress barometer"),
    ]

    table_rows = []
    hidden_assets: List[str] = []
    china_10y_metric = _bundle_metric(bundle, "china_rates", "china_10y")
    spread_metric = _bundle_metric(bundle, "china_rates", "cn_us_10y_spread")

    # Kept deliberately tight for the five-minute scan. SOXX earns a place
    # because the overnight semis tape leads Hong Kong tech more directly than
    # broad US beta; the rest of the AI/TMT chain has its own block in Layer 2.
    main_labels = {
        "S&P 500",
        "Nasdaq 100",
        "Semiconductors (SOXX)",
        "Hang Seng Index",
        "Hang Seng TECH ETF (3033.HK)",
        "US 10Y",
        "China 10Y",
        "DXY",
        "USD/CNH",
        "VIX",
    }
    secondary_assets: List[str] = []

    for category, name, label, note in rows:
        if name == "China 10Y":
            price = china_10y_metric.get("display_value", "N/A")
            pct = china_10y_metric.get("change_display", "")
            read = note
        elif name == "CN-US 10Y spread":
            price = spread_metric.get("display_value", "N/A")
            pct = spread_metric.get("change_display", "")
            read = note
        else:
            price = _summary_price(bundle, category, name)
            item = ((bundle.get("market_summary", {}) or {}).get(category, {}) or {}).get(name, {}) or {}
            pct = format_summary_change(item) if category == "Rates" else _summary_pct(bundle, category, name)
            read = note

        last_value = price if isinstance(price, str) else _fmt_price(price)
        if category == "Rates" and name == "10Y Treasury" and last_value != "N/A":
            last_value = f"{last_value}%"
        move_value = pct if isinstance(pct, str) and pct else _fmt_alert_pct(pct)
        if str(last_value).strip() == "N/A" and str(move_value).strip() in {"", "N/A"}:
            hidden_assets.append(label)
            continue
        if label not in main_labels:
            secondary_assets.append(label)
            continue
        interpretation_value = pct
        if category == "Rates" and name == "10Y Treasury":
            interpretation_value, _ = summary_change(item)
        interpretation, confirmation = _asset_interpretation(bundle, name, interpretation_value if interpretation_value not in (None, "") else move_value)
        table_rows.append(
            (
                label,
                f"{last_value} / {move_value}",
                _safe_sentence_clip(interpretation, 230),
                _safe_sentence_clip(confirmation, 185),
            )
        )

    if not table_rows:
        return "Market snapshot coverage was limited for this run; use the local-flow and rates tables below as the firmer evidence."
    lines: List[str] = []
    if hidden_assets:
        lines.append(
            f"_Coverage gate: {len(hidden_assets)} unavailable market fields are suppressed from the main table; source status remains in the appendix._"
        )
        lines.append("")
    lines.append(
        "_Decision-useful subset: each row states the implication and the next test. Descriptive secondary monitors are kept out of the five-minute scan._"
    )
    lines.append("")
    lines.append(_make_table(["Signal", "Last / move", "Interpretation", "Confirmation / invalidation"], table_rows))
    if secondary_assets:
        lines.append("")
        lines.append(f"_Secondary monitors retained in the source bundle: {', '.join(secondary_assets)}._")
    return "\n".join(lines)


def _pick_metrics_by_name(items: List[Dict[str, Any]], preferred_names: Sequence[str]) -> List[Dict[str, Any]]:
    selected: List[Dict[str, Any]] = []
    used: set[int] = set()
    for preferred in preferred_names:
        preferred_lower = preferred.lower()
        for idx, item in enumerate(items):
            if idx in used:
                continue
            metric = str(item.get("metric", "")).lower()
            if preferred_lower in metric or metric in preferred_lower:
                selected.append(item)
                used.add(idx)
                break
    return selected


def _resolved_hk_leadership(bundle: Dict[str, Any]) -> str:
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    return canonical_hk_leadership(hk_desk_view.get("leadership", ""), llm_sections.get("hk_local_leadership", ""))


def _resolved_hk_lens(bundle: Dict[str, Any]) -> str:
    """Prefer the deterministic, evidence-bearing lens over a style label."""
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    lens = str(hk_desk_view.get("lens", "") or "").strip()
    if lens:
        return lens
    return _resolved_hk_leadership(bundle)


def _render_hk_quick_checks(bundle: Dict[str, Any]) -> str:
    hk_desk_view = bundle.get("hk_desk_view", {}) or {}
    leadership_display = str(hk_desk_view.get("headline", "") or "").strip() or _resolved_hk_leadership(bundle)
    rows = bundle.get("hk_quick_checks", []) or []
    rows = _pick_metrics_by_name(
        rows,
        (
            "Main Board turnover vs 20D",
            "Southbound / Northbound net flow",
            "Short-selling ratio",
            "AH premium index",
            "USD/HKD spot vs band",
            "HIBOR 1M",
            "Hong Kong leadership",
        ),
    )
    hidden_count = sum(1 for item in rows if str(item.get("status", "")).lower() == "unavailable")
    rows = [item for item in rows if str(item.get("status", "")).lower() != "unavailable"]
    if not rows:
        return "Hong Kong local checks did not refresh enough main-table evidence for this run; use Section 2.3 for any available public-flow detail."
    table_rows = []
    for item in rows:
        table_rows.append(
            (
                item.get("metric", ""),
                leadership_display if item.get("metric") == "Hong Kong leadership" else item.get("value", ""),
                _status_label(str(item.get("status", ""))),
                _compact_source_as_of(item),
                _truncate(item.get("note", ""), 220),
            )
        )
    lines: List[str] = []
    if hidden_count:
        lines.append(f"_Coverage gate: {hidden_count} unavailable local checks are suppressed here and retained in validation metadata._")
        lines.append("")
    lines.append(_make_table(["Check", "Value", "Status", "Source / as of", "Why it matters"], table_rows))
    return "\n".join(lines)


def _render_risk_dashboard(bundle: Dict[str, Any]) -> str:
    risk = ((bundle.get("attribution", {}) or {}).get("risk_dashboard", {}) or {})
    if not risk:
        return "Risk dashboard was unavailable."

    components = risk.get("components", []) or []
    lines = [
        f"**Composite risk score:** `{risk.get('score', 'N/A')}/100` | **Regime:** `{risk.get('bucket', 'Mixed')}`",
    ]
    if components:
        rows = [
            (
                item.get("label", ""),
                f"{item.get('delta', 0):+}",
                _truncate(item.get("evidence", ""), 80, suffix=""),
            )
            for item in components[:4]
        ]
        lines.append("")
        lines.append(_make_table(["Component", "Score impact", "Evidence"], rows))
    return "\n".join(lines)


def _render_non_trading_focus(bundle: Dict[str, Any]) -> str:
    focus = bundle.get("non_trading_focus", {}) or {}
    if not focus:
        return ""

    lines: List[str] = ["**Non-Trading Focus Map**", f"- {focus.get('summary', '')}"]
    still_moving = focus.get("still_moving", []) or []
    if still_moving:
        rows = [
            (
                item.get("label", ""),
                item.get("category", ""),
                _fmt_price(item.get("price")),
                _fmt_pct(item.get("change_pct")),
                _truncate(item.get("interpretation", ""), 78, suffix=""),
            )
            for item in still_moving[:6]
        ]
        lines.append(_make_table(["Monitor", "Bucket", "Last", "Move", "Why it matters"], rows))

    event_watch = focus.get("event_watch", []) or []
    if event_watch:
        rows = [
            (
                item.get("channel", ""),
                _truncate(item.get("signal", ""), 66, suffix=""),
                _truncate(item.get("why", ""), 82, suffix=""),
                _truncate(item.get("next_check", ""), 82, suffix=""),
            )
            for item in event_watch[:8]
        ]
        lines.append("\n**Weekend / Holiday Event Docket**")
        lines.append(_make_table(["Channel", "Signal", "Why monitor", "Next check"], rows))

    action_items = focus.get("action_items", []) or []
    if action_items:
        lines.append("\n**Still-moving actions to track**")
        for item in action_items[:5]:
            lines.append(f"- **{item.get('bucket', '')}:** {_truncate(item.get('item', ''), 72, suffix='')} | {_truncate(item.get('read', ''), 96, suffix='')}")

    next_open = focus.get("next_open", []) or []
    if next_open:
        lines.append("\n**Next-open preparation**")
        lines.extend(f"- {line}" for line in next_open[:4])

    return "\n".join(lines)


def _render_weekly_review(bundle: Dict[str, Any]) -> str:
    weekly = bundle.get("weekly_review", {}) or {}
    if not weekly:
        return ""

    lines: List[str] = ["**Weekly Review Map**", f"- {weekly.get('summary', '')}"]
    method_note = weekly.get("method_note", "")
    if method_note:
        lines.append(f"- Method note: {method_note}")

    cross_assets = weekly.get("cross_assets", []) or []
    if cross_assets:
        rows = [
            (
                item.get("asset", ""),
                item.get("latest_move", "N/A"),
                _truncate(item.get("read", ""), 78, suffix=""),
            )
            for item in cross_assets[:8]
        ]
        lines.append("\n**Cross-asset weekly dashboard**")
        lines.append(_make_table(["Asset", "Latest move", "Weekly read"], rows))

    hk_tape = weekly.get("hk_tape", []) or []
    if hk_tape:
        rows = [
            (
                item.get("signal", ""),
                item.get("latest_move", "N/A"),
                _truncate(item.get("read", ""), 78, suffix=""),
            )
            for item in hk_tape[:6]
        ]
        lines.append("\n**Hong Kong / China weekly tape**")
        lines.append(_make_table(["Signal", "Latest move", "Read"], rows))

    trend_summary = weekly.get("trend_summary", {}) or {}
    trend_rows = trend_summary.get("rows", []) or []
    if trend_rows:
        window = trend_summary.get("window", {}) or {}
        lines.append("\n**Five-session trend evidence**")
        if window.get("start") and window.get("end"):
            lines.append(f"_Window: {window.get('start')} to {window.get('end')}_")
        rows = [
            (
                item.get("signal", ""),
                item.get("weekly_change", "N/A"),
                item.get("latest", "N/A"),
                _truncate(item.get("read", ""), 86, suffix=""),
            )
            for item in trend_rows[:6]
        ]
        lines.append(_make_table(["Signal", "Five-session change", "Latest", "Desk read"], rows))

    flow_lines = weekly.get("flow_lines", []) or []
    if flow_lines:
        lines.append("\n**Flow and attribution clues**")
        lines.extend(f"- {_truncate(line, 120, suffix='')}" for line in flow_lines[:4])

    desk_questions = weekly.get("desk_questions", []) or []
    if desk_questions:
        lines.append("\n**Next-week desk questions**")
        lines.extend(f"- {_truncate(line, 128, suffix='')}" for line in desk_questions[:6])

    developments = weekly.get("developments", []) or []
    if developments:
        rows = [
            (
                item.get("bucket", ""),
                _truncate(item.get("item", ""), 64, suffix=""),
                _truncate(item.get("read", ""), 88, suffix=""),
            )
            for item in developments[:6]
        ]
        lines.append("\n**Key developments to retain**")
        lines.append(_make_table(["Bucket", "Item", "Why it matters"], rows))

    next_week = weekly.get("next_week", []) or []
    if next_week:
        rows = [
            (
                item.get("date", ""),
                _truncate(item.get("event", ""), 64, suffix=""),
                _truncate(item.get("read", ""), 88, suffix=""),
            )
            for item in next_week[:8]
        ]
        lines.append("\n**Next-week playbook**")
        lines.append(_make_table(["Date", "Catalyst", "Desk read"], rows))

    return "\n".join(lines)
