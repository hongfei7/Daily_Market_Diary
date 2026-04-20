from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yfinance as yf

from modules.text_normalizer import normalize_news_text
from professional.attribution import build_attribution
from professional.models import WatchlistDefinition, WatchlistSnapshot


SECTOR_LABELS = {
    "Technology": "Technology",
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Energy": "Energy",
    "Consumer": "Consumer",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Other": "Other",
}


def _parse_pct(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace("%", "").replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _parse_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").strip()
        if not cleaned:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def _format_signed(value: Optional[float], digits: int = 2, suffix: str = "%") -> str:
    if value is None:
        return "N/A"
    return f"{value:+.{digits}f}{suffix}"


def _summary_item(summary: Dict[str, Any], category: str, name: str) -> Dict[str, Any]:
    item = (summary or {}).get(category, {}).get(name, {})
    return item if isinstance(item, dict) else {}


def _snapshot_row(summary: Dict[str, Any], category: str, name: str, label: str, question: str) -> Dict[str, Any]:
    item = _summary_item(summary, category, name)
    price = _parse_float(item.get("Price"))
    change_pct = _parse_pct(item.get("Pct Change"))
    return {
        "label": label,
        "category": category,
        "symbol": name,
        "price": price,
        "change_pct": change_pct,
        "question": question,
    }


def build_market_snapshot(summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracked = [
        ("Equities", "S&P 500", "S&P 500", "Risk appetite"),
        ("Equities", "Nasdaq 100", "Nasdaq 100", "Growth style"),
        ("Equities", "Euro Stoxx 50", "Euro Stoxx 50", "European risk appetite"),
        ("Equities", "Hang Seng Index", "Hang Seng Index", "Hong Kong beta"),
        ("Equities", "Hang Seng China Enterprises", "HSCEI", "China SOE / H-share tone"),
        ("Equities", "Hang Seng TECH ETF", "Hang Seng TECH", "Hong Kong growth / internet tone"),
        ("Equities", "China Large-Cap (FXI)", "China proxy (FXI)", "China sentiment"),
        ("Rates", "10Y Treasury", "US 10Y", "Rates path"),
        ("FX", "DXY", "DXY", "Global liquidity"),
        ("FX", "USD/CNH", "USD/CNH", "China FX sentiment"),
        ("FX", "USD/HKD", "USD/HKD", "Hong Kong funding lens"),
        ("Commodities", "Crude Oil", "WTI crude", "Geopolitics and demand"),
        ("Commodities", "Gold", "Gold", "Hedge / real yields"),
        ("Commodities", "Copper", "Copper", "Growth expectations"),
        ("Crypto", "Bitcoin", "Bitcoin", "Risk appetite supplement"),
        ("Vol", "VIX", "VIX", "Volatility and hedging"),
    ]
    return [_snapshot_row(summary, *row) for row in tracked]


def _get_row(rows: Iterable[Dict[str, Any]], label: str) -> Dict[str, Any]:
    for row in rows:
        if row.get("label") == label:
            return row
    return {}


def build_chart_read(chart_features: Dict[str, Any]) -> Dict[str, List[str]]:
    fx_bullets: List[str] = []
    asset_bullets: List[str] = []

    fx = chart_features.get("fx_composite", {}) or {}
    if fx.get("available"):
        turning_points = fx.get("turning_points", []) or []
        direction = "higher" if (fx.get("net_pp") or 0) > 0 else "lower"
        fx_bullets.append(
            f"USD composite moved {direction} by {_format_signed(fx.get('net_pp'), digits=2, suffix='pct')} intraday, with a {fx.get('range_pp', 'N/A')}pct range."
        )
        if turning_points:
            tp = " / ".join(f"{item['time']} {item['direction']}" for item in turning_points[:3])
            fx_bullets.append(f"Main turning points appeared around {tp}, suggesting event-driven repricing rather than a clean one-way USD move.")
    else:
        fx_bullets.append("USD chart features were unavailable, so the read should lean more on closing data and the event tape.")

    divergence = chart_features.get("divergence", {}) or {}
    if divergence:
        asset_bullets.append(
            f"The biggest cross-asset divergence was {divergence.get('best_asset')} versus {divergence.get('worst_asset')}, a spread of {divergence.get('spread_pp')}pct."
        )

    assets = chart_features.get("assets", {}) or {}
    for label in ("Gold", "Oil", "Bitcoin"):
        asset = assets.get(label, {}) or {}
        if asset.get("available"):
            asset_bullets.append(
                f"{label} moved {_format_signed(asset.get('net_pp'), digits=2, suffix='pct')} intraday with a {asset.get('range_pp', 'N/A')}pct high-low range."
            )

    if not asset_bullets:
        asset_bullets.append("Gold / oil / Bitcoin chart features were unavailable.")

    return {"fx": fx_bullets[:3], "assets": asset_bullets[:4]}


def build_market_overview(summary: Dict[str, Any], chart_features: Dict[str, Any]) -> Dict[str, Any]:
    rows = build_market_snapshot(summary)

    spx = _get_row(rows, "S&P 500").get("change_pct")
    ndx = _get_row(rows, "Nasdaq 100").get("change_pct")
    hsi = _get_row(rows, "Hang Seng Index").get("change_pct")
    hstech = _get_row(rows, "Hang Seng TECH").get("change_pct")
    fxi = _get_row(rows, "China proxy (FXI)").get("change_pct")
    dxy = _get_row(rows, "DXY").get("change_pct")
    us10y = _get_row(rows, "US 10Y").get("change_pct")
    vix = _get_row(rows, "VIX").get("change_pct")
    oil = _get_row(rows, "WTI crude").get("change_pct")
    gold = _get_row(rows, "Gold").get("change_pct")

    score = 0
    for value in (spx, ndx, hsi, hstech, fxi):
        if value is not None:
            score += 1 if value > 0.3 else -1 if value < -0.3 else 0
    if dxy is not None:
        score += -1 if dxy > 0.3 else 1 if dxy < -0.3 else 0
    if vix is not None:
        score += -1 if vix > 1.0 else 1 if vix < -1.0 else 0
    if us10y is not None:
        score += -1 if us10y > 0.5 else 1 if us10y < -0.5 else 0

    if score >= 2:
        risk_regime = "Risk-On"
    elif score <= -2:
        risk_regime = "Risk-Off"
    else:
        risk_regime = "Neutral"

    chart_read = build_chart_read(chart_features)
    usd_net = (chart_features.get("fx_composite", {}) or {}).get("net_pp") or 0
    usd_bias = "USD stronger" if usd_net > 0.15 else "USD softer" if usd_net < -0.15 else "USD range-bound"
    rate_bias = "lower yields supported duration" if (us10y or 0) < -0.5 else "higher yields pressured valuations" if (us10y or 0) > 0.5 else "rates were not the dominant driver"
    asset_div = chart_features.get("divergence", {}) or {}
    divergence_text = f"{asset_div.get('best_asset')} outperformed {asset_div.get('worst_asset')}" if asset_div else "cross-asset divergence stayed modest"
    theme = f"{risk_regime} backdrop with {usd_bias}, {rate_bias}, and {divergence_text}"

    notes = [
        f"Risk appetite snapshot: S&P 500 {_format_signed(spx)} / Nasdaq 100 {_format_signed(ndx)} / Hang Seng {_format_signed(hsi)} / HSTECH {_format_signed(hstech)} / FXI {_format_signed(fxi)}.",
        f"Rates and liquidity: US 10Y {_format_signed(us10y)} / DXY {_format_signed(dxy)} / VIX {_format_signed(vix)}.",
        f"Commodities and hedges: WTI {_format_signed(oil)} / Gold {_format_signed(gold)}.",
    ]

    key_questions = [
        f"Is risk appetite rising or fading? The overnight tape reads closer to `{risk_regime}`.",
        "Did rates expectations move? Focus on whether US 10Y, DXY, and growth style moved together.",
        "Were commodities the real story? Watch WTI and gold for geopolitics versus inflation signals.",
        "Is Hong Kong setup internet-led, SOE-led, or broad beta-led? Compare HSTECH, HSCEI, and HSI.",
        "Did offshore markets move China risk appetite? Focus on HSI, FXI, USD/CNH, and USD/HKD.",
    ]

    return {
        "theme": theme,
        "risk_regime": risk_regime,
        "snapshot_rows": rows,
        "notes": notes,
        "questions": key_questions,
        "chart_read": chart_read,
    }


def build_hk_desk_view(summary: Dict[str, Any]) -> Dict[str, Any]:
    rows = build_market_snapshot(summary)
    hsi = _get_row(rows, "Hang Seng Index").get("change_pct")
    hscei = _get_row(rows, "HSCEI").get("change_pct")
    hstech = _get_row(rows, "Hang Seng TECH").get("change_pct")
    fxi = _get_row(rows, "China proxy (FXI)").get("change_pct")
    usdcnh = _get_row(rows, "USD/CNH").get("change_pct")
    usdhkd = _get_row(rows, "USD/HKD").get("price")

    if hstech is not None and hscei is not None:
        if hstech - hscei > 0.5:
            leadership = "Hong Kong growth / internet led"
        elif hscei - hstech > 0.5:
            leadership = "State-owned / old-economy H-shares led"
        else:
            leadership = "Leadership was broad and balanced"
    else:
        leadership = "Leadership could not be determined cleanly"

    lines = [
        f"Hang Seng {_format_signed(hsi)} / HSCEI {_format_signed(hscei)} / HSTECH {_format_signed(hstech)}.",
        f"Offshore China proxy FXI {_format_signed(fxi)} and USD/CNH {_format_signed(usdcnh)} frame cross-border risk appetite.",
        f"USD/HKD last traded around {_fmt_price_for_hk(usdhkd)}, which keeps the Hong Kong funding lens in focus.",
    ]

    return {"leadership": leadership, "lines": lines}


def _fmt_price_for_hk(value: Any) -> str:
    price = _parse_float(value)
    if price is None:
        return "N/A"
    return f"{price:.4f}"


def _macro_profile(indicator: str, config: Dict[str, Any]) -> Dict[str, Any]:
    indicator_upper = indicator.upper()
    for key, profile in (config.get("macro_indicator_map") or {}).items():
        if key.upper() in indicator_upper:
            return profile
    return {
        "impact": "Watch whether it changes the day's core market narrative",
        "industries": ["To be assessed"],
        "beat_direction": "If the print beats, check whether the market reprices materially",
        "miss_direction": "If the print misses, watch for a style or rates pivot",
    }


def build_macro_agenda(report_date: str, macro_data: Dict[str, Any], config: Dict[str, Any]) -> List[Dict[str, Any]]:
    agenda: List[Dict[str, Any]] = []
    calendar = (macro_data or {}).get("calendar", {}) or {}
    released = calendar.get("released", []) or []
    upcoming = calendar.get("upcoming", []) or []
    cb_events = (macro_data or {}).get("central_bank_events", []) or []

    for item in released:
        profile = _macro_profile(item.get("indicator", ""), config)
        surprise = item.get("surprise", "inline")
        direction = profile["beat_direction"] if surprise == "beat" else profile["miss_direction"] if surprise == "miss" else "The print was broadly inline; focus on the second-order market reaction"
        agenda.append(
            {
                "date": report_date,
                "time": item.get("time", ""),
                "country": item.get("country", ""),
                "event": item.get("indicator", ""),
                "status": "Released",
                "impact": profile["impact"],
                "affected_industries": profile["industries"],
                "direction": direction,
                "attention": {"high": 5, "medium": 3, "low": 1}.get(item.get("impact", "medium"), 3),
                "score": 80 + {"high": 15, "medium": 8, "low": 3}.get(item.get("impact", "medium"), 8),
                "detail": f"Actual {item.get('actual')} / Forecast {item.get('forecast')} / Prior {item.get('previous')}",
            }
        )

    for item in upcoming:
        profile = _macro_profile(item.get("indicator", ""), config)
        agenda.append(
            {
                "date": report_date,
                "time": item.get("time", ""),
                "country": item.get("country", ""),
                "event": item.get("indicator", ""),
                "status": "Upcoming",
                "impact": profile["impact"],
                "affected_industries": profile["industries"],
                "direction": "The result will determine whether the current market theme continues",
                "attention": {"high": 5, "medium": 3, "low": 1}.get(item.get("impact", "medium"), 3),
                "score": 70 + {"high": 15, "medium": 8, "low": 3}.get(item.get("impact", "medium"), 8),
                "detail": f"Forecast {item.get('forecast')} / Prior {item.get('previous')}",
            }
        )

    for item in cb_events:
        agenda.append(
            {
                "date": report_date,
                "time": item.get("time", ""),
                "country": item.get("bank", ""),
                "event": f"{item.get('speaker', '')}: {item.get('title', '')}".strip(": "),
                "status": "Central bank",
                "impact": "Policy path, liquidity conditions, and cross-asset risk appetite",
                "affected_industries": ["Technology", "Financials", "Gold"],
                "direction": "Watch for any unexpectedly hawkish or dovish language",
                "attention": 5 if item.get("importance") == "high" else 3,
                "score": 78 if item.get("importance") == "high" else 68,
                "detail": item.get("event_type", "speech"),
            }
        )

    agenda.sort(key=lambda row: row.get("score", 0), reverse=True)
    return agenda


def _contains_coverage(text: str, coverage_terms: List[str]) -> bool:
    lowered = text.lower()
    return any(term and term.lower() in lowered for term in coverage_terms)


def _news_grade(score: float) -> str:
    if score >= 3.5:
        return "A"
    if score >= 1.8:
        return "B"
    return "C"


def _news_importance(text: str, sector_label: str) -> Tuple[str, str]:
    lowered = text.lower()
    if any(word in lowered for word in ("earnings", "guidance", "profit warning", "results", "outlook")):
        return "This can directly reshape earnings forecasts and the valuation framework", "Short-term catalyst"
    if any(word in lowered for word in ("upgrade", "downgrade", "price target", "rating")):
        return "This signals a marginal shift in sell-side consensus", "Short-term catalyst"
    if any(word in lowered for word in ("merger", "acquisition", "deal", "placement", "buyback")):
        return "Capital allocation or shareholder-return events can trigger a valuation reset", "Medium-term trend"
    if any(word in lowered for word in ("regulation", "approval", "policy", "rulemaking")):
        return f"Policy and regulation can reshape the {sector_label} industry logic", "Medium-term trend"
    if any(word in lowered for word in ("launch", "product", "order", "contract", "capacity")):
        return "This is more about order visibility and product-cycle validation", "Short-term catalyst"
    return f"Assess whether the story can propagate through the {sector_label} value chain", "Monitor"


def build_sector_news_digest(sector_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    watchlists = config.get("watchlists", {}) or {}
    coverage_terms: List[str] = []
    for bucket_items in watchlists.values():
        for item in bucket_items:
            coverage_terms.append(item.get("name", ""))
            coverage_terms.append(item.get("ticker", "").split(".")[0])

    graded_news: List[Dict[str, Any]] = []
    sector_news = (sector_data or {}).get("sector_news", {}) or {}
    for sector, news_items in sector_news.items():
        sector_label = SECTOR_LABELS.get(sector, sector)
        for news in news_items:
            title = normalize_news_text(news.get("title", ""), strip_html_tags=True)
            summary = normalize_news_text(news.get("summary", ""), strip_html_tags=True)
            text = f"{title} {summary}"
            score = float(news.get("importance_score", 0.0))
            if _contains_coverage(text, coverage_terms):
                score += 1.5
            if any(token in text.lower() for token in ("earnings", "guidance", "merger", "deal", "regulation", "approval", "upgrade", "downgrade", "buyback", "placement", "results")):
                score += 1.2
            why, horizon = _news_importance(text, sector_label)
            graded_news.append(
                {
                    "sector": sector_label,
                    "title": title,
                    "summary": summary,
                    "grade": _news_grade(score),
                    "why": why,
                    "impact_target": f"Map first into {sector_label} leaders and close peers",
                    "horizon": horizon,
                    "score": round(score, 2),
                    "source": normalize_news_text(news.get("source", ""), strip_html_tags=False),
                    "url": normalize_news_text(news.get("link", ""), strip_html_tags=False),
                }
            )

    graded_news.sort(key=lambda item: item.get("score", 0), reverse=True)

    sell_side: List[Dict[str, Any]] = []
    for change in (sector_data or {}).get("analyst_changes", []) or []:
        sell_side.append(
            {
                "ticker": change.get("ticker", ""),
                "firm": change.get("firm", ""),
                "action": change.get("action", ""),
                "summary": f"{change.get('from_rating', '')} -> {change.get('to_rating', '')}",
                "target_change": f"{change.get('previous_target', '')} -> {change.get('price_target', '')}",
            }
        )

    return {
        "graded_news": graded_news,
        "sell_side": sell_side,
        "earnings_calendar": (sector_data or {}).get("earnings_calendar", []) or [],
    }


def _tracker_interpretation(label: str, change_pct: Optional[float], chart_features: Dict[str, Any]) -> str:
    value = change_pct or 0.0
    if label == "DXY":
        if value > 0.3:
            return "A stronger dollar points to a more defensive or rate-differential driven tape."
        if value < -0.3:
            return "A softer dollar makes it easier for risk appetite and duration to extend."
    if label == "US 10Y":
        if value > 0.5:
            return "Higher yields can pressure long-duration and growth valuations."
        if value < -0.5:
            return "Lower yields tend to support growth style and gold."
    if label == "WTI crude":
        if value > 1.0:
            return "A strong crude move needs to be split between geopolitics and demand repair."
        if value < -1.0:
            return "Softer oil argues for a more cautious cyclical-growth read."
    if label == "Gold":
        fx_net = chart_features.get("fx_composite", {}).get("net_pp")
        if value > 0.5 and (fx_net or 0) > 0:
            return "Gold rising with the dollar looks more like geopolitical or pure hedge demand."
        if value > 0.5:
            return "A firm gold price suggests hedge demand or lower real yields."
    if label == "Copper" and value < -0.8:
        return "Weak copper argues for more caution on cyclical growth."
    if label == "Bitcoin" and value > 1.5:
        return "A strong crypto tape reinforces the risk-on read."
    if label == "VIX":
        if value > 2.0:
            return "Higher volatility argues for tighter sizing and tighter stops."
        if value < -2.0:
            return "Lower volatility signals easing stress in the tape."
    return "Keep tracking it to confirm whether the core daily narrative is holding."


def build_high_frequency_trackers(summary: Dict[str, Any], chart_features: Dict[str, Any]) -> List[Dict[str, Any]]:
    tracked = [
        ("Rates", "10Y Treasury", "US 10Y"),
        ("FX", "DXY", "DXY"),
        ("FX", "USD/CNH", "USD/CNH"),
        ("Commodities", "Crude Oil", "WTI crude"),
        ("Commodities", "Gold", "Gold"),
        ("Commodities", "Copper", "Copper"),
        ("Crypto", "Bitcoin", "Bitcoin"),
        ("Vol", "VIX", "VIX"),
    ]
    rows: List[Dict[str, Any]] = []
    for category, name, label in tracked:
        item = _summary_item(summary, category, name)
        if not item:
            continue
        change_pct = _parse_pct(item.get("Pct Change"))
        rows.append(
            {
                "label": label,
                "price": _parse_float(item.get("Price")),
                "change_pct": change_pct,
                "interpretation": _tracker_interpretation(label, change_pct, chart_features),
                "priority": abs(change_pct or 0.0),
            }
        )
    rows.sort(key=lambda row: row.get("priority", 0), reverse=True)
    return rows


def _classify_catalyst(catalyst: str) -> str:
    lowered = (catalyst or "").lower()
    if any(token in lowered for token in ("earnings", "guidance", "upgrade", "downgrade")):
        return "Announcement / expectations"
    if any(token in lowered for token in ("policy", "regulation", "approval")):
        return "Policy catalyst"
    if any(token in lowered for token in ("volume", "flow")):
        return "Flow-driven"
    return "Event-driven"


def _format_hkd_billions(value: Any) -> str:
    number = _parse_float(value)
    if number is None:
        return "N/A"
    return f"HK${number / 1_000_000_000:.1f}bn"


def build_movers_and_flows(movers_data: Dict[str, Any], risk_data: Dict[str, Any]) -> Dict[str, Any]:
    movers: List[Dict[str, Any]] = []
    premarket = (movers_data or {}).get("premarket_movers", {}) or {}
    for side in ("gainers", "losers"):
        for stock in (premarket.get(side, []) or [])[:3]:
            movers.append(
                {
                    "ticker": stock.get("ticker", ""),
                    "move": stock.get("change_pct"),
                    "title": f"{stock.get('ticker', '')} {stock.get('change_pct', 0):+.2f}%",
                    "summary": stock.get("catalyst", ""),
                    "attribution": _classify_catalyst(stock.get("catalyst", "")),
                    "score": abs(stock.get("change_pct", 0)) + 1.5,
                }
            )

    flow_bullets: List[str] = []
    etf_flows = (movers_data or {}).get("etf_flows", []) or []
    if etf_flows:
        top = etf_flows[:3]
        summary = "; ".join(
            f"{item['ticker']} {item['change_pct']:+.2f}% / volume ratio {item['volume_ratio']:.2f}x / {item['estimated_flow_direction']}"
            for item in top
        )
        flow_bullets.append(f"ETF flow anomalies were concentrated in: {summary}.")

    short_sell = (movers_data or {}).get("short_sell", {}) or {}
    short_sell_data = short_sell.get("data", {}) if isinstance(short_sell, dict) else {}
    short_sell_meta = short_sell.get("meta", {}) if isinstance(short_sell, dict) else {}
    short_market = (short_sell_data or {}).get("market", {}) or {}
    if short_market.get("short_ratio_pct") is not None:
        flow_bullets.append(
            "HKEX short selling: "
            f"{short_market.get('short_ratio_pct'):.2f}% of market turnover "
            f"({_format_hkd_billions(short_market.get('short_turnover_hkd'))} short turnover, "
            f"as of {short_sell_meta.get('effective_date', 'N/A')})."
        )

    top_short_ratio = (short_sell_data or {}).get("top_short_ratio", []) or []
    if top_short_ratio:
        leaders = "; ".join(
            f"{item.get('ticker') or str(item.get('code', '')) + '.HK'} {item.get('short_ratio_pct'):.2f}%"
            for item in top_short_ratio[:3]
        )
        flow_bullets.append(f"Highest stock-level short ratios were concentrated in: {leaders}.")

    options = (movers_data or {}).get("unusual_options", []) or []
    if options:
        opt = options[0]
        flow_bullets.append(
            f"Options activity centered on {opt.get('ticker')} {opt.get('option_type')}, with Vol/OI at {opt.get('volume_oi_ratio')}x and a {opt.get('sentiment')} bias."
        )

    sentiment = (risk_data or {}).get("sentiment_indicators", {}) or {}
    put_call = sentiment.get("put_call_ratio", {}) or {}
    if put_call:
        flow_bullets.append(
            f"Put/Call structure shows equity {put_call.get('equity')} / index {put_call.get('index')}, implying {put_call.get('interpretation', '')}."
        )

    return {
        "movers": movers,
        "flow_bullets": flow_bullets,
        "etf_flows": etf_flows[:8],
        "short_sell": short_sell,
        "short_sell_top_ratio": top_short_ratio[:8],
        "short_sell_top_value": ((short_sell_data or {}).get("top_short_value", []) or [])[:8],
        "short_sell_watchlist_hits": ((short_sell_data or {}).get("watchlist_hits", []) or [])[:8],
    }


def _stock_connect_bullet(stock_connect_data: Dict[str, Any]) -> List[str]:
    if not stock_connect_data or stock_connect_data.get("status") not in {"ok", "partial"}:
        return []
    data = stock_connect_data.get("data", {}) or {}
    meta = stock_connect_data.get("meta", {}) or {}
    bullets: List[str] = []
    southbound = data.get("southbound", {}) or {}
    northbound = data.get("northbound", {}) or {}
    if southbound.get("net_buy") is not None:
        bullets.append(
            "Stock Connect Southbound: "
            f"net {_format_hkd_billions(float(southbound.get('net_buy', 0)) * 1_000_000.0)} "
            f"on turnover {_format_hkd_billions(float(southbound.get('total_turnover') or 0) * 1_000_000.0)} "
            f"(as of {meta.get('effective_date', 'N/A')})."
        )
    elif southbound.get("total_turnover") is not None:
        bullets.append(
            "Stock Connect Southbound: "
            f"turnover {_format_hkd_billions(float(southbound.get('total_turnover') or 0) * 1_000_000.0)}; "
            "net-buy is not available in the public daily file."
        )
    if northbound.get("total_turnover") is not None:
        bullets.append(
            "Stock Connect Northbound: "
            f"turnover RMB{float(northbound.get('total_turnover') or 0) / 1_000:.1f}bn; "
            "public file provides total turnover/top active names, not full net-buy."
        )
    return bullets


def _ah_premium_bullet(ah_premium_data: Dict[str, Any]) -> List[str]:
    if not ah_premium_data or ah_premium_data.get("status") not in {"ok", "partial"}:
        return []
    data = ah_premium_data.get("data", {}) or {}
    average = data.get("average_premium")
    top = data.get("top_premium", []) or []
    if average is None:
        return []
    leader = top[0] if top else {}
    suffix = (
        f"; widest premium: {leader.get('name', '')} {leader.get('premium_pct', 'N/A')}%"
        if leader
        else ""
    )
    return [f"AH premium: simple covered-pair average {average:+.2f}%{suffix}."]


def _strip_html(text: str) -> str:
    return normalize_news_text(text, strip_html_tags=True)


def _extract_news_url(content: Dict[str, Any]) -> str:
    for key in ("canonicalUrl", "clickThroughUrl"):
        value = content.get(key) or {}
        if isinstance(value, dict) and value.get("url"):
            return value["url"]
    return ""


def _fetch_single_watchlist(definition: WatchlistDefinition, news_limit: int) -> WatchlistSnapshot:
    snapshot = WatchlistSnapshot(definition=definition)
    try:
        ticker = yf.Ticker(definition.ticker)
        hist = ticker.history(period="6mo")
        if not hist.empty:
            close = hist["Close"].dropna()
            if len(close) >= 2:
                last = float(close.iloc[-1])
                prev = float(close.iloc[-2])
                snapshot.last_price = round(last, 2)
                snapshot.daily_change_pct = round(((last / prev) - 1) * 100, 2) if prev else None
                window = close.tail(60)
                low = float(window.min())
                high = float(window.max())
                if high > low:
                    pos = ((last - low) / (high - low)) * 100
                    snapshot.range_position_pct = round(pos, 1)
                    if pos >= 75:
                        snapshot.range_label = "Top of range"
                    elif pos <= 25:
                        snapshot.range_label = "Bottom of range"
                    else:
                        snapshot.range_label = "Mid-range"

        recent_news: List[Dict[str, Any]] = []
        for raw in (getattr(ticker, "news", None) or [])[:news_limit]:
            content = raw.get("content", {}) if isinstance(raw, dict) else {}
            if not content:
                continue
            recent_news.append(
                {
                    "title": normalize_news_text(content.get("title", ""), strip_html_tags=True),
                    "summary": _strip_html(content.get("summary") or content.get("description") or ""),
                    "source": normalize_news_text(
                        (content.get("provider") or {}).get("displayName", "Yahoo Finance"),
                        strip_html_tags=False,
                    ),
                    "published": content.get("pubDate", ""),
                    "url": normalize_news_text(_extract_news_url(content), strip_html_tags=False),
                }
            )
        snapshot.recent_news = recent_news
    except Exception as exc:
        snapshot.note = f"Fetch failed: {type(exc).__name__}"

    move = snapshot.daily_change_pct
    pos = snapshot.range_position_pct
    if snapshot.note:
        return snapshot
    if move is None:
        snapshot.note = "No market snapshot was available; consider wiring in a dedicated data source."
    elif move >= 2:
        snapshot.note = "Short-term price strength is clear; fresh catalysts could trigger broader group follow-through."
    elif move <= -2:
        snapshot.note = "Short-term pressure is visible; check for a fundamental or regulatory reason."
    elif pos is not None and pos >= 75:
        snapshot.note = "The name sits near the top of its recent range, so watch for profit-taking under high expectations."
    elif pos is not None and pos <= 25:
        snapshot.note = "The name sits near the bottom of its recent range and is worth monitoring for a catalyst-led reversal."
    else:
        snapshot.note = "Positioning is neutral for now, so use it mainly to monitor marginal information changes."
    return snapshot


def build_watchlist_digest(config: Dict[str, Any], report_date: str) -> Dict[str, List[Dict[str, Any]]]:
    del report_date

    report_config = config.get("report", {}) or {}
    news_limit = int(report_config.get("watchlist_news_limit", 2))
    max_workers = int(report_config.get("watchlist_workers", 4))

    buckets = {
        "core_coverage": "Core coverage",
        "focus_pool": "Priority follow-up",
        "learning_pool": "Learning watchlist",
    }
    results: Dict[str, List[Dict[str, Any]]] = {label: [] for label in buckets.values()}

    tasks: List[Tuple[str, WatchlistDefinition]] = []
    for key, label in buckets.items():
        for item in (config.get("watchlists", {}) or {}).get(key, []) or []:
            tasks.append(
                (
                    label,
                    WatchlistDefinition(
                        ticker=item.get("ticker", ""),
                        name=item.get("name", ""),
                        sector=item.get("sector", ""),
                        bucket=label,
                        thesis=item.get("thesis", ""),
                        upcoming_catalyst=item.get("upcoming_catalyst", ""),
                        catalyst_date=item.get("catalyst_date", ""),
                    ),
                )
            )

    if not tasks:
        return results

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_fetch_single_watchlist, definition, news_limit): label
            for label, definition in tasks
        }
        for future in as_completed(future_map):
            label = future_map[future]
            try:
                snapshot = future.result()
                results[label].append(snapshot.to_dict())
            except Exception:
                continue

    for bucket_items in results.values():
        bucket_items.sort(key=lambda item: abs(item.get("daily_change_pct") or 0.0), reverse=True)

    return results


def build_catalyst_calendar(
    report_date: str,
    macro_agenda: List[Dict[str, Any]],
    sector_data: Dict[str, Any],
    risk_data: Dict[str, Any],
    watchlists: Dict[str, List[Dict[str, Any]]],
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    catalysts: List[Dict[str, Any]] = []
    report_config = (config or {}).get("report", {}) or {}
    window_days = int(report_config.get("catalyst_window_days", 7))
    base_date = datetime.strptime(report_date, "%Y-%m-%d")
    cutoff_date = base_date + timedelta(days=window_days)

    for item in macro_agenda:
        if item.get("status") in {"Upcoming", "Central bank"}:
            catalysts.append(
                {
                    "date": report_date,
                    "time": item.get("time", ""),
                    "event": item.get("event", ""),
                    "category": item.get("status", ""),
                    "impact": item.get("impact", ""),
                    "importance": item.get("attention", 3),
                    "score": item.get("score", 0),
                }
            )

    for item in (sector_data or {}).get("earnings_calendar", []) or []:
        earnings_date = item.get("date") or report_date
        catalysts.append(
            {
                "date": earnings_date,
                "time": item.get("time", ""),
                "event": f"{item.get('company', item.get('ticker', ''))} earnings",
                "category": "Earnings",
                "impact": f"EPS est. {item.get('eps_estimate')} / revenue est. {item.get('revenue_estimate')}",
                "importance": 4,
                "score": 72,
            }
        )

    for item in (risk_data or {}).get("upcoming_events", []) or []:
        catalysts.append(
            {
                "date": item.get("date", report_date),
                "time": "",
                "event": item.get("description", ""),
                "category": item.get("type", "Event"),
                "impact": "Watch whether it changes risk budgets or the theme-trading cadence",
                "importance": {"critical": 5, "high": 4, "medium": 3}.get(item.get("importance"), 2),
                "score": {"critical": 85, "high": 76, "medium": 65}.get(item.get("importance"), 55),
            }
        )

    for bucket_items in watchlists.values():
        for item in bucket_items:
            catalyst = item.get("upcoming_catalyst")
            if not catalyst:
                continue
            catalyst_date = item.get("catalyst_date") or report_date
            catalysts.append(
                {
                    "date": catalyst_date,
                    "time": "",
                    "event": f"{item.get('name')}: {catalyst}",
                    "category": item.get("bucket", "Watchlist"),
                    "impact": item.get("thesis", ""),
                    "importance": 3,
                    "score": 60,
                }
            )

    def sort_key(item: Dict[str, Any]) -> Tuple[datetime, str]:
        raw_date = item.get("date") or report_date
        try:
            parsed = datetime.strptime(raw_date, "%Y-%m-%d")
        except ValueError:
            parsed = datetime.strptime(report_date, "%Y-%m-%d")
        return parsed, str(item.get("time", ""))

    filtered: List[Dict[str, Any]] = []
    for item in catalysts:
        raw_date = item.get("date") or report_date
        try:
            parsed = datetime.strptime(raw_date, "%Y-%m-%d")
        except ValueError:
            parsed = base_date
        if base_date <= parsed <= cutoff_date:
            filtered.append(item)

    filtered.sort(key=lambda item: (sort_key(item)[0], sort_key(item)[1], -float(item.get("score", 0))))
    return filtered


def build_source_links(
    sector_digest: Dict[str, Any],
    watchlists: Dict[str, List[Dict[str, Any]]],
    report_config: Dict[str, Any],
    company_events: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    links: List[Dict[str, str]] = []
    seen = set()
    news_limit = int((report_config or {}).get("top_news_items", 8))
    story_limit = int((report_config or {}).get("watchlist_story_limit", 2))
    total_limit = int((report_config or {}).get("top_source_links", 15))

    for news in sector_digest.get("graded_news", [])[: max(news_limit, 1) + 4]:
        url = news.get("url")
        if url and url not in seen:
            seen.add(url)
            links.append({"label": news.get("title", ""), "url": url, "source": news.get("source", "")})

    for bucket_items in watchlists.values():
        for item in bucket_items:
            for news in item.get("recent_news", [])[:story_limit]:
                url = news.get("url")
                if url and url not in seen:
                    seen.add(url)
                    links.append({"label": news.get("title", ""), "url": url, "source": news.get("source", "")})

    for item in ((company_events or {}).get("announcements", []) or [])[:8]:
        url = item.get("url")
        if url and url not in seen:
            seen.add(url)
            links.append(
                {
                    "label": f"{item.get('ticker', '')} {item.get('title', '')}",
                    "url": url,
                    "source": item.get("source", "HKEXnews"),
                }
            )

    return links[:total_limit]


def build_must_watch(
    overview: Dict[str, Any],
    macro_agenda: List[Dict[str, Any]],
    sector_digest: Dict[str, Any],
    high_frequency: List[Dict[str, Any]],
    movers_digest: Dict[str, Any],
    catalysts: List[Dict[str, Any]],
    report_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    quick_limit = int((report_config or {}).get("quick_items_limit", 10))
    top_macro = int((report_config or {}).get("top_macro_events", 4))
    top_news = int((report_config or {}).get("top_news_items", 4))
    top_trackers = int((report_config or {}).get("top_high_frequency_items", 3))
    top_movers = int((report_config or {}).get("top_movers", 2))
    top_catalysts = int((report_config or {}).get("top_catalysts", 3))

    items: List[Dict[str, Any]] = [
        {
            "bucket": "Overnight regime",
            "title": overview.get("theme", ""),
            "summary": f"Start by deciding whether the day is about {overview.get('risk_regime')} conditions or a style pivot.",
            "score": 95,
        }
    ]

    for event in macro_agenda[:top_macro]:
        items.append(
            {
                "bucket": "Macro / policy",
                "title": f"{event.get('event')} ({event.get('status')})",
                "summary": f"{event.get('impact')} | Industries: {', '.join(event.get('affected_industries', []))}",
                "score": event.get("score", 0),
            }
        )

    for news in sector_digest.get("graded_news", [])[:top_news]:
        items.append(
            {
                "bucket": "News / announcements",
                "title": f"[{news.get('grade')}] {news.get('title')}",
                "summary": news.get("why", ""),
                "score": int(news.get("score", 0) * 20),
            }
        )

    for tracker in high_frequency[:top_trackers]:
        items.append(
            {
                "bucket": "High-frequency data",
                "title": f"{tracker.get('label')} {_format_signed(tracker.get('change_pct'))}",
                "summary": tracker.get("interpretation", ""),
                "score": int(abs(tracker.get("change_pct") or 0) * 10) + 40,
            }
        )

    for mover in movers_digest.get("movers", [])[:top_movers]:
        items.append(
            {
                "bucket": "Mover attribution",
                "title": mover.get("title", ""),
                "summary": f"{mover.get('attribution')} | {mover.get('summary')}",
                "score": int(mover.get("score", 0) * 10) + 30,
            }
        )

    for catalyst in catalysts[:top_catalysts]:
        items.append(
            {
                "bucket": "Catalysts",
                "title": catalyst.get("event", ""),
                "summary": catalyst.get("impact", ""),
                "score": catalyst.get("score", 0),
            }
        )

    items.sort(key=lambda item: item.get("score", 0), reverse=True)
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for item in items:
        title = item.get("title")
        if title in seen:
            continue
        seen.add(title)
        deduped.append(item)
        if len(deduped) >= quick_limit:
            break
    return deduped


def _format_price_or_na(value: Any, digits: int = 2) -> str:
    price = _parse_float(value)
    if price is None:
        return "N/A"
    return f"{price:.{digits}f}"


def _summarize_hk_etf_proxy(movers_data: Dict[str, Any]) -> str:
    hk_flows = []
    for item in (movers_data or {}).get("etf_flows", []) or []:
        ticker = item.get("ticker", "")
        if ticker in {"2800.HK", "2828.HK", "3033.HK"}:
            hk_flows.append(
                f"{ticker} {item.get('change_pct', 0):+.2f}% on {item.get('volume_ratio', 1):.2f}x volume ({item.get('estimated_flow_direction', 'n/a')})"
            )
    if not hk_flows:
        return "No live Hong Kong ETF proxy was available."
    return "; ".join(hk_flows[:3])


def _usdhkd_band_read(price: Any) -> str:
    level = _parse_float(price)
    if level is None:
        return "No live USD/HKD quote was available."
    if level >= 7.845:
        return "Close to the weak-side Convertibility Undertaking; keep HKMA liquidity operations in focus."
    if level <= 7.755:
        return "Close to the strong-side Convertibility Undertaking; watch for liquidity absorption or funding shifts."
    return "Inside the linked-exchange band without immediate boundary stress."


def _adapter_metric(adapter_data: Dict[str, Any], key: str) -> Dict[str, Any]:
    item = (adapter_data or {}).get(key, {})
    return item if isinstance(item, dict) else {}


def _quick_check_row(
    metric: str,
    value: str,
    status: str,
    note: str,
    source: str = "",
    as_of: str = "",
) -> Dict[str, str]:
    return {
        "metric": metric,
        "value": value,
        "status": status,
        "note": note,
        "source": source,
        "as_of": as_of,
    }


def build_hk_quick_checks(
    summary: Dict[str, Any],
    movers_data: Dict[str, Any],
    hk_desk_view: Dict[str, Any],
    hk_local_data: Dict[str, Any],
) -> List[Dict[str, str]]:
    usdhkd_item = _summary_item(summary, "FX", "USD/HKD")
    hsi_item = _summary_item(summary, "Equities", "Hang Seng Index")
    hstech_item = _summary_item(summary, "Equities", "Hang Seng TECH ETF")
    band_metric = _adapter_metric(hk_local_data, "linked_exchange_band")
    turnover_metric = _adapter_metric(hk_local_data, "main_board_turnover")
    turnover_ratio_metric = _adapter_metric(hk_local_data, "turnover_vs_20d")
    hibor_metric = _adapter_metric(hk_local_data, "hibor_1m")
    aggregate_metric = _adapter_metric(hk_local_data, "aggregate_balance")
    base_rate_metric = _adapter_metric(hk_local_data, "base_rate")
    short_selling_metric = _adapter_metric(hk_local_data, "short_selling_ratio")
    southbound_metric = _adapter_metric(hk_local_data, "southbound_net_flow")
    northbound_metric = _adapter_metric(hk_local_data, "northbound_net_flow")
    ah_metric = _adapter_metric(hk_local_data, "ah_premium_index")

    usdhkd_value = _format_price_or_na(usdhkd_item.get("Price"), digits=4)
    band_value = band_metric.get("display_value", "")
    band_note = band_metric.get("note", "")
    usdhkd_note = _usdhkd_band_read(usdhkd_item.get("Price"))
    if band_value and band_value != "N/A":
        usdhkd_display = f"{usdhkd_value} | band {band_value}"
        usdhkd_status = "live_hybrid"
        usdhkd_source = "Yahoo Finance + HKMA linked band"
        usdhkd_note = f"{usdhkd_note} {band_note}".strip()
        usdhkd_as_of = usdhkd_item.get("As Of", "") or band_metric.get("as_of", "")
    else:
        usdhkd_display = usdhkd_value
        usdhkd_status = "live_quote" if usdhkd_value != "N/A" else "unavailable"
        usdhkd_source = "Yahoo Finance"
        usdhkd_as_of = usdhkd_item.get("As Of", "")

    turnover_display = turnover_ratio_metric.get("display_value") or turnover_metric.get("display_value", "N/A")
    turnover_note = turnover_ratio_metric.get("note") or turnover_metric.get("note") or "Turnover context was unavailable."
    flow_value = "N/A"
    flow_note = southbound_metric.get("note", "") or northbound_metric.get("note", "")
    flow_status = "unavailable"
    flow_source = ""
    flow_as_of = ""
    if southbound_metric.get("status") != "unavailable" or northbound_metric.get("status") != "unavailable":
        flow_value = f"Southbound {southbound_metric.get('display_value', 'N/A')} | Northbound {northbound_metric.get('display_value', 'N/A')}"
        flow_status = "live_public"
        flow_source = southbound_metric.get("source", "") or northbound_metric.get("source", "")
        flow_as_of = southbound_metric.get("as_of", "") or northbound_metric.get("as_of", "")

    rows = [
        _quick_check_row(
            metric="Main Board turnover vs 20D",
            value=turnover_display,
            status=turnover_metric.get("status", turnover_ratio_metric.get("status", "unavailable")),
            note=turnover_note,
            source=turnover_metric.get("source", ""),
            as_of=turnover_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="HIBOR 1M",
            value=hibor_metric.get("display_value", "N/A"),
            status=hibor_metric.get("status", "unavailable"),
            note=hibor_metric.get("note", "Hong Kong funding data was unavailable."),
            source=hibor_metric.get("source", ""),
            as_of=hibor_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Aggregate Balance",
            value=aggregate_metric.get("display_value", "N/A"),
            status=aggregate_metric.get("status", "unavailable"),
            note=aggregate_metric.get("note", "Aggregate Balance data was unavailable."),
            source=aggregate_metric.get("source", ""),
            as_of=aggregate_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Base Rate / linked band",
            value=f"Base rate {base_rate_metric.get('display_value', 'N/A')} | band {band_metric.get('display_value', 'N/A')}",
            status=base_rate_metric.get("status", band_metric.get("status", "unavailable")),
            note=base_rate_metric.get("note", "") or band_metric.get("note", "") or "Linked-rate policy context was unavailable.",
            source=base_rate_metric.get("source", "") or band_metric.get("source", ""),
            as_of=base_rate_metric.get("as_of", "") or band_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="USD/HKD spot vs band",
            value=usdhkd_display,
            status=usdhkd_status,
            note=usdhkd_note,
            source=usdhkd_source,
            as_of=usdhkd_as_of,
        ),
        _quick_check_row(
            metric="Short-selling ratio",
            value=short_selling_metric.get("display_value", "N/A"),
            status=short_selling_metric.get("status", "unavailable"),
            note=short_selling_metric.get("note", "Short-selling ratio was unavailable."),
            source=short_selling_metric.get("source", ""),
            as_of=short_selling_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Southbound / Northbound net flow",
            value=flow_value,
            status=flow_status,
            note=flow_note or "Stock Connect public data were unavailable or incomplete for this report date.",
            source=flow_source,
            as_of=flow_as_of,
        ),
        _quick_check_row(
            metric="AH premium index",
            value=ah_metric.get("display_value", "N/A"),
            status=ah_metric.get("status", "unavailable"),
            note=ah_metric.get("note", "A/H premium could not be calculated from the configured public quote set."),
            source=ah_metric.get("source", ""),
            as_of=ah_metric.get("as_of", ""),
        ),
        _quick_check_row(
            metric="Hong Kong leadership",
            value=hk_desk_view.get("leadership", "N/A"),
            status="proxy",
            note="Use HSI / HSCEI / HSTECH relative moves as the opening style read.",
            source="HSI / HSCEI / HSTECH relative performance",
            as_of=usdhkd_item.get("As Of", ""),
        ),
        _quick_check_row(
            metric="HSI vs HSTECH",
            value=f"HSI {_format_signed(_parse_pct(hsi_item.get('Pct Change')))} | HSTECH {_format_signed(_parse_pct(hstech_item.get('Pct Change')))}",
            status="proxy",
            note="This is the fastest check for whether the day is beta-led or growth-led in Hong Kong.",
            source="Yahoo Finance summary snapshot",
            as_of=hsi_item.get("As Of", "") or hstech_item.get("As Of", ""),
        ),
        _quick_check_row(
            metric="HK ETF flow proxy",
            value=_summarize_hk_etf_proxy(movers_data),
            status="proxy",
            note="Use only as a fallback lens when official Stock Connect flow evidence is incomplete.",
            source="Hong Kong ETF volume proxy",
            as_of="",
        ),
    ]
    return rows


def build_company_event_digest(sector_data: Dict[str, Any], sector_digest: Dict[str, Any]) -> Dict[str, Any]:
    earnings_rows: List[Dict[str, Any]] = []
    for item in (sector_data or {}).get("earnings_calendar", []) or []:
        earnings_rows.append(
            {
                "ticker": item.get("ticker", ""),
                "company": item.get("company", ""),
                "time": item.get("time", ""),
                "comparison": f"EPS est. {item.get('eps_estimate', 'N/A')} | revenue est. {item.get('revenue_estimate', 'N/A')}",
            }
        )

    hkex_announcements = ((sector_data or {}).get("hkex_announcements", {}) or {}).get("data", {}) or {}
    announcement_rows: List[Dict[str, Any]] = []
    for item in (hkex_announcements.get("watchlist_hits", []) or hkex_announcements.get("top_announcements", []) or [])[:10]:
        announcement_rows.append(
            {
                "grade": item.get("grade", ""),
                "ticker": item.get("ticker", ""),
                "company": item.get("company", ""),
                "event_type": item.get("event_type", ""),
                "title": item.get("title", ""),
                "release_time": item.get("release_time", ""),
                "source": item.get("source", "HKEXnews"),
                "url": item.get("url", ""),
                "score": item.get("score", 0),
            }
        )

    return {
        "earnings": earnings_rows,
        "ratings": (sector_digest or {}).get("sell_side", []) or [],
        "announcements": announcement_rows,
        "hkex_meta": ((sector_data or {}).get("hkex_announcements", {}) or {}).get("meta", {}) or {},
        "ipo_watch": "IPO / grey-market / first-day performance should be wired to a dedicated Hong Kong ECM adapter.",
    }


def build_flow_tracker(
    hk_quick_checks: List[Dict[str, str]],
    movers_digest: Dict[str, Any],
    attribution: Dict[str, Any],
    stock_connect_data: Optional[Dict[str, Any]] = None,
    ah_premium_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    check_map = {item.get("metric", ""): item for item in hk_quick_checks or []}
    key_metrics = [
        check_map.get("Main Board turnover vs 20D", {}),
        check_map.get("Short-selling ratio", {}),
        check_map.get("Southbound / Northbound net flow", {}),
        check_map.get("AH premium index", {}),
        check_map.get("HIBOR 1M", {}),
        check_map.get("Aggregate Balance", {}),
        check_map.get("HK ETF flow proxy", {}),
    ]
    key_metrics = [item for item in key_metrics if item]

    return {
        "summary": attribution.get("flow_summary", "Flow evidence was not conclusive."),
        "key_metrics": key_metrics,
        "flow_bullets": (
            _stock_connect_bullet(stock_connect_data or {})
            + _ah_premium_bullet(ah_premium_data or {})
            + ((movers_digest or {}).get("flow_bullets", []) or [])
        ),
        "stock_connect": stock_connect_data or {},
        "ah_premium": ah_premium_data or {},
        "short_sell_top_ratio": (movers_digest or {}).get("short_sell_top_ratio", []) or [],
        "short_sell_top_value": (movers_digest or {}).get("short_sell_top_value", []) or [],
        "short_sell_watchlist_hits": (movers_digest or {}).get("short_sell_watchlist_hits", []) or [],
        "etf_flows": (movers_digest or {}).get("etf_flows", []) or [],
    }


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
    for bucket, items in (watchlists or {}).items():
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
        for bucket_items in (watchlists or {}).values():
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
            f"{tracker.get('label', '')} {_format_signed(tracker.get('change_pct'))}: {tracker.get('interpretation', '')}"
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


def build_today_forward(report_date: str, macro_agenda: List[Dict[str, Any]], catalysts: List[Dict[str, Any]]) -> Dict[str, Any]:
    today = report_date
    today_macro = [item for item in macro_agenda if item.get("date", today) == today][:6]
    today_catalysts = [item for item in catalysts if item.get("date", today) == today][:8]
    next_catalysts = catalysts[:10]

    focus_lines = []
    if today_macro:
        focus_lines.append(
            f"Macro: {today_macro[0].get('event', '')} is the first item to anchor the open and the rates/FX response."
        )
    if today_catalysts:
        focus_lines.append(
            f"Corporate / event: {today_catalysts[0].get('event', '')} is the cleanest same-day catalyst to prepare for."
        )
    if not focus_lines:
        focus_lines.append("The calendar is relatively light, so the market may trade more off positioning and overnight headlines.")

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


def build_day_mode(report_date: str, config: Dict[str, Any]) -> Dict[str, Any]:
    day = datetime.strptime(report_date, "%Y-%m-%d")
    calendar = config.get("calendar", {}) or {}
    closed_weekdays = set(int(value) for value in (calendar.get("closed_weekdays", []) or []))
    closed_dates = set(str(value) for value in (calendar.get("closed_dates", []) or []))
    is_closed = day.weekday() in closed_weekdays or report_date in closed_dates

    if is_closed:
        return {
            "mode": "non_trading_day",
            "label": "Non-trading day",
            "is_trading_day": False,
            "note": "Shift weight from execution prep toward synthesis, theme work, and next-session preparation.",
        }
    return {
        "mode": "trading_day",
        "label": "Trading day",
        "is_trading_day": True,
        "note": "Keep the report execution-oriented: what matters by the Hong Kong open, what can move leadership, and what needs fast follow-up.",
    }


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
    hk_local_metrics = ((hk_local_data or {}).get("data", {}) or {})
    china_rate_metrics = ((china_rates_data or {}).get("data", {}) or {})

    overview = build_market_overview(summary, chart_features)
    hk_desk_view = build_hk_desk_view(summary)
    morning_date = briefing_date or report_date
    global_date = global_market_date or market_meta.get("requested_date", report_date)
    local_date = hk_data_date or report_date
    day_mode = build_day_mode(report_date, config)
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
    today_forward = build_today_forward(morning_date, macro_agenda, catalysts)
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
    )

    return {
        "meta": {
            "report_date": report_date,
            "review_date": report_date,
            "briefing_date": morning_date,
            "data_through": local_date,
            "global_market_date": global_date,
            "hk_data_date": local_date,
            "requested_date": market_meta.get("requested_date", report_date),
            "effective_date": market_meta.get("effective_date", report_date),
            "summary_date": market_meta.get("summary_date", report_date),
            "market_quality": market_meta.get("market_quality", {}),
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "config_path": config.get("config_path", ""),
        },
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
        "reflection_prompts": reflection_prompts,
        "source_links": source_links,
        "must_watch": must_watch,
        "chart_features": chart_features,
        "raw_news_headlines": news_headlines[:20],
        "risk": risk_data,
        "report_config": report_config,
    }
