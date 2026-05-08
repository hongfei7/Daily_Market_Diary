from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Tuple

import yfinance as yf

from market_diary.modules.text_normalizer import normalize_news_text
from market_diary.professional.models import WatchlistDefinition, WatchlistSnapshot
from market_diary.professional.relevance import watchlist_story_relevance


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
            title = normalize_news_text(content.get("title", ""), strip_html_tags=True)
            summary = _strip_html(content.get("summary") or content.get("description") or "")
            relevance = watchlist_story_relevance(definition.name, definition.ticker, definition.sector, title, summary)
            if relevance < 2.0:
                continue
            recent_news.append(
                {
                    "title": title,
                    "summary": summary,
                    "source": normalize_news_text(
                        (content.get("provider") or {}).get("displayName", "Yahoo Finance"),
                        strip_html_tags=False,
                    ),
                    "published": content.get("pubDate", ""),
                    "url": normalize_news_text(_extract_news_url(content), strip_html_tags=False),
                }
            )
        snapshot.recent_news = recent_news
    except Exception:
        if snapshot.last_price is None and snapshot.daily_change_pct is None:
            snapshot.note = "Quote detail was not refreshed in the current public data run."

    move = snapshot.daily_change_pct
    pos = snapshot.range_position_pct
    if snapshot.note:
        return snapshot
    if move is None:
        snapshot.note = "Market snapshot detail was not refreshed in the current public data run."
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
