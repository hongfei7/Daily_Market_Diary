from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
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


def _fetch_single_watchlist(
    definition: WatchlistDefinition,
    news_limit: int,
    report_date: str,
) -> WatchlistSnapshot:
    snapshot = WatchlistSnapshot(definition=definition)
    try:
        ticker = yf.Ticker(definition.ticker)
        target_day = datetime.strptime(report_date, "%Y-%m-%d").date()
        hist = ticker.history(
            start=(target_day - timedelta(days=190)).isoformat(),
            end=(target_day + timedelta(days=1)).isoformat(),
            interval="1d",
        )
        if not hist.empty:
            valid = [index.date() <= target_day for index in hist.index]
            hist = hist.loc[valid]
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

    if snapshot.note:
        return snapshot
    snapshot.note = _compose_note(snapshot)
    return snapshot


def _compose_note(snapshot: WatchlistSnapshot) -> str:
    """Build a per-name note from every dimension that resolved.

    The previous if/elif chain branched on the daily move alone, so any two names
    up more than 2% received a word-for-word identical note and their range
    position was never mentioned. Composing the available facts keeps distinct
    names reading distinctly, and keeps the note tied to observable data.
    """
    move = snapshot.daily_change_pct
    pos = snapshot.range_position_pct

    if move is None:
        return "Market snapshot detail was not refreshed in the current public data run."

    # Assembled as: "<move>, <range position>, so <implication>."
    parts: List[str] = []

    # 1. What the tape did today, with the magnitude stated rather than implied.
    if move >= 2:
        parts.append(f"Up {move:.2f}% on the session")
    elif move <= -2:
        parts.append(f"Down {abs(move):.2f}% on the session")
    elif move > 0:
        parts.append(f"Marginally higher (+{move:.2f}%)")
    elif move < 0:
        parts.append(f"Marginally lower ({move:.2f}%)")
    else:
        parts.append("Unchanged on the session")

    # 2. Where that leaves it in its own 60-session range.
    if pos is not None:
        if pos >= 75:
            parts.append(f"in the top quartile of its 60-session range ({pos:.0f}%)")
        elif pos <= 25:
            parts.append(f"still in the bottom quartile of its 60-session range ({pos:.0f}%)")
        else:
            parts.append(f"mid-range at {pos:.0f}% of its 60-session band")

    # 3. The question the combination actually raises.
    if move >= 2 and pos is not None and pos <= 25:
        implication = "treat this as a bounce off a depressed base until it clears the range midpoint"
    elif move >= 2 and pos is not None and pos >= 75:
        implication = "the move extends an existing trend and carries profit-taking risk"
    elif move <= -2 and pos is not None and pos >= 75:
        implication = "check whether this is profit-taking or the start of a trend break"
    elif move <= -2 and pos is not None and pos <= 25:
        implication = "look for a fundamental or regulatory driver before treating it as value"
    elif move >= 2:
        implication = "check whether a catalyst can carry the move into the wider group"
    elif move <= -2:
        implication = "check for a fundamental or regulatory reason"
    else:
        implication = "use it to monitor marginal information changes rather than as a signal"

    note = f"{', '.join(parts)}, so {implication}."
    if snapshot.recent_news:
        count = len(snapshot.recent_news)
        note += f" {count} relevant headline{'s' if count != 1 else ''} attached."
    return note


def build_watchlist_digest(config: Dict[str, Any], report_date: str) -> Dict[str, List[Dict[str, Any]]]:
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
            executor.submit(_fetch_single_watchlist, definition, news_limit, report_date): label
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
