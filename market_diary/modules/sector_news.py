"""Sector and company news aggregation adapters."""

import hashlib
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import requests

from market_diary.modules.adapter_hkex_announce import fetch_hkex_announcements
from market_diary.modules.provenance import provenance_record, unavailable_record
from market_diary.modules.text_normalizer import normalize_news_text

try:
    import feedparser
except ImportError:
    feedparser = None


NEWS_REQUEST_TIMEOUT = (
    float(os.environ.get("DMD_NEWS_CONNECT_TIMEOUT_SECONDS", "3")),
    float(os.environ.get("DMD_NEWS_READ_TIMEOUT_SECONDS", "6")),
)
NEWS_USER_AGENT = "DailyMarketDiary/1.0"


def _keyword_match(text: str, keywords: List[str]) -> bool:
    """Word-boundary keyword match.

    Substring matching made short keywords like "ai" match "said" and "ev"
    match "never". Lookarounds confine a keyword to a whole token (or a
    multi-word phrase like "electric vehicle") without splitting on hyphens.
    """
    for keyword in keywords:
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        if re.search(pattern, text):
            return True
    return False


def _cache_path(cache_dir: str, cache_key: str) -> str:
    digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
    return os.path.join(cache_dir, f"sector_data_{digest}.json")


def _load_cache(cache_dir: str, cache_key: str) -> Optional[Dict[str, Any]]:
    if not cache_dir:
        return None
    path = _cache_path(cache_dir, cache_key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _save_cache(cache_dir: str, cache_key: str, payload: Dict[str, Any]) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(cache_dir, cache_key)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _fetch_feed(url: str):
    response = requests.get(url, headers={"User-Agent": NEWS_USER_AGENT}, timeout=NEWS_REQUEST_TIMEOUT)
    response.raise_for_status()
    return feedparser.parse(response.content)


class SectorNewsAggregator:
    """Fetch and bucket headlines into a desk-friendly sector map."""

    SECTORS = {
        "China Internet": [
            "tencent",
            "alibaba",
            "meituan",
            "jd.com",
            "internet",
            "e-commerce",
            "gaming",
            "cloud",
        ],
        "Financials": ["bank", "insurance", "broker", "exchange", "wealth", "payment"],
        "Autos and EV": ["auto", "ev", "electric vehicle", "battery", "byd", "tesla"],
        "Semiconductors and AI": ["semiconductor", "chip", "ai", "server", "gpu"],
        "Property and Construction": ["property", "developer", "mortgage", "real estate"],
        "Consumer": ["consumer", "retail", "luxury", "travel", "macau", "beverage"],
        "Healthcare": ["pharma", "biotech", "medical", "drug"],
        "Energy and Materials": ["oil", "gas", "coal", "copper", "lithium", "steel", "aluminum"],
        "Industrials": ["shipping", "logistics", "manufacturing", "industrial", "aerospace"],
    }

    NEWS_SOURCES = {
        "reuters_business": "http://feeds.reuters.com/reuters/businessNews",
        "reuters_markets": "http://feeds.reuters.com/reuters/marketsNews",
        "bloomberg_markets": "https://feeds.bloomberg.com/markets/news.rss",
        "cnbc_markets": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
        "wsj_markets": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    }

    def fetch_sector_news(self, max_per_sector: int = 3) -> Dict[str, List[Dict]]:
        """Fetch recent headlines and rank them by sector importance."""
        all_news = self._fetch_all_news()
        categorized = self._categorize_news(all_news)

        filtered: Dict[str, List[Dict]] = {}
        for sector, news_list in categorized.items():
            filtered[sector] = sorted(
                news_list,
                key=lambda item: item.get("importance_score", 0),
                reverse=True,
            )[:max_per_sector]

        return filtered

    def _fetch_all_news(self) -> List[Dict]:
        """Fetch raw headlines from the configured RSS sources."""
        if feedparser is None:
            return []

        all_news: List[Dict] = []
        for source_name, url in self.NEWS_SOURCES.items():
            try:
                feed = _fetch_feed(url)
                for entry in feed.entries[:20]:
                    title = normalize_news_text(entry.get("title", ""), strip_html_tags=True)
                    summary = normalize_news_text(entry.get("summary", ""), strip_html_tags=True, max_length=240)
                    if not title:
                        continue
                    all_news.append(
                        {
                            "title": title,
                            "summary": summary,
                            "link": normalize_news_text(entry.get("link", ""), strip_html_tags=False),
                            "published": entry.get("published", ""),
                            "source": normalize_news_text(source_name, strip_html_tags=False),
                        }
                    )
            except Exception as exc:
                print(f"[sector_news] Error fetching {source_name}: {exc}")

        return all_news

    def _categorize_news(self, news_list: List[Dict]) -> Dict[str, List[Dict]]:
        """Assign headlines to sectors using keyword rules."""
        categorized = {sector: [] for sector in self.SECTORS}

        for news in news_list:
            text = f"{news['title']} {news['summary']}".lower()
            news["importance_score"] = self._calculate_importance(news)

            matched = False
            for sector, keywords in self.SECTORS.items():
                if _keyword_match(text, keywords):
                    categorized[sector].append(news)
                    matched = True
                    break

            if not matched:
                categorized.setdefault("Other", []).append(news)

        return categorized

    def _calculate_importance(self, news: Dict) -> float:
        """Score headlines by event intensity and source quality."""
        score = 0.0
        text = f"{news['title']} {news['summary']}".lower()

        high_priority = [
            "earnings",
            "guidance",
            "merger",
            "acquisition",
            "regulation",
            "approval",
            "buyback",
            "placement",
            "profit warning",
            "tariff",
        ]
        medium_priority = ["deal", "contract", "partnership", "launch", "expansion", "pricing"]

        for keyword in high_priority:
            if keyword in text:
                score += 2.0

        for keyword in medium_priority:
            if keyword in text:
                score += 1.0

        source = news.get("source", "")
        if "bloomberg" in source:
            score += 1.5
        elif "reuters" in source:
            score += 1.2
        elif "wsj" in source or "cnbc" in source:
            score += 1.0

        return score

    def fetch_earnings_calendar(self, date: str, watchlists: Optional[Dict[str, Any]] = None) -> List[Dict]:
        """Best-effort earnings calendar from Yahoo Finance ticker calendars.

        For each configured watchlist ticker, read the next earnings date from
        ``yf.Ticker(...).calendar`` and keep only dates on or after ``date``.
        This is a public, unattributed-when-empty source: HK-listed tickers often
        return no calendar, so the list degrades to empty rather than fabricating
        a date. A dedicated, licensed earnings feed would be more complete.
        """
        if not watchlists:
            return []
        try:
            import yfinance as yf
        except Exception:
            return []

        try:
            target = datetime.strptime(date, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return []

        rows: List[Dict] = []
        seen_tickers = set()
        # Cap the number of yfinance lookups so this best-effort fetch never
        # pushes the sector-news step past its timeout.
        max_tickers = 8
        for bucket_items in (watchlists or {}).values():
            for item in (bucket_items or []):
                if not isinstance(item, dict):
                    continue
                if len(seen_tickers) >= max_tickers:
                    break
                ticker = str(item.get("ticker", "") or "").strip()
                if not ticker or ticker in seen_tickers:
                    continue
                seen_tickers.add(ticker)
                try:
                    calendar = yf.Ticker(ticker).calendar
                    raw_dates = calendar.get("Earnings Date", []) if isinstance(calendar, dict) else []
                    if isinstance(raw_dates, (str, int, float)) or hasattr(raw_dates, "date"):
                        raw_dates = [raw_dates]
                    future = []
                    for d in raw_dates:
                        try:
                            parsed = d.date() if hasattr(d, "date") else datetime.strptime(str(d)[:10], "%Y-%m-%d").date()
                        except (TypeError, ValueError):
                            continue
                        if parsed >= target:
                            future.append(parsed)
                    if not future:
                        continue
                    rows.append(
                        {
                            "date": min(future).isoformat(),
                            "company": str(item.get("name", "") or ticker),
                            "ticker": ticker,
                            "time": "",
                            "eps_estimate": None,
                            "revenue_estimate": None,
                            "as_of": date,
                            "date_confidence": "aggregator_reported",
                            "source": "Yahoo Finance ticker calendar",
                            "source_url": f"https://finance.yahoo.com/quote/{ticker}",
                        }
                    )
                except Exception:
                    # A flaky ticker must not take down the whole calendar.
                    continue
        return rows

    def fetch_analyst_changes(self, date: str) -> List[Dict]:
        """Return no rating claims until a dated, attributable source is configured."""
        return []

    def format_for_report(
        self,
        sector_news: Dict[str, List[Dict]],
        earnings: List[Dict],
        analyst_changes: List[Dict],
    ) -> str:
        """Format the news payload into a readable fallback block."""
        lines: List[str] = ["### Sector and Company News", ""]

        for sector, news_list in sector_news.items():
            if not news_list:
                continue
            lines.append(f"#### {sector}")
            lines.append("")
            for news in news_list:
                lines.append(f"- **{news['title']}**")
                if news.get("summary"):
                    lines.append(f"  {news['summary']}")
                lines.append(f"  *Source: {news['source']}*")
                lines.append("")

        if earnings:
            lines.append("#### Earnings Calendar")
            lines.append("")
            lines.append("| Ticker | Company | Timing | EPS Estimate | Revenue Estimate |")
            lines.append("|--------|---------|--------|--------------|------------------|")
            for item in earnings:
                lines.append(
                    f"| {item['ticker']} | {item['company']} | {item['time']} | "
                    f"{item['eps_estimate']} | {item['revenue_estimate']} |"
                )
            lines.append("")

        if analyst_changes:
            lines.append("#### Analyst Rating Changes")
            lines.append("")
            for change in analyst_changes:
                lines.append(
                    f"- **{change['ticker']}** | {change['firm']} | {change['action']} | "
                    f"{change['from_rating']} -> {change['to_rating']} | "
                    f"Target {change['previous_target']} -> {change['price_target']}"
                )
            lines.append("")

        return "\n".join(lines)


def fetch_sector_data(date: str, config: Optional[Dict[str, Any]] = None, cache_dir: str = "") -> Dict:
    """Public entry point for sector and company news."""
    watchlists = (config or {}).get("watchlists", {}) if isinstance(config, dict) else {}
    cache_key = json.dumps(
        {"schema": 2, "date": date, "watchlists": watchlists},
        sort_keys=True,
        ensure_ascii=True,
    )
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None:
        print("[sector_news] using cached sector/company payload")
        return cached

    aggregator = SectorNewsAggregator()
    sector_news = aggregator.fetch_sector_news(max_per_sector=3)
    earnings = aggregator.fetch_earnings_calendar(date, watchlists=watchlists)
    analyst_changes = aggregator.fetch_analyst_changes(date)
    hkex_announcements = fetch_hkex_announcements(
        report_date=date,
        watchlists=watchlists,
    )

    provenance = []
    news_items = [item for items in sector_news.values() for item in items]
    if news_items:
        provenance.append(
            provenance_record(
                source_name="Public market-news RSS feeds",
                source_url="https://feeds.bloomberg.com/markets/news.rss",
                as_of=date,
                source_type="public",
                status="ok",
                confidence=0.7,
                note="Headline aggregation only; company-event claims require a linked primary or licensed source.",
            )
        )
    else:
        provenance.append(
            unavailable_record(
                "Public market-news RSS feeds",
                date,
                "No sector headline passed the feed and relevance checks.",
            )
        )

    hkex_meta = (hkex_announcements or {}).get("meta", {}) or {}
    hkex_status = str((hkex_announcements or {}).get("status", "unavailable") or "unavailable")
    provenance.append(
        provenance_record(
            source_name=str(hkex_meta.get("source") or "HKEXnews"),
            source_url=str(hkex_meta.get("source_url") or "https://www1.hkexnews.hk/"),
            as_of=str(hkex_meta.get("effective_date") or date),
            source_type="official",
            status=hkex_status,
            confidence=0.95 if hkex_status == "ok" else 0.0,
            note="Official listed-company announcements.",
        )
    )

    has_news = bool(news_items)
    has_hkex = hkex_status == "ok"
    status = "ok" if has_news and has_hkex else "partial" if has_news or has_hkex else "unavailable"

    payload = {
        "status": status,
        "sector_news": sector_news,
        "earnings_calendar": earnings,
        "analyst_changes": analyst_changes,
        "earnings_calendar_status": "ok" if earnings else "unavailable",
        "analyst_changes_status": "unavailable",
        "hkex_announcements": hkex_announcements,
        "formatted_text": aggregator.format_for_report(sector_news, earnings, analyst_changes),
        "provenance": provenance,
    }
    _save_cache(cache_dir, cache_key, payload)
    return payload
