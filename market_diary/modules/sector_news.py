"""Sector and company news aggregation adapters."""

import hashlib
import json
import os
from typing import Any, Dict, List, Optional

import requests

from modules.adapter_hkex_announce import fetch_hkex_announcements
from modules.text_normalizer import normalize_news_text

try:
    import feedparser
except ImportError:
    feedparser = None


NEWS_REQUEST_TIMEOUT = (
    float(os.environ.get("DMD_NEWS_CONNECT_TIMEOUT_SECONDS", "3")),
    float(os.environ.get("DMD_NEWS_READ_TIMEOUT_SECONDS", "6")),
)
NEWS_USER_AGENT = "DailyMarketDiary/1.0"


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
                if any(keyword in text for keyword in keywords):
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

    def fetch_earnings_calendar(self, date: str) -> List[Dict]:
        """Return a placeholder earnings calendar."""
        return [
            {
                "ticker": "0700.HK",
                "company": "Tencent Holdings",
                "time": "After Market Close",
                "eps_estimate": "4.15 HKD",
                "revenue_estimate": "163.2bn HKD",
            },
            {
                "ticker": "0388.HK",
                "company": "Hong Kong Exchanges and Clearing",
                "time": "Before Market Open",
                "eps_estimate": "2.81 HKD",
                "revenue_estimate": "6.0bn HKD",
            },
        ]

    def fetch_analyst_changes(self, date: str) -> List[Dict]:
        """Return a placeholder analyst rating change list."""
        return [
            {
                "ticker": "9988.HK",
                "firm": "Morgan Stanley",
                "action": "Upgrade",
                "from_rating": "Equal Weight",
                "to_rating": "Overweight",
                "price_target": "105",
                "previous_target": "92",
            }
        ]

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
    cache_key = json.dumps({"date": date, "watchlists": watchlists}, sort_keys=True, ensure_ascii=True)
    cached = _load_cache(cache_dir, cache_key)
    if cached is not None:
        print("[sector_news] using cached sector/company payload")
        return cached

    aggregator = SectorNewsAggregator()
    sector_news = aggregator.fetch_sector_news(max_per_sector=3)
    earnings = aggregator.fetch_earnings_calendar(date)
    analyst_changes = aggregator.fetch_analyst_changes(date)
    hkex_announcements = fetch_hkex_announcements(
        report_date=date,
        watchlists=watchlists,
    )

    payload = {
        "sector_news": sector_news,
        "earnings_calendar": earnings,
        "analyst_changes": analyst_changes,
        "hkex_announcements": hkex_announcements,
        "formatted_text": aggregator.format_for_report(sector_news, earnings, analyst_changes),
    }
    _save_cache(cache_dir, cache_key, payload)
    return payload
