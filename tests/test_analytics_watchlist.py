from _bootstrap import ROOT  # noqa: F401

import pandas as pd

from professional import analytics_watchlist
from professional.analytics_watchlist import build_watchlist_digest


class _FakeTicker:
    news = [
        {
            "content": {
                "title": "<b>Nvidia lands new AI order</b>",
                "summary": "Demand&nbsp;<i>accelerated</i>",
                "provider": {"displayName": "Mock News"},
                "pubDate": "2026-04-13T12:00:00Z",
                "canonicalUrl": {"url": "https://example.com/news"},
            }
        }
    ]

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, period: str) -> pd.DataFrame:
        assert period == "6mo"
        return pd.DataFrame({"Close": [10.0, 11.0, 12.0, 13.0]})


def test_watchlist_digest_uses_ticker_snapshot_and_normalizes_news() -> None:
    original_ticker = analytics_watchlist.yf.Ticker
    analytics_watchlist.yf.Ticker = _FakeTicker
    try:
        digest = build_watchlist_digest(
            {
                "report": {"watchlist_news_limit": 1, "watchlist_workers": 1},
                "watchlists": {
                    "core_coverage": [
                        {
                            "ticker": "NVDA",
                            "name": "Nvidia",
                            "sector": "AI",
                            "thesis": "AI infrastructure demand",
                            "upcoming_catalyst": "Earnings",
                            "catalyst_date": "2026-04-20",
                        }
                    ],
                    "focus_pool": [],
                    "learning_pool": [],
                },
            },
            "2026-04-13",
        )
    finally:
        analytics_watchlist.yf.Ticker = original_ticker

    item = digest["Core coverage"][0]
    assert item["ticker"] == "NVDA"
    assert item["last_price"] == 13.0
    assert item["daily_change_pct"] == 8.33
    assert item["range_label"] == "Top of range"
    # The note is composed from the move, the range position and the headline
    # count, so two names with a similar move do not share identical copy.
    assert "Up 8.33% on the session" in item["note"]
    assert "top quartile of its 60-session range" in item["note"]
    assert "1 relevant headline attached" in item["note"]
    assert item["recent_news"][0]["title"] == "Nvidia lands new AI order"
    assert item["recent_news"][0]["summary"] == "Demand accelerated"
    assert item["recent_news"][0]["source"] == "Mock News"
    assert item["recent_news"][0]["url"] == "https://example.com/news"
    assert digest["Priority follow-up"] == []
    assert digest["Learning watchlist"] == []


if __name__ == "__main__":
    test_watchlist_digest_uses_ticker_snapshot_and_normalizes_news()
    print("Analytics watchlist test passed")
