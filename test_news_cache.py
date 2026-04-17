import hashlib
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "market_diary"))

from modules.data_fetcher import fetch_news
from modules.sector_news import fetch_sector_data


def main() -> None:
    with tempfile.TemporaryDirectory() as cache_dir:
        headlines = ["- Cached headline one", "- Cached headline two"]
        headline_path = os.path.join(cache_dir, "headlines_2026-04-17_10.json")
        with open(headline_path, "w", encoding="utf-8") as handle:
            json.dump(headlines, handle, ensure_ascii=False)
        assert fetch_news(max_per_feed=10, cache_dir=cache_dir, cache_key="2026-04-17") == headlines

    with tempfile.TemporaryDirectory() as cache_dir:
        config = {"watchlists": {"core_coverage": [{"ticker": "0700.HK", "name": "Tencent"}]}}
        cache_key = json.dumps({"date": "2026-04-16", "watchlists": config["watchlists"]}, sort_keys=True, ensure_ascii=True)
        digest = hashlib.md5(cache_key.encode("utf-8")).hexdigest()[:12]
        payload = {
            "sector_news": {"China Internet": []},
            "earnings_calendar": [],
            "analyst_changes": [],
            "hkex_announcements": {"status": "cached"},
            "formatted_text": "cached payload",
        }
        sector_path = os.path.join(cache_dir, f"sector_data_{digest}.json")
        with open(sector_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        assert fetch_sector_data("2026-04-16", config=config, cache_dir=cache_dir) == payload

    print("News cache test passed")


if __name__ == "__main__":
    main()
