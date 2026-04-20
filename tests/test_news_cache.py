import hashlib
import json
import os
import shutil

from _bootstrap import ROOT  # noqa: F401

from modules.data_fetcher import fetch_news
from modules.sector_news import fetch_sector_data


def main() -> None:
    cache_root = ROOT / "tmp_test_news_cache"
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
    cache_root.mkdir()

    try:
        cache_dir = cache_root / "headlines"
        cache_dir.mkdir()
        headlines = ["- Cached headline one", "- Cached headline two"]
        headline_path = os.path.join(str(cache_dir), "headlines_2026-04-17_10.json")
        with open(headline_path, "w", encoding="utf-8") as handle:
            json.dump(headlines, handle, ensure_ascii=False)
        assert fetch_news(max_per_feed=10, cache_dir=str(cache_dir), cache_key="2026-04-17") == headlines

        cache_dir = cache_root / "sector"
        cache_dir.mkdir()
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
        sector_path = os.path.join(str(cache_dir), f"sector_data_{digest}.json")
        with open(sector_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        assert fetch_sector_data("2026-04-16", config=config, cache_dir=str(cache_dir)) == payload
    finally:
        shutil.rmtree(cache_root, ignore_errors=True)

    print("News cache test passed")


if __name__ == "__main__":
    main()
