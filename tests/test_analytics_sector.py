from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from professional.analytics_sector import build_sector_news_digest


def test_sector_news_digest_scores_coverage_and_sell_side() -> None:
    config = {
        "watchlists": {
            "core_coverage": [
                {"ticker": "0700.HK", "name": "Tencent"},
            ]
        }
    }
    sector_data = {
        "sector_news": {
            "Technology": [
                {
                    "title": "<b>Tencent</b> results beat expectations",
                    "summary": "Guidance improves for cloud and games.",
                    "source": "Reuters",
                    "link": "https://example.com/tencent",
                    "importance_score": 1.0,
                },
                {
                    "title": "Hardware supplier launches new product",
                    "summary": "Order visibility is improving.",
                    "source": "Newswire",
                    "link": "https://example.com/hardware",
                    "importance_score": 0.5,
                },
            ]
        },
        "analyst_changes": [
            {
                "ticker": "0700.HK",
                "firm": "Broker",
                "action": "Upgrade",
                "from_rating": "Neutral",
                "to_rating": "Buy",
                "previous_target": "380",
                "price_target": "430",
            }
        ],
        "earnings_calendar": [{"ticker": "0700.HK"}],
    }

    digest = build_sector_news_digest(sector_data, config)
    top = digest["graded_news"][0]

    assert top["title"] == "Tencent results beat expectations"
    assert top["grade"] == "A"
    assert top["horizon"] == "Short-term catalyst"
    assert top["score"] == 3.7
    assert digest["sell_side"][0]["summary"] == "Neutral -> Buy"
    assert digest["sell_side"][0]["target_change"] == "380 -> 430"
    assert digest["earnings_calendar"] == [{"ticker": "0700.HK"}]


def main() -> None:
    test_sector_news_digest_scores_coverage_and_sell_side()
    print("Analytics sector test passed")


if __name__ == "__main__":
    main()
