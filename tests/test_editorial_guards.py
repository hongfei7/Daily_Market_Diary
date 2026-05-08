from _bootstrap import ROOT  # noqa: F401

from professional.analytics_sector import build_sector_news_digest
from professional.report_blocks import _render_hk_review_block, _render_watchlists
from professional.report_sections import _render_hk_quick_checks, _render_selected_news


def main() -> None:
    config = {
        "watchlists": {
            "core_coverage": [
                {"name": "Tencent", "ticker": "0700.HK", "sector": "China Internet"},
                {"name": "SMIC", "ticker": "0981.HK", "sector": "Semiconductors and AI"},
            ]
        }
    }
    sector_data = {
        "sector_news": {
            "Semiconductors and AI": [
                {
                    "title": "United's CEO Is Here to Buy Your Struggling Airline",
                    "summary": "Airline consolidation is back in focus.",
                    "importance_score": 4.0,
                    "source": "Bloomberg",
                    "link": "https://example.com/united",
                },
                {
                    "title": "Intel cuts guidance as chip margins reset",
                    "summary": "Semiconductor sentiment weakens after the earnings update.",
                    "importance_score": 3.2,
                    "source": "CNBC",
                    "link": "https://example.com/intel",
                },
            ]
        }
    }
    digest = build_sector_news_digest(sector_data, config)
    titles = [item["title"] for item in digest["graded_news"]]
    assert "Intel cuts guidance as chip margins reset" in titles
    assert "United's CEO Is Here to Buy Your Struggling Airline" not in titles

    bundle = {
        "hk_desk_view": {
            "leadership": "Leadership was broad and balanced",
            "lines": ["Hang Seng +0.4% / HSCEI +0.6% / HSTECH +0.2%."],
        },
        "llm_sections": {
            "hk_local_leadership": "Value/SOE-led: HSCEI outperformed HSTECH and banks carried the tape.",
            "hk_follow_through": "Watch whether HSCEI extends leadership versus HSTECH.",
            "selected_news": [
                {
                    "headline": "Intel cuts guidance as chip margins reset",
                    "why_it_matters": "This can reset semiconductor earnings expectations.",
                    "hk_market_impact": "Relevant for Hong Kong semiconductor proxies and HSTECH sentiment.",
                },
                {
                    "headline": "United CEO signals potential airline M&A activity",
                    "why_it_matters": "Airline consolidation could resume.",
                    "hk_market_impact": "Lower direct relevance to Hong Kong. Indirect impact via travel sentiment only.",
                },
            ],
        },
        "hk_quick_checks": [
            {
                "metric": "Hong Kong leadership",
                "value": "Leadership was broad and balanced",
                "status": "proxy",
                "source": "HSI / HSCEI / HSTECH",
                "as_of": "2026-04-24",
                "note": "Use HSI / HSCEI / HSTECH relative moves as the opening style read.",
            }
        ],
        "watchlists": {
            "Core coverage": [
                {
                    "name": "Alibaba",
                    "ticker": "9988.HK",
                    "last_price": 130.4,
                    "daily_change_pct": -0.84,
                    "range_label": "Bottom of range",
                    "note": "Quote detail was not refreshed in the current public data run.",
                    "recent_news": [],
                }
            ]
        },
        "flow_tracker": {"stock_connect": {"data": {"southbound": {"top_active": [{"ticker": "0700.HK"}]}}}},
        "report_config": {},
    }

    quick = _render_hk_quick_checks(bundle)
    assert "State-owned / old-economy H-shares led" in quick
    review = _render_hk_review_block(bundle)
    assert "**Style leadership.** State-owned / old-economy H-shares led." in review
    assert "LLM local leadership read" not in review

    selected_news = _render_selected_news(bundle)
    assert "Intel cuts guidance as chip margins reset" in selected_news
    assert "United CEO signals potential airline M&A activity" not in selected_news

    watchlists = _render_watchlists(bundle)
    assert "Quote detail was not refreshed in the current public data run." in watchlists
    assert "worth monitoring for a |" not in watchlists
    assert "Fetch failed" not in watchlists

    print("Editorial guard test passed")


if __name__ == "__main__":
    main()
