import os
import sys

from _bootstrap import ROOT  # noqa: F401

from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.email_builder import build_email_html, build_email_subject, build_email_text


def _fixture():
    return {
        "market_data": {
            "summary": {
                "Equities": {
                    "S&P 500": {"Price": 5000, "Pct Change": "1.20%"},
                    "Hang Seng Index": {"Price": 17450, "Pct Change": "0.95%"},
                    "Hang Seng TECH ETF": {"Price": 4.88, "Pct Change": "1.30%"},
                },
                "Rates": {"10Y Treasury": {"Price": 4.15, "Pct Change": "-0.60%"}},
                "FX": {
                    "DXY": {"Price": 104.2, "Pct Change": "-0.35%"},
                    "USD/CNH": {"Price": 7.18, "Pct Change": "0.10%"},
                    "USD/HKD": {"Price": 7.82, "Pct Change": "0.00%"},
                },
                "Commodities": {"Brent Crude": {"Price": 85.0, "Pct Change": "1.80%"}},
                "Vol": {"VIX": {"Price": 16.2, "Pct Change": "-3.50%"}},
            },
            "meta": {
                "requested_date": "2026-04-13",
                "effective_date": "2026-04-13",
                "summary_date": "2026-04-13",
                "market_quality": {"available": 10, "total": 12, "fallback": [], "stale": [], "missing": ["Rates / China 10Y"]},
            },
        },
        "chart_features": {"fx_composite": {"available": False}, "assets": {}, "divergence": {}},
        "macro_data": {"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
        "sector_data": {"sector_news": {}, "earnings_calendar": [], "analyst_changes": []},
        "movers_data": {"premarket_movers": {"gainers": [], "losers": []}, "etf_flows": [], "unusual_options": [], "short_sell": {"status": "error", "data": {}}},
        "risk_data": {"upcoming_events": [], "sentiment_indicators": {}},
        "hk_local_data": {
            "status": "ok",
            "data": {
                "main_board_turnover": {"display_value": "HK$207.9bn", "status": "live_local", "source": "HKEX Daily Quotations", "as_of": "2026-04-13", "note": "Participation was active."},
                "turnover_vs_20d": {"display_value": "1.18x | +18% vs 20D", "status": "live_local", "source": "HKEX Daily Quotations", "as_of": "2026-04-13", "note": "Trailing 20-session average turnover was HK$176.3bn."},
                "hibor_1m": {"display_value": "2.23%", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Funding conditions were stable."},
                "aggregate_balance": {"display_value": "HK$54.4bn", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Liquidity remained ample."},
                "base_rate": {"display_value": "4.00%", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Base-rate anchor remained unchanged."},
                "linked_exchange_band": {"display_value": "7.7500 to 7.8500", "status": "live_local", "source": "HKMA Daily Figures - Interbank Liquidity", "as_of": "2026-04-13", "note": "Official USD/HKD band."},
            },
            "meta": {"report_date": "2026-04-13"},
        },
        "china_rates_data": {
            "status": "ok",
            "data": {
                "china_10y": {"display_value": "1.79%", "status": "live_public", "source": "Eastmoney Treasury Yield History", "as_of": "2026-04-13", "note": "China local rates anchor.", "change_display": "-1.2bp"},
                "cn_us_10y_spread": {"display_value": "-250.7bp", "status": "live_public", "source": "Eastmoney Treasury Yield History", "as_of": "2026-04-13", "note": "Relative carry lens.", "change_display": "-3.4bp"},
            },
            "meta": {"report_date": "2026-04-13"},
        },
    }


def main() -> None:
    data = _fixture()
    config = load_professional_config()
    config["watchlists"] = {"core_coverage": [], "focus_pool": [], "learning_pool": []}
    bundle = build_professional_bundle(
        report_date="2026-04-13",
        config=config,
        market_data=data["market_data"],
        chart_features=data["chart_features"],
        macro_data=data["macro_data"],
        sector_data=data["sector_data"],
        movers_data=data["movers_data"],
        risk_data=data["risk_data"],
        news_headlines=[],
        hk_local_data=data["hk_local_data"],
        china_rates_data=data["china_rates_data"],
    )
    bundle["llm_sections"] = {
        "one_line_market_pulse": "Hong Kong opened with a modestly constructive overseas setup but still needs local flow confirmation.",
        "deep_read_setup": "The market tone was constructive but not decisive, with the dollar softer and volatility contained.",
        "interview_answer": "The setup is mildly constructive. I would still want local flow confirmation before becoming more aggressive.",
    }

    subject = build_email_subject(bundle)
    text = build_email_text(bundle)
    html = build_email_html(bundle, dashboard_cid="research_dashboard")

    assert "Hong Kong Morning Briefing" in subject
    assert "Top checklist" in text
    assert "Deep-read setup" in html
    assert "cid:research_dashboard" in html
    assert "Hong Kong local checks" in text
    assert "Hong Kong local checks" in html

    print("Email delivery test passed")


if __name__ == "__main__":
    main()
