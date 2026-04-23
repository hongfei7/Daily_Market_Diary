import os
import sys

from _bootstrap import ROOT  # noqa: F401

from professional.fact_checker import run_fact_check
from professional.report_builder import render_professional_report
from professional.report_quality import build_report_quality


def _bundle(pulse: str):
    return {
        "meta": {
            "briefing_date": "2026-04-14",
            "report_date": "2026-04-13",
            "global_market_date": "2026-04-13",
            "hk_data_date": "2026-04-13",
            "effective_date": "2026-04-13",
            "generated_at": "2026-04-14 06:00:00",
            "market_quality": {"available": 18, "total": 20, "fallback": [], "missing": [], "stale": []},
        },
        "overview": {
            "theme": "Risk-On backdrop with softer dollar pressure",
            "risk_regime": "Risk-On",
            "questions": [],
            "notes": [],
            "chart_read": {"fx": [], "assets": []},
        },
        "day_mode": {"label": "Trading day", "is_trading_day": True, "note": "Execution-oriented."},
        "hk_desk_view": {"leadership": "Growth-led", "lines": []},
        "market_summary": {
            "Equities": {
                "S&P 500": {"Pct Change": "1.20%", "Price": 5000},
                "Nasdaq 100": {"Pct Change": "1.60%", "Price": 18000},
                "Hang Seng Index": {"Pct Change": "0.80%", "Price": 18000},
                "Hang Seng TECH ETF": {"Pct Change": "1.10%", "Price": 5.0},
                "China Large-Cap (FXI)": {"Pct Change": "0.50%", "Price": 30.0},
            },
            "FX": {"DXY": {"Pct Change": "-0.35%", "Price": 104.2}, "USD/CNH": {"Pct Change": "-0.10%", "Price": 7.18}},
            "Rates": {"10Y Treasury": {"Pct Change": "-0.60%", "Price": 4.15}},
            "Commodities": {"Brent Crude": {"Pct Change": "1.80%", "Price": 85}, "Gold": {"Pct Change": "0.40%", "Price": 2300}},
            "Vol": {"VIX": {"Pct Change": "-3.50%", "Price": 16.2}},
        },
        "hk_local": {
            "short_selling_ratio": {"value": 15.0, "display_value": "15.00%", "status": "live_local"},
            "ah_premium_index": {"value": 45.25, "display_value": "45.25%", "status": "live_public"},
        },
        "china_rates": {"china_10y": {"display_value": "1.79%", "status": "live_public"}},
        "hk_quick_checks": [
            {"metric": "Short-selling ratio", "value": "15.00%", "status": "live_local", "source": "HKEX", "as_of": "2026-04-13", "note": "Official data."},
            {"metric": "AH premium index", "value": "45.25%", "status": "live_public", "source": "Yahoo", "as_of": "2026-04-13", "note": "Calculated data."},
            {"metric": "HK ETF flow proxy", "value": "N/A", "status": "proxy", "source": "ETF proxy", "as_of": "", "note": "Fallback."},
        ],
        "stock_connect": {"status": "ok"},
        "ah_premium": {"status": "ok"},
        "company_events": {"hkex_meta": {"status": "ok"}, "earnings": [], "ratings": [], "announcements": []},
        "llm_sections": {
            "one_line_market_pulse": pulse,
            "deep_read_setup": pulse,
            "overnight_drivers": ["Lower yields supported growth leadership.", "Softer dollar pressure helped offshore China sentiment."],
            "overnight_hk_implication": "Hong Kong should confirm the setup through HSTECH leadership and stable USD/CNH.",
            "hk_review_setup": "Hong Kong follow-through should be judged through style leadership rather than index direction alone. Flow confirmation matters because price action without Southbound support can fade quickly.",
            "hk_local_leadership": "Growth is leading if HSTECH outperforms HSCEI.",
            "hk_follow_through": "Watch Southbound active names, short-selling concentration, USD/CNH, and USD/HKD.",
            "task_meta": {
                "tasks": {
                    "overnight_review": {"status": "ok"},
                    "hk_review": {"status": "ok"},
                    "final_framing": {"status": "ok"},
                }
            },
        },
        "must_watch": [
            {
                "title": "Very long morning item",
                "bucket": "Market",
                "summary": (
                    "This is a deliberately long summary designed to exercise the report renderer's "
                    "length control while preserving a professional edit marker instead of making the "
                    "sentence look unfinished at the end of the line."
                ),
            }
        ],
        "macro_agenda": [],
        "sector_digest": {"graded_news": []},
        "high_frequency": [],
        "movers_digest": {"etf_flows": [], "flow_bullets": []},
        "flow_tracker": {},
        "attribution": {
            "dominant_drivers": [],
            "risk_dashboard": {
                "score": 70,
                "bucket": "Risk-on",
                "components": [{"label": "US beta", "delta": 1.4, "evidence": "S&P 500 +1.20%"}],
            },
        },
        "theme_deep_dive": {},
        "today_forward": {},
        "watchlists": {},
        "source_links": [],
        "risk": {},
        "report_config": {},
    }


def main() -> None:
    good = _bundle("S&P 500 rose 1.20% as softer dollar pressure and lower yields supported risk-on sentiment.")
    good["fact_check"] = run_fact_check(good)
    good["report_quality"] = build_report_quality(good)
    assert good["fact_check"]["status"] == "ok"
    assert good["fact_check"]["numeric_claims_checked"] >= 1
    assert good["report_quality"]["score"] > 75

    bad = _bundle("S&P 500 rose 3.00% as a stronger dollar and higher yields supported risk-off sentiment.")
    bad["fact_check"] = run_fact_check(bad)
    bad["report_quality"] = build_report_quality(bad)
    assert bad["fact_check"]["status"] == "warning"
    assert bad["fact_check"]["numeric_mismatches"]
    assert bad["fact_check"]["logic_warnings"]
    assert bad["report_quality"]["warnings"]

    report = render_professional_report(good, charts_section="_No charts._")
    assert "Report Quality and Validation" in report
    assert "LLM fact-check guardrail" in report
    assert "Quality score" in report
    assert "**Composite risk score:** `70/100`" in report
    assert "| Component | Score impact | Evidence |" in report
    assert "- **Composite risk score:**" not in report
    assert "#### Market Setup" in report
    assert "**Core tape.**" in report
    assert "#### Key Drivers" in report
    assert "#### Hong Kong Read-Through" in report
    assert "**Opening implication.**" in report
    assert "#### Style and Local Leadership" in report
    assert "#### Flow Confirmation" in report
    assert "No rotating theme configured" in report
    assert "..." not in report
    assert "[trimmed]" not in report

    print("Report quality test passed")


if __name__ == "__main__":
    main()
