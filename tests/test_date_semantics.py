import os
import sys
from types import SimpleNamespace

from _bootstrap import ROOT  # noqa: F401

from main_professional import _previous_calendar_day
from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.date_policy import build_day_mode, build_report_mode, previous_calendar_day, resolve_report_dates
from professional.report_builder import render_professional_report


def test_report_omits_absent_visual_sections_until_assets_exist() -> None:
    config = load_professional_config()
    bundle = build_professional_bundle(
        report_date="2026-04-18",
        briefing_date="2026-04-19",
        global_market_date="2026-04-18",
        hk_data_date="2026-04-17",
        config=config,
        market_data={"summary": {}, "meta": {"requested_date": "2026-04-18", "effective_date": "2026-04-17"}},
        chart_features={},
        macro_data={"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
        sector_data={},
        movers_data={},
        risk_data={},
        news_headlines=[],
    )

    report_without_visuals = render_professional_report(bundle, charts_section="")
    assert "### 3.3 Daily One Chart" not in report_without_visuals
    assert "### 3.4 Hong Kong Trend Pack" not in report_without_visuals

    bundle["daily_one_chart"] = {
        "title": "Daily One Chart",
        "rel_path": "charts/test_daily_one_chart.png",
        "caption": "Chart read. Daily lens.",
    }
    bundle["trend_pack"] = {
        "title": "Hong Kong Trend Pack",
        "rel_path": "charts/test_hk_trend_pack.png",
        "caption": "Trend read. Weekly lens.",
    }

    report_with_visuals = render_professional_report(bundle, charts_section="")
    assert "### 3.3 Daily One Chart" in report_with_visuals
    assert "### 3.4 Hong Kong Trend Pack" in report_with_visuals


def main() -> None:
    config = load_professional_config()

    assert previous_calendar_day("2026-04-20") == "2026-04-19"
    assert _previous_calendar_day("2026-04-20") == "2026-04-19"
    assert _previous_calendar_day("2026-04-18") == "2026-04-17"

    empty_args = {
        "date": "",
        "review_date": "",
        "global_date": "",
        "hk_date": "",
    }
    monday_dates = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-20", **empty_args), config)
    saturday_dates = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-18", **empty_args), config)
    tuesday_dates = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-21", **empty_args), config)
    thursday_dates = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-16", **empty_args), config)
    friday_dates = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-17", **empty_args), config)
    sunday_dates = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-19", **empty_args), config)

    assert monday_dates["review_date"] == "2026-04-19"
    assert monday_dates["global_market_date"] == "2026-04-19"
    assert monday_dates["hk_data_date"] == "2026-04-17"
    assert saturday_dates["review_date"] == "2026-04-17"
    assert saturday_dates["hk_data_date"] == "2026-04-17"
    assert tuesday_dates["review_date"] == "2026-04-20"
    assert tuesday_dates["hk_data_date"] == "2026-04-20"
    assert thursday_dates["review_date"] == "2026-04-15"
    assert thursday_dates["hk_data_date"] == "2026-04-15"
    assert friday_dates["review_date"] == "2026-04-16"
    assert friday_dates["hk_data_date"] == "2026-04-16"
    assert sunday_dates["review_date"] == "2026-04-18"
    assert sunday_dates["hk_data_date"] == "2026-04-17"

    monday_morning_review = build_day_mode("2026-04-19", config)
    saturday_morning_review = build_day_mode("2026-04-17", config)
    sunday_morning_review = build_report_mode("2026-04-18", config, briefing_date="2026-04-19")

    assert monday_morning_review["mode"] == "non_trading_event_watch"
    assert monday_morning_review["is_trading_day"] is False
    assert saturday_morning_review["mode"] == "trading_daily"
    assert saturday_morning_review["is_trading_day"] is True
    assert sunday_morning_review["mode"] == "weekly_review"
    assert sunday_morning_review["period_start"] == "2026-04-13"
    assert sunday_morning_review["period_end"] == "2026-04-17"

    holiday_config = {**config, "calendar": {**(config.get("calendar", {}) or {}), "closed_dates": ["2026-04-15"]}}
    holiday_mode = build_report_mode("2026-04-15", holiday_config, briefing_date="2026-04-16")
    assert holiday_mode["mode"] == "holiday_reopen_playbook"
    assert holiday_mode["next_hk_trading_day"] == "2026-04-16"

    bundle = build_professional_bundle(
        report_date="2026-04-19",
        briefing_date="2026-04-20",
        global_market_date="2026-04-19",
        hk_data_date="2026-04-17",
        config=config,
        market_data={"summary": {}, "meta": {"requested_date": "2026-04-19", "effective_date": "2026-04-17"}},
        chart_features={},
        macro_data={
            "calendar": {
                "released": [],
                "upcoming": [
                    {
                        "time": "20:30",
                        "country": "US",
                        "indicator": "Retail Sales MoM",
                        "forecast": "0.2%",
                        "previous": "0.1%",
                        "impact": "high",
                    }
                ],
            },
            "central_bank_events": [],
        },
        sector_data={},
        movers_data={},
        risk_data={},
        news_headlines=[],
    )
    assert bundle["meta"]["briefing_date"] == "2026-04-20"
    assert bundle["meta"]["review_date"] == "2026-04-19"
    assert bundle["meta"]["effective_date"] == "2026-04-17"
    assert bundle["date_semantics"]["hk_data_date"] == "2026-04-17"
    assert bundle["date_semantics"]["hk_cash_role"] == "last completed HK/China cash-market reference tape"
    assert bundle["day_mode"]["mode"] == "non_trading_event_watch"

    report = render_professional_report(bundle, charts_section="")
    assert "Date policy" in report
    assert "Still-Moving Global Financial Actions" in report
    assert "Non-Trading Focus Map" in report
    assert "Weekend / Holiday Event Docket" in report
    assert "Hong Kong Last Cash-Tape Quick Check (Reference)" in report
    assert "Last Available Hong Kong / A-share Tape (Reference Only)" in report
    assert "last-available reference" in report
    assert "### 3.3 Daily One Chart" not in report
    assert "### 3.4 Hong Kong Trend Pack" not in report

    weekly_bundle = build_professional_bundle(
        report_date="2026-04-18",
        briefing_date="2026-04-19",
        global_market_date="2026-04-18",
        hk_data_date="2026-04-17",
        config=config,
        market_data={"summary": {}, "meta": {"requested_date": "2026-04-18", "effective_date": "2026-04-17"}},
        chart_features={},
        macro_data={"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
        sector_data={},
        movers_data={},
        risk_data={},
        news_headlines=[],
    )
    weekly_report = render_professional_report(weekly_bundle, charts_section="")
    assert weekly_bundle["day_mode"]["mode"] == "weekly_review"
    assert weekly_bundle["weekly_review"]["window"]["start"] == "2026-04-13"
    assert "Weekly Review Map" in weekly_report
    assert "Next Week Checklist" in weekly_report
    assert "Weekly Cross-Asset Review" in weekly_report
    assert "Next-week desk questions" in weekly_report
    assert "Non-Trading Focus Map" not in weekly_report
    assert "### 3.4 Hong Kong Trend Pack" not in weekly_report

    weekly_bundle["weekly_review"]["trend_summary"] = {
        "status": "ok",
        "window": {"start": "2026-04-13", "end": "2026-04-17"},
        "rows": [
            {
                "signal": "Southbound flow",
                "weekly_change": "+4.0bn over 5 sessions",
                "latest": "+1.4bn on 2026-04-17",
                "read": "Southbound flow stayed net positive into the weekly close.",
            }
        ],
    }
    weekly_bundle["trend_pack"] = {
        "title": "Hong Kong Trend Pack",
        "rel_path": "charts/test_hk_trend_pack.png",
        "caption": "Trend read. Weekly lens.",
    }
    weekly_report_with_trends = render_professional_report(weekly_bundle, charts_section="")
    assert "Five-session trend evidence" in weekly_report_with_trends
    assert "Southbound flow" in weekly_report_with_trends
    assert "### 3.4 Hong Kong Trend Pack" in weekly_report_with_trends

    print("Date semantics test passed")


if __name__ == "__main__":
    main()
