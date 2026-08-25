import os
import re
import sys
from types import SimpleNamespace

from _bootstrap import ROOT  # noqa: F401

from main_professional import _previous_calendar_day
from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.date_policy import (
    build_day_mode,
    build_report_mode,
    is_hk_trading_day,
    previous_calendar_day,
    previous_hk_trading_day,
    resolve_report_dates,
)
from professional.report_builder import render_professional_report


def _empty_args() -> dict:
    return {"date": "", "review_date": "", "global_date": "", "hk_date": ""}


def test_report_omits_absent_visual_sections_until_assets_exist() -> None:
    config = load_professional_config()
    bundle = build_professional_bundle(
        report_date="2026-04-18",
        briefing_date="2026-04-19",
        global_market_date="2026-04-17",
        hk_data_date="2026-04-17",
        config=config,
        market_data={"summary": {}, "meta": {"requested_date": "2026-04-17", "effective_date": "2026-04-17"}},
        chart_features={},
        macro_data={"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
        sector_data={},
        movers_data={},
        risk_data={},
        news_headlines=[],
    )

    report_without_visuals = render_professional_report(bundle, charts_section="")
    assert not re.search(r"^###\s+3\.\d+\s+Daily One Chart$", report_without_visuals, re.MULTILINE)
    assert not re.search(r"^###\s+3\.\d+\s+Hong Kong Trend Pack$", report_without_visuals, re.MULTILINE)

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
    assert re.search(r"^###\s+3\.\d+\s+Daily One Chart$", report_with_visuals, re.MULTILINE)
    assert re.search(r"^###\s+3\.\d+\s+Hong Kong Trend Pack$", report_with_visuals, re.MULTILINE)


def test_calendar_day_helpers() -> None:
    assert previous_calendar_day("2026-04-20") == "2026-04-19"
    assert _previous_calendar_day("2026-04-20") == "2026-04-19"
    assert _previous_calendar_day("2026-04-18") == "2026-04-17"


def test_resolve_report_dates_weekday_semantics() -> None:
    config = load_professional_config()

    monday = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-20", **_empty_args()), config)
    assert monday["review_date"] == "2026-04-19"          # Sunday (narrative day)
    assert monday["global_market_date"] == "2026-04-17"   # Friday (last global session, not Sunday)
    assert monday["hk_data_date"] == "2026-04-17"         # Friday (last HK session)

    sunday = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-19", **_empty_args()), config)
    assert sunday["review_date"] == "2026-04-18"
    assert sunday["global_market_date"] == "2026-04-17"
    assert sunday["hk_data_date"] == "2026-04-17"

    tuesday = resolve_report_dates(SimpleNamespace(briefing_date="2026-04-21", **_empty_args()), config)
    assert tuesday["review_date"] == "2026-04-20"
    assert tuesday["global_market_date"] == "2026-04-20"
    assert tuesday["hk_data_date"] == "2026-04-20"


def test_report_mode_keyed_on_briefing_day() -> None:
    config = load_professional_config()

    sunday = build_report_mode("2026-04-19", config)
    assert sunday["mode"] == "weekly_review"
    assert sunday["is_trading_day"] is False
    assert sunday["period_start"] == "2026-04-13"
    assert sunday["period_end"] == "2026-04-17"

    monday = build_report_mode("2026-08-24", config)
    assert monday["mode"] == "week_ahead"
    assert monday["week_start"] == "2026-08-24"
    assert monday["week_end"] == "2026-08-28"
    assert monday["last_hk_trading_day"] == "2026-08-21"
    assert monday["target_hk_session"] == "2026-08-24"
    assert monday["next_hk_trading_day"] == "2026-08-25"

    thursday = build_report_mode("2026-04-16", config)
    assert thursday["mode"] == "trading_daily"
    assert thursday["is_trading_day"] is True

    saturday = build_report_mode("2026-04-18", config)
    assert saturday["mode"] == "trading_daily"
    assert saturday["last_hk_trading_day"] == "2026-04-17"
    assert saturday["target_hk_session"] == "2026-04-20"
    assert saturday["next_hk_trading_day"] == "2026-04-20"


def test_holiday_modes() -> None:
    config = load_professional_config()
    # Easter Monday 2026-04-06 is a weekday HK market holiday.
    assert is_hk_trading_day("2026-04-06", config) is False
    assert previous_hk_trading_day("2026-04-07", config) == "2026-04-02"

    easter_monday = build_report_mode("2026-04-06", config)
    assert easter_monday["mode"] == "holiday_event_watch"
    assert easter_monday["last_hk_trading_day"] == "2026-04-02"
    assert easter_monday["target_hk_session"] == "2026-04-07"

    reopen = build_report_mode("2026-04-07", config)
    assert reopen["mode"] == "holiday_reopen_playbook"
    assert reopen["last_hk_trading_day"] == "2026-04-02"


def test_monday_week_ahead_bundle_and_report() -> None:
    config = load_professional_config()
    bundle = build_professional_bundle(
        report_date="2026-08-23",
        briefing_date="2026-08-24",
        global_market_date="2026-08-21",
        hk_data_date="2026-08-21",
        config=config,
        market_data={"summary": {}, "meta": {"requested_date": "2026-08-21", "effective_date": "2026-08-21"}},
        chart_features={},
        macro_data={"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
        sector_data={},
        movers_data={},
        risk_data={},
        news_headlines=[],
    )
    assert bundle["day_mode"]["mode"] == "week_ahead"
    assert bundle["week_ahead"]["week_start"] == "2026-08-24"
    report = render_professional_report(bundle, charts_section="")
    assert "Week Ahead" in report
    assert "Week Ahead Map" in report
    assert "This Week's Calendar" in report
    assert "Base Case / Risk Case" in report
    assert "What to Watch This Week" in report


def test_sunday_weekly_review_bundle_and_report() -> None:
    config = load_professional_config()
    bundle = build_professional_bundle(
        report_date="2026-04-18",
        briefing_date="2026-04-19",
        global_market_date="2026-04-17",
        hk_data_date="2026-04-17",
        config=config,
        market_data={"summary": {}, "meta": {"requested_date": "2026-04-17", "effective_date": "2026-04-17"}},
        chart_features={},
        macro_data={"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
        sector_data={},
        movers_data={},
        risk_data={},
        news_headlines=[],
    )
    assert bundle["day_mode"]["mode"] == "weekly_review"
    assert bundle["weekly_review"]["window"]["start"] == "2026-04-13"
    report = render_professional_report(bundle, charts_section="")
    assert "Weekly Review" in report
    assert "Weekly Review Map" in report
    assert "Next Week Checklist" in report


def main() -> None:
    test_calendar_day_helpers()
    test_report_omits_absent_visual_sections_until_assets_exist()
    test_resolve_report_dates_weekday_semantics()
    test_report_mode_keyed_on_briefing_day()
    test_holiday_modes()
    test_monday_week_ahead_bundle_and_report()
    test_sunday_weekly_review_bundle_and_report()
    print("Date semantics test passed")


if __name__ == "__main__":
    main()
