import os
import sys
from types import SimpleNamespace

from _bootstrap import ROOT  # noqa: F401

from main_professional import _previous_calendar_day
from professional.analytics import build_professional_bundle
from professional.config import load_professional_config
from professional.date_policy import build_day_mode, previous_calendar_day, resolve_report_dates
from professional.report_builder import render_professional_report


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

    assert monday_dates["review_date"] == "2026-04-19"
    assert monday_dates["global_market_date"] == "2026-04-19"
    assert monday_dates["hk_data_date"] == "2026-04-17"
    assert saturday_dates["review_date"] == "2026-04-17"
    assert saturday_dates["hk_data_date"] == "2026-04-17"
    assert tuesday_dates["review_date"] == "2026-04-20"
    assert tuesday_dates["hk_data_date"] == "2026-04-20"

    monday_morning_review = build_day_mode("2026-04-19", config)
    saturday_morning_review = build_day_mode("2026-04-17", config)

    assert monday_morning_review["mode"] == "non_trading_day"
    assert monday_morning_review["is_trading_day"] is False
    assert saturday_morning_review["mode"] == "trading_day"
    assert saturday_morning_review["is_trading_day"] is True

    bundle = build_professional_bundle(
        report_date="2026-04-19",
        briefing_date="2026-04-20",
        global_market_date="2026-04-19",
        hk_data_date="2026-04-17",
        config=config,
        market_data={"summary": {}, "meta": {"requested_date": "2026-04-19", "effective_date": "2026-04-17"}},
        chart_features={},
        macro_data={"calendar": {"released": [], "upcoming": []}, "central_bank_events": []},
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
    assert bundle["day_mode"]["mode"] == "non_trading_day"

    report = render_professional_report(bundle, charts_section="")
    assert "Date policy" in report
    assert "Still-Moving Global Financial Actions" in report
    assert "Non-Trading Focus Map" in report
    assert "Hong Kong Last Cash-Tape Quick Check (Reference)" in report
    assert "Last Available Hong Kong / A-share Tape (Reference Only)" in report
    assert "last-available reference" in report

    print("Date semantics test passed")


if __name__ == "__main__":
    main()
