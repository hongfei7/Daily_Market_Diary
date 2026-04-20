import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "market_diary"))

from main_professional import _previous_calendar_day
from professional.analytics import build_day_mode, build_professional_bundle
from professional.config import load_professional_config


def main() -> None:
    config = load_professional_config()

    assert _previous_calendar_day("2026-04-20") == "2026-04-19"
    assert _previous_calendar_day("2026-04-18") == "2026-04-17"

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
        hk_data_date="2026-04-19",
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
    assert bundle["day_mode"]["mode"] == "non_trading_day"

    print("Date semantics test passed")


if __name__ == "__main__":
    main()
