"""The macro calendar and risk feed were unimplemented stubs.

Both returned ``[]`` unconditionally, so ``source_health`` reported them as
unavailable every day and the report said "The macro calendar was light for this
run" — which implied a quiet calendar rather than a missing source.

The replacement is rule-driven rather than scraped: Chinese and Hong Kong
releases follow stable monthly rules, so the schedule is auditable and has no
fragile page to break.
"""

from __future__ import annotations

from datetime import date

import _bootstrap  # noqa: F401

from market_diary.modules.macro_calendar import fetch_macro_data
from market_diary.modules.macro_schedule import CHANNELS, scheduled_events
from market_diary.modules.risk_radar import RiskRadar, fetch_risk_data


class TestMacroSchedule:
    def test_lpr_lands_on_the_twentieth(self):
        events = scheduled_events(date(2026, 8, 19), days_back=0, days_forward=5)
        lpr = [item for item in events if "LPR" in item["indicator"]]
        assert lpr and lpr[0]["date"] == "2026-08-20"
        assert lpr[0]["timing_confidence"] == "exact"

    def test_payrolls_lands_on_the_first_friday(self):
        events = scheduled_events(date(2026, 11, 2), days_back=0, days_forward=7)
        nfp = [item for item in events if "Payrolls" in item["indicator"]]
        assert nfp and nfp[0]["date"] == "2026-11-06"
        assert date.fromisoformat(nfp[0]["date"]).weekday() == 4

    def test_every_event_declares_a_transmission_channel(self):
        """A release that cannot be tied to a channel does not belong here."""
        events = scheduled_events(date(2026, 9, 8), days_back=2, days_forward=10)
        assert events
        for item in events:
            assert item["channel"] in CHANNELS
            assert item["channel_note"]

    def test_month_end_rules_clamp_to_real_dates(self):
        """A day-31 rule must not produce 31 February."""
        events = scheduled_events(date(2026, 2, 27), days_back=0, days_forward=3)
        for item in events:
            date.fromisoformat(item["date"])  # raises if invalid

    def test_events_are_split_into_released_and_upcoming(self):
        events = scheduled_events(date(2026, 11, 2), days_back=2, days_forward=5)
        assert {item["status"] for item in events} <= {"released", "upcoming"}
        for item in events:
            if item["status"] == "released":
                assert item["date"] < "2026-11-02"
            else:
                assert item["date"] >= "2026-11-02"


class TestMacroAdapter:
    def test_calendar_reports_partial_not_unavailable(self):
        payload = fetch_macro_data("2026-08-19")
        assert payload["status"] == "partial"
        assert payload["calendar"]["upcoming"]

    def test_no_forecast_or_actual_values_are_invented(self):
        payload = fetch_macro_data("2026-09-11")
        rows = payload["calendar"]["released"] + payload["calendar"]["upcoming"]
        assert rows
        for row in rows:
            assert row["actual"] == ""
            assert row["forecast"] == ""
            assert row["previous"] == ""

    def test_coverage_gap_is_stated_explicitly(self):
        payload = fetch_macro_data("2026-09-11")
        assert "not sourced" in payload["meta"]["coverage_note"]

    def test_malformed_date_degrades_quietly(self):
        payload = fetch_macro_data("not-a-date")
        assert payload["status"] == "unavailable"
        assert payload["calendar"]["upcoming"] == []


class TestRiskRadar:
    def test_levels_are_derived_from_the_price(self):
        levels = fetch_risk_data({"HSI": 25453.23})["technical_levels"]["HSI"]
        assert levels["nearest_support"] == 25000.0
        assert levels["nearest_resistance"] == 25500.0

    def test_usd_hkd_uses_the_convertibility_undertakings(self):
        """The peg, not a round-number grid, defines USD/HKD boundaries."""
        levels = fetch_risk_data({"USD/HKD": 7.8431})["technical_levels"]["USD/HKD"]
        assert levels["nearest_support"] == 7.75
        assert levels["nearest_resistance"] == 7.85

    def test_unknown_symbols_are_skipped_rather_than_guessed(self):
        assert "MYSTERY" not in fetch_risk_data({"MYSTERY": 42.0})["technical_levels"]

    def test_non_numeric_prices_do_not_crash(self):
        assert fetch_risk_data({"HSI": None, "SPX": "n/a"})["technical_levels"] == {}

    def test_events_come_from_the_same_schedule_as_the_calendar(self):
        """The risk feed and the macro section must not disagree."""
        events = RiskRadar().fetch_upcoming_events(7, reference=date(2026, 9, 8))
        scheduled = scheduled_events(date(2026, 9, 8), days_back=0, days_forward=7)
        assert {item["date"] for item in events} <= {item["date"] for item in scheduled}

    def test_unsourced_dimensions_stay_empty(self):
        payload = fetch_risk_data({"HSI": 25000})
        assert payload["geopolitical_risks"] == []
        assert payload["sentiment_indicators"] == {}
        assert "remain unsourced" in payload["meta"]["coverage_note"]
