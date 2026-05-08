from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from professional.analytics_macro import build_macro_agenda


def test_macro_agenda_builds_and_sorts_event_types() -> None:
    config = {
        "macro_indicator_map": {
            "CPI": {
                "impact": "Rates and dollar sensitivity",
                "industries": ["Technology", "Internet"],
                "beat_direction": "Hot inflation pressures duration",
                "miss_direction": "Cool inflation helps growth",
            }
        }
    }
    macro_data = {
        "calendar": {
            "released": [
                {
                    "time": "20:30",
                    "country": "US",
                    "indicator": "CPI YoY",
                    "actual": "3.4%",
                    "forecast": "3.2%",
                    "previous": "3.1%",
                    "impact": "high",
                    "surprise": "beat",
                }
            ],
            "upcoming": [
                {
                    "time": "10:00",
                    "country": "CN",
                    "indicator": "Industrial Profits",
                    "forecast": "2.0%",
                    "previous": "1.5%",
                    "impact": "medium",
                }
            ],
        },
        "central_bank_events": [
            {
                "time": "22:00",
                "bank": "Federal Reserve",
                "speaker": "Chair",
                "title": "Policy Outlook",
                "importance": "high",
                "event_type": "speech",
            }
        ],
    }

    agenda = build_macro_agenda("2026-04-24", macro_data, config)

    statuses = [item["status"] for item in agenda]
    assert statuses[0] == "Released"
    assert sorted(statuses) == ["Central bank", "Released", "Upcoming"]
    assert agenda[0]["event"] == "CPI YoY"
    assert agenda[0]["impact"] == "Rates and dollar sensitivity"
    assert agenda[0]["direction"] == "Hot inflation pressures duration"
    central_bank = next(item for item in agenda if item["status"] == "Central bank")
    upcoming = next(item for item in agenda if item["status"] == "Upcoming")
    assert central_bank["event"] == "Chair: Policy Outlook"
    assert upcoming["affected_industries"] == ["Market-wide"]


def main() -> None:
    test_macro_agenda_builds_and_sorts_event_types()
    print("Analytics macro test passed")


if __name__ == "__main__":
    main()
