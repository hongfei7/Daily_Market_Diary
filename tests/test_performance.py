from __future__ import annotations

from market_diary.professional.performance import build_performance_ledger, signal_from_bundle


def test_backtest_uses_next_available_close_and_costs() -> None:
    observations = [
        {"as_of": "2026-08-03", "prices": {"Hang Seng Index": 90.0}},
        {"as_of": "2026-08-04", "prices": {"Hang Seng Index": 100.0}},
        {"as_of": "2026-08-05", "prices": {"Hang Seng Index": 110.0}},
        {"as_of": "2026-08-06", "prices": {"Hang Seng Index": 99.0}},
    ]
    signals = [
        {
            "signal_id": "hk-beta:2026-08-03:2026-08-04",
            "report_date": "2026-08-04",
            "market_as_of": "2026-08-03",
            "signal": "Risk-on",
            "position": 1,
        }
    ]
    ledger = build_performance_ledger(
        observations=observations,
        signals=signals,
        benchmarks=("Hang Seng Index",),
        horizons=(1,),
        cost_bps=10,
    )
    result = ledger["benchmarks"]["Hang Seng Index"]
    outcome = result["outcomes"][0]
    assert outcome["entry_date"] == "2026-08-04"
    assert outcome["horizons"]["1"]["exit_date"] == "2026-08-05"
    assert outcome["horizons"]["1"]["directional_return_net"] == 0.098
    assert result["series"][0]["position"] == 0
    assert result["series"][1]["position"] == 1
    assert result["series"][1]["strategy_return_net"] == 0.099
    assert ledger["methodology"]["look_ahead_guard"] is True


def test_manual_review_signal_is_recorded_but_not_traded() -> None:
    bundle = {
        "meta": {"briefing_date": "2026-08-03", "effective_date": "2026-08-01"},
        "overview": {"risk_regime": "Risk-on", "theme": "Fixture"},
        "report_quality": {"release_recommendation": {"action": "manual_review"}},
        "source_health": {"status": "healthy"},
    }
    signal = signal_from_bundle(bundle)
    assert signal is not None
    assert signal["status"] == "blocked"
    assert signal["position"] == 0


if __name__ == "__main__":
    test_backtest_uses_next_available_close_and_costs()
    test_manual_review_signal_is_recorded_but_not_traded()
    print("Performance tests passed")
