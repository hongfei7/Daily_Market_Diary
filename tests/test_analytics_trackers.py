from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from professional.analytics_trackers import build_high_frequency_trackers


def test_high_frequency_trackers_sort_and_interpret() -> None:
    summary = {
        "Rates": {
            "10Y Treasury": {
                "Price": 4.25,
                "Change Unit": "bp",
                "Change Value": 7.0,
                "Change Display": "+7.0bp",
            }
        },
        "FX": {
            "DXY": {"Price": 105.2, "Pct Change": "0.40%"},
            "USD/CNH": {"Price": 7.22, "Pct Change": "0.10%"},
        },
        "Commodities": {
            "Crude Oil": {"Price": 82.0, "Pct Change": "-1.30%"},
            "Copper": {"Price": 4.1, "Pct Change": "-0.90%"},
        },
        "Vol": {"VIX": {"Price": 19.5, "Pct Change": "2.50%"}},
    }
    chart_features = {"fx_composite": {"net_pp": 0.4}}

    rows = build_high_frequency_trackers(summary, chart_features)

    # Ranking is magnitude weighted by transmission to Hong Kong, so US 10Y
    # (rates channel, 0.7) outranks Copper (second-order macro, 0.4) despite a
    # smaller normalised move.
    assert [row["label"] for row in rows[:3]] == ["VIX", "WTI crude", "US 10Y"]
    assert rows[0]["priority"] == 2.5 * 0.6
    assert rows[0]["raw_magnitude"] == 2.5
    assert rows[0]["interpretation"] == "Higher volatility argues for tighter sizing and tighter stops."
    assert any(row["label"] == "US 10Y" and "Higher yields" in row["interpretation"] for row in rows)
    assert any(row["label"] == "DXY" and "stronger dollar" in row["interpretation"] for row in rows)


def test_large_move_in_weakly_linked_asset_ranks_below_semis() -> None:
    """The 2026-08-20 failure: Bitcoin +7.48% and Gold +4.92% led the checklist.

    Neither bears meaningfully on Hong Kong tech, yet both outranked the
    semiconductor complex that drives it.
    """
    summary = {
        "Crypto": {"Bitcoin": {"Price": 70000, "Pct Change": "7.48%"}},
        "Commodities": {"Gold": {"Price": 3400, "Pct Change": "4.92%"}},
        "Equities": {
            "Semiconductors (SOXX)": {"Price": 280, "Pct Change": "-3.10%"},
            "NVIDIA": {"Price": 180, "Pct Change": "-2.40%"},
        },
    }

    rows = build_high_frequency_trackers(summary, {})
    labels = [row["label"] for row in rows]

    assert labels.index("SOXX") < labels.index("Bitcoin")
    assert labels.index("NVDA") < labels.index("Gold")
    # Every row explains why it ranks where it does.
    assert all(row["relevance"] for row in rows)
    assert next(row for row in rows if row["label"] == "SOXX")["relevance"] == "direct read-through to HK tech"
    assert next(row for row in rows if row["label"] == "Bitcoin")["relevance"] == "weak HK linkage; context only"


def test_weights_can_be_overridden_from_config() -> None:
    summary = {"Crypto": {"Bitcoin": {"Price": 70000, "Pct Change": "1.00%"}}}
    default = build_high_frequency_trackers(summary, {})[0]
    boosted = build_high_frequency_trackers(summary, {}, weight_overrides={"Bitcoin": 1.0})[0]
    assert boosted["priority"] > default["priority"]
    assert boosted["hk_weight"] == 1.0


def test_unknown_instrument_gets_a_neutral_weight() -> None:
    from professional.analytics_trackers import DEFAULT_TRANSMISSION_WEIGHT, transmission_weight

    assert transmission_weight("Something Unlisted") == DEFAULT_TRANSMISSION_WEIGHT


def main() -> None:
    test_high_frequency_trackers_sort_and_interpret()
    print("Analytics trackers test passed")


if __name__ == "__main__":
    main()
