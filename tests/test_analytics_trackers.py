from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from professional.analytics_trackers import build_high_frequency_trackers


def test_high_frequency_trackers_sort_and_interpret() -> None:
    summary = {
        "Rates": {"10Y Treasury": {"Price": 4.25, "Pct Change": "0.70%"}},
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

    assert [row["label"] for row in rows[:3]] == ["VIX", "WTI crude", "Copper"]
    assert rows[0]["priority"] == 2.5
    assert rows[0]["interpretation"] == "Higher volatility argues for tighter sizing and tighter stops."
    assert any(row["label"] == "US 10Y" and "Higher yields" in row["interpretation"] for row in rows)
    assert any(row["label"] == "DXY" and "stronger dollar" in row["interpretation"] for row in rows)


def main() -> None:
    test_high_frequency_trackers_sort_and_interpret()
    print("Analytics trackers test passed")


if __name__ == "__main__":
    main()
