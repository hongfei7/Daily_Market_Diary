from __future__ import annotations

from market_diary.professional.instruments import annotate_summary_item, format_summary_change


def test_hstech_proxy_identity_is_explicit() -> None:
    item = annotate_summary_item(
        "Equities",
        "Hang Seng TECH ETF",
        "3033.HK",
        {"Price": 4.73, "Change": 0.02, "Pct Change": "0.25%", "As Of": "2026-07-31", "Quality": "fresh"},
        "2026-08-03",
    )
    assert item["Display Name"] == "Hang Seng TECH ETF (3033.HK)"
    assert item["Security Type"] == "etf"
    assert item["Price Unit"] == "HKD_per_share"
    assert item["Trading Freshness Days"] == 1
    assert item["Quality"] == "fresh"


def test_treasury_change_is_basis_points_not_price_return() -> None:
    item = annotate_summary_item(
        "Rates",
        "10Y Treasury",
        "^TNX",
        {"Price": 4.745, "Change": 0.082, "Pct Change": "1.76%", "As Of": "2026-07-31", "Quality": "fresh"},
        "2026-08-03",
    )
    assert item["Change Unit"] == "bp"
    assert item["Change Value"] == 8.2
    assert format_summary_change(item) == "+8.2bp"
