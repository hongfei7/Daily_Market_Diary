import os
import sys

import pandas as pd

from _bootstrap import ROOT  # noqa: F401

from modules import data_fetcher


def _price_frame(index_values, close_values):
    return pd.DataFrame({"Close": close_values}, index=pd.to_datetime(index_values))


def test_daily_history_fallback() -> None:
    original_download = data_fetcher._request_download
    original_history = data_fetcher._request_history

    try:
        data_fetcher._request_download = lambda *args, **kwargs: pd.DataFrame()

        def fake_history(symbol, start=None, end=None, interval="1d", period=None):
            if interval == "1d":
                return _price_frame(["2026-04-10", "2026-04-13"], [100.0, 103.0])
            return pd.DataFrame()

        data_fetcher._request_history = fake_history

        summary = data_fetcher._calc_summary_for_symbol("TEST", "2026-04-13")
        assert summary is not None
        assert summary["Price"] == 103.0
        assert summary["Pct Change"] == "3.00%"
        assert summary["Quality"] == "fresh"
        assert summary["Basis"] == "daily_close"
        assert summary["Source"].startswith("history_window")
    finally:
        data_fetcher._request_download = original_download
        data_fetcher._request_history = original_history


def test_intraday_cache_fallback() -> None:
    original_download = data_fetcher._request_download
    original_history = data_fetcher._request_history

    try:
        data_fetcher._request_download = lambda *args, **kwargs: pd.DataFrame()
        data_fetcher._request_history = lambda *args, **kwargs: pd.DataFrame()

        intraday_cache = pd.DataFrame(
            {
                "time": pd.to_datetime(["2026-04-13 09:30", "2026-04-13 16:00"]),
                "price": [100.0, 102.0],
            }
        )

        summary = data_fetcher._calc_summary_for_symbol(
            "TEST",
            "2026-04-13",
            intraday_cache=intraday_cache,
        )
        assert summary is not None
        assert summary["Price"] == 102.0
        assert summary["Pct Change"] == "2.00%"
        assert summary["Quality"] == "intraday_fallback"
        assert summary["Basis"] == "intraday_session"
        assert summary["Source"] == "intraday_cache"
    finally:
        data_fetcher._request_download = original_download
        data_fetcher._request_history = original_history


def test_quality_rollup() -> None:
    quality = data_fetcher._build_summary_quality(
        {
            "Equities": {
                "S&P 500": {
                    "Price": 5000,
                    "Pct Change": "1.00%",
                    "Freshness Days": 0,
                    "Quality": "fresh",
                    "Source": "download_window",
                },
                "FXI": {
                    "Price": 30,
                    "Pct Change": "0.50%",
                    "Freshness Days": 2,
                    "Quality": "intraday_fallback",
                    "Source": "intraday_cache",
                },
                "HSI": "No Data",
            }
        }
    )

    assert quality["available"] == 2
    assert quality["total"] == 3
    assert quality["missing"] == ["Equities / HSI"]
    assert quality["stale"] == ["Equities / FXI"]
    assert quality["fallback"] == ["Equities / FXI"]


def main() -> None:
    test_daily_history_fallback()
    test_intraday_cache_fallback()
    test_quality_rollup()
    print("Market data resilience test passed")


if __name__ == "__main__":
    main()
