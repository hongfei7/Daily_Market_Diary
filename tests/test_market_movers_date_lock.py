from __future__ import annotations

import pandas as pd

from _bootstrap import ROOT  # noqa: F401
from modules import market_movers


class _FakeTicker:
    calls = []

    def __init__(self, ticker: str) -> None:
        self.ticker = ticker

    def history(self, **kwargs):
        self.calls.append(kwargs)
        return pd.DataFrame(
            {"Close": [100.0, 102.0, 50.0], "Volume": [1000, 1200, 9000]},
            index=pd.to_datetime(["2026-08-21", "2026-08-24", "2026-08-25"]),
        )


def test_etf_activity_is_locked_to_requested_date(monkeypatch) -> None:
    monkeypatch.setattr(market_movers.yf, "Ticker", _FakeTicker)
    monkeypatch.setattr(market_movers.MarketMoversAnalyzer, "MAJOR_ETFS", {"TEST": "Test ETF"})

    rows = market_movers.MarketMoversAnalyzer().fetch_etf_flows("2026-08-24")

    assert rows[0]["as_of"] == "2026-08-24"
    assert rows[0]["price"] == 102.0
    assert rows[0]["change_pct"] == 2.0
    assert _FakeTicker.calls[-1]["end"] == "2026-08-25"
