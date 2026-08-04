from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from modules import adapter_hkex_announce, adapter_shortsell, adapter_stockconnect


class _FakeResponse:
    def __init__(self, text: str, status_code: int = 200) -> None:
        self.text = text
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_stockconnect_reuses_single_session_across_lookback(monkeypatch) -> None:
    session_creations = {"count": 0}

    class FakeSession:
        def get(self, url: str, timeout: float):
            if "20260413" in url:
                raise RuntimeError("weekend")
            return _FakeResponse(
                """
                tabData = [
                  {
                    "market": "SSE Southbound",
                    "date": "2026-04-12",
                    "tradingDay": true,
                    "content": [
                      {"table": {"schema": [["Total Turnover", "Buy Turnover", "Sell Turnover"]], "tr": [
                        {"td": [["10,000.00"]]},
                        {"td": [["6,000.00"]]},
                        {"td": [["4,000.00"]]}
                      ]}},
                      {"table": {"schema": [["Rank", "Stock Code", "Stock Name", "Buy Turnover", "Sell Turnover", "Total Turnover"]], "tr": []}}
                    ]
                  }
                ];
                """
            )

    def fake_session():
        session_creations["count"] += 1
        return FakeSession()

    monkeypatch.setattr(adapter_stockconnect, "_session", fake_session)

    payload = adapter_stockconnect.fetch_stock_connect_data("2026-04-13", lookback_days=2)

    assert payload["status"] == "ok"
    assert payload["meta"]["effective_date"] == "2026-04-12"
    assert session_creations["count"] == 1


def test_shortsell_reuses_single_session_across_lookback(monkeypatch) -> None:
    session_creations = {"count": 0}

    class FakeSession:
        def get(self, url: str, timeout: float):
            if "d260413e" in url:
                raise RuntimeError("holiday")
            return _FakeResponse(
                """
                DATE: 12 APR 2026
                <a name = "short_selling">SHORT SELLING TURNOVER - DAILY REPORT</a>
                0700 TENCENT 100,000 20,000,000 200,000 100,000,000
                Short Selling of all Designated Securities as % total turnover : 20.00%
                (C) Short Selling of all Designated Securities
                Short Selling Turnover Total Value ($) : HKD 20,000,000
                Total market turnover : HKD 100,000,000
                <a name = "adj_short">
                """
            )

    def fake_session():
        session_creations["count"] += 1
        return FakeSession()

    monkeypatch.setattr(adapter_shortsell, "_session", fake_session)

    payload = adapter_shortsell.fetch_short_sell_data("2026-04-13", lookback_days=2)

    assert payload["status"] == "ok"
    assert payload["meta"]["effective_date"] == "2026-04-12"
    assert payload["data"]["market"]["short_ratio_pct"] == 20.0
    assert session_creations["count"] == 1


def test_hkex_announcements_reuses_single_session_for_all_sources(monkeypatch) -> None:
    session_creations = {"count": 0}

    predefined_html = """
    <table>
      <tbody>
        <tr>
          <td class="release-time">13/04/2026 18:30</td>
          <td class="stock-short-code">0700</td>
          <td class="stock-short-name">Tencent</td>
          <td class="headline">Annual results announcement</td>
          <td class="doc-link"><a href="/docs/test.pdf">Annual results announcement</a></td>
        </tr>
      </tbody>
    </table>
    """
    profit_warning_html = """
    <table>
      <tr>
        <td>0700</td>
        <td>Tencent</td>
        <td>13 April 2026</td>
        <td><a href="/reports/test.pdf">warning</a></td>
      </tr>
    </table>
    """

    class FakeSession:
        def get(self, url: str, timeout: float):
            if "profitwarning" in url:
                return _FakeResponse(profit_warning_html)
            return _FakeResponse(predefined_html)

    def fake_session():
        session_creations["count"] += 1
        return FakeSession()

    monkeypatch.setattr(adapter_hkex_announce, "_session", fake_session)

    payload = adapter_hkex_announce.fetch_hkex_announcements("2026-04-13")

    assert payload["status"] in {"ok", "partial"}
    assert payload["meta"]["available_count"] >= 1
    assert session_creations["count"] == 1


def test_profit_warning_priority_requires_portfolio_relevance() -> None:
    item = {
        "code": "0229",
        "company": "Raymond Industrial",
        "title": "Profit warning / alert announcement",
        "document": "Announcements Concerning Profit Warning",
        "event_type": "Profit warning",
    }

    market_score = adapter_hkex_announce._importance_score(item, {"codes": set(), "names": set()})
    portfolio_score = adapter_hkex_announce._importance_score(item, {"codes": {"0229"}, "names": set()})

    assert adapter_hkex_announce._grade(market_score) == "C"
    assert adapter_hkex_announce._grade(portfolio_score) == "A"


def test_profit_warning_filing_extract_keeps_primary_facts() -> None:
    text = """
    The Group is expected to record a loss in the range of HK$10 million to HK$11 million
    for the six months ended 30 June 2026 as compared to a net profit of HK$32.5 million in 2025.
    The expected loss was mainly attributable to exchange losses and higher commodity prices.
    The interim results announcement is expected to be published on 21 August 2026.
    """

    details = adapter_hkex_announce._extract_filing_details(text)

    assert details["filing_detail_status"] == "parsed"
    assert "HK$10 million to HK$11 million" in details["filing_extract"]
    assert "exchange losses" in details["filing_drivers"]
    assert "21 August 2026" in details["next_disclosure"]
