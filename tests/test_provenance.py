from _bootstrap import ROOT  # noqa: F401

from modules.macro_calendar import fetch_macro_data
from modules.market_movers import MarketMoversAnalyzer
from modules.provenance import audit_source_provenance, provenance_record
from modules.risk_radar import fetch_risk_data
from modules.sector_news import SectorNewsAggregator


def test_placeholder_adapters_do_not_emit_financial_claims():
    sector = SectorNewsAggregator()
    assert sector.fetch_earnings_calendar("2026-08-03") == []
    assert sector.fetch_analyst_changes("2026-08-03") == []

    movers = MarketMoversAnalyzer()
    assert movers.fetch_premarket_movers() == {"gainers": [], "losers": [], "most_active": []}
    assert movers.fetch_block_trades_cn("2026-08-03") == []
    assert movers.fetch_unusual_options("2026-08-03") == []

    macro = fetch_macro_data("2026-08-03")
    assert macro["status"] == "unavailable"
    assert macro["calendar"]["released"] == []
    assert macro["calendar"]["upcoming"] == []
    assert macro["central_bank_events"] == []

    risk = fetch_risk_data({"HSI": 25000})
    assert risk["status"] == "unavailable"
    assert risk["geopolitical_risks"] == []
    assert risk["upcoming_events"] == []
    assert risk["sentiment_indicators"] == {}
    assert risk["technical_levels"] == {}


def test_provenance_schema_accepts_verified_and_unavailable_sources():
    payloads = {
        "verified": {
            "provenance": [
                provenance_record(
                    source_name="HKEX",
                    source_url="https://www.hkex.com.hk/",
                    as_of="2026-08-03",
                    source_type="official",
                    status="ok",
                    confidence=0.95,
                )
            ]
        },
        "unavailable": {
            "provenance": [
                provenance_record(
                    source_name="Macro calendar",
                    source_url="",
                    as_of="2026-08-03",
                    source_type="unavailable",
                    status="unavailable",
                    confidence=0.0,
                )
            ]
        },
    }
    audit = audit_source_provenance(payloads)
    assert audit["status"] == "ok"
    assert audit["checked_records"] == 2
    assert audit["unavailable_records"] == 1


def test_provenance_schema_rejects_synthetic_and_missing_sources():
    payloads = {
        "synthetic": {
            "provenance": [
                {
                    "source_name": "Example feed",
                    "source_url": "",
                    "as_of": "2026-08-03",
                    "collected_at": "2026-08-03T00:00:00+00:00",
                    "source_type": "public",
                    "status": "synthetic",
                    "confidence": 1.0,
                }
            ]
        },
        "missing": {},
    }
    audit = audit_source_provenance(payloads)
    assert audit["status"] == "error"
    assert any("non-production" in item for item in audit["errors"])
    assert any("missing provenance" in item for item in audit["errors"])


def main() -> None:
    test_placeholder_adapters_do_not_emit_financial_claims()
    test_provenance_schema_accepts_verified_and_unavailable_sources()
    test_provenance_schema_rejects_synthetic_and_missing_sources()
    print("Provenance test passed")


if __name__ == "__main__":
    main()
