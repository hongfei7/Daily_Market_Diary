from _bootstrap import ROOT  # noqa: F401

from professional.analytics_public_flow import enrich_hk_local_with_public_flow


def test_public_flow_enrichment_fills_only_unavailable_metrics() -> None:
    hk_local_metrics = {
        "main_board_turnover": {"status": "live_local", "display_value": "HK$200.0bn"},
        "southbound_net_flow": {"status": "unavailable", "display_value": "N/A"},
        "northbound_net_flow": {"status": "unavailable", "display_value": "N/A"},
        "ah_premium_index": {"status": "unavailable", "display_value": "N/A"},
    }
    stock_connect_data = {
        "status": "ok",
        "data": {
            "southbound": {"net_buy": 4000.25, "total_turnover": 20000.5},
            "northbound": {"net_buy": None, "total_turnover": 15000.0},
        },
        "meta": {"source": "HKEX Stock Connect Historical Daily", "effective_date": "2026-04-13"},
    }
    ah_premium_data = {
        "status": "ok",
        "data": {"average_premium": 32.4, "rows": [{"name": "CRRC", "premium_pct": 82.5}]},
        "meta": {"source": "Public Yahoo Finance quotes - calculated A/H premium", "effective_date": "2026-04-13"},
    }

    enriched = enrich_hk_local_with_public_flow(
        "2026-04-14",
        hk_local_metrics,
        stock_connect_data,
        ah_premium_data,
    )

    assert enriched["main_board_turnover"]["display_value"] == "HK$200.0bn"
    assert enriched["southbound_net_flow"]["status"] == "live_public"
    assert "Net HK$4.0bn" in enriched["southbound_net_flow"]["display_value"]
    assert "turnover HK$20.0bn" in enriched["southbound_net_flow"]["display_value"]
    assert enriched["northbound_net_flow"]["status"] == "partial_public"
    assert "Turnover RMB15.0bn | net unavailable" in enriched["northbound_net_flow"]["display_value"]
    assert enriched["ah_premium_index"]["display_value"] == "32.40%"


def test_public_flow_enrichment_preserves_existing_live_metrics() -> None:
    hk_local_metrics = {
        "southbound_net_flow": {
            "status": "live_local",
            "display_value": "Keep this local value",
        }
    }
    stock_connect_data = {
        "status": "ok",
        "data": {"southbound": {"net_buy": 4000.25, "total_turnover": 20000.5}},
        "meta": {},
    }

    enriched = enrich_hk_local_with_public_flow(
        "2026-04-14",
        hk_local_metrics,
        stock_connect_data,
        None,
    )

    assert enriched["southbound_net_flow"]["display_value"] == "Keep this local value"


if __name__ == "__main__":
    test_public_flow_enrichment_fills_only_unavailable_metrics()
    test_public_flow_enrichment_preserves_existing_live_metrics()
    print("Analytics public flow test passed")
