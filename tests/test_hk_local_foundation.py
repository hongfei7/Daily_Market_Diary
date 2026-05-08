import os
import sys
from datetime import date, timedelta

from _bootstrap import ROOT  # noqa: F401

from modules import china_rates as china_rates_module
from modules import hk_local_data as hk_local_module


def _turnover_history(anchor: date, sessions: int = 21):
    history = []
    current = anchor
    value = 200_000_000_000.0
    while len(history) < sessions:
        if current.weekday() < 5:
            history.append(
                {
                    "date": current,
                    "turnover_hkd": value,
                    "source": "HKEX Daily Quotations",
                    "source_url": "https://www.hkex.com.hk/",
                }
            )
            value -= 2_500_000_000.0
        current -= timedelta(days=1)
    return history


def main() -> None:
    original_turnover = hk_local_module._collect_turnover_history
    original_hkma = hk_local_module._fetch_hkma_record
    original_short = hk_local_module._fetch_short_sell_snapshot
    original_rows = china_rates_module._fetch_rows

    try:
        hk_local_module._collect_turnover_history = lambda target: _turnover_history(date(2026, 4, 13))
        hk_local_module._fetch_hkma_record = lambda target: {
            "end_of_date": "2026-04-13",
            "hibor_fixing_1m": 2.23244,
            "closing_balance": 54440,
            "disc_win_base_rate": 4.0,
            "cu_weakside": 7.85,
            "cu_strongside": 7.75,
        }
        hk_local_module._fetch_short_sell_snapshot = lambda target, turnover_map: {
            "value": 11.5,
            "display_value": "11.50%",
            "status": "live_local",
            "source": "HKEX Short Selling Turnover Report",
            "as_of": "2026-04-13",
            "freshness_days": 1,
            "quality": "fresh",
            "fallback_used": False,
            "note": "Short-selling turnover as a share of Main Board turnover.",
            "change_value": None,
            "change_display": "",
        }

        stock_connect_payload = {
            "status": "ok",
            "data": {
                "southbound": {"net_buy": 4000.25, "total_turnover": 20000.5, "net_buy_available": True},
                "northbound": {"net_buy": None, "total_turnover": 15000.0, "net_buy_available": False},
            },
            "meta": {"source": "HKEX Stock Connect Historical Daily", "effective_date": "2026-04-13"},
        }
        ah_premium_payload = {
            "status": "ok",
            "data": {"average_premium": 32.4, "rows": [{"name": "CRRC", "premium_pct": 82.5}]},
            "meta": {"source": "Public Yahoo Finance quotes - calculated A/H premium", "effective_date": "2026-04-13"},
        }

        hk_local_payload = hk_local_module.fetch_hk_local_data(
            "2026-04-14",
            stock_connect_data=stock_connect_payload,
            ah_premium_data=ah_premium_payload,
        )
        assert hk_local_payload["status"] == "ok"
        assert hk_local_payload["data"]["main_board_turnover"]["status"] == "live_local"
        assert hk_local_payload["data"]["turnover_vs_20d"]["status"] == "live_local"
        assert hk_local_payload["data"]["hibor_1m"]["status"] == "live_local"
        assert hk_local_payload["data"]["southbound_net_flow"]["status"] == "live_local"
        assert hk_local_payload["data"]["northbound_net_flow"]["display_value"].endswith("net not reported")
        assert hk_local_payload["data"]["ah_premium_index"]["status"] == "live_public"
        assert hk_local_payload["meta"]["available_metrics"] >= 4

        hk_local_module._fetch_hkma_record = lambda target: None
        hk_local_module._fetch_short_sell_snapshot = lambda target, turnover_map: {
            "value": None,
            "display_value": "N/A",
            "status": "unavailable",
            "source": "HKEX Short Selling Turnover Report",
            "as_of": "",
            "freshness_days": None,
            "quality": "unavailable",
            "fallback_used": False,
            "note": "Unavailable in this test case.",
            "change_value": None,
            "change_display": "",
        }
        partial_payload = hk_local_module.fetch_hk_local_data("2026-04-14")
        assert partial_payload["status"] == "ok"
        assert partial_payload["data"]["main_board_turnover"]["status"] == "live_local"
        assert partial_payload["data"]["hibor_1m"]["status"] == "unavailable"

        china_rates_module._fetch_rows = lambda: [
            {
                "SOLAR_DATE": "2026-04-13 00:00:00",
                "EMM00166466": 1.7933,
                "EMG00001310": 4.30,
            },
            {
                "SOLAR_DATE": "2026-04-10 00:00:00",
                "EMM00166466": 1.8010,
                "EMG00001310": 4.34,
            },
        ]
        china_payload = china_rates_module.fetch_china_rates_data("2026-04-13")
        assert china_payload["status"] == "ok"
        assert china_payload["data"]["china_10y"]["status"] == "live_public"
        assert china_payload["data"]["cn_us_10y_spread"]["display_value"].endswith("bp")

        china_rates_module._fetch_rows = lambda: [
            {
                "SOLAR_DATE": "2026-04-15 00:00:00",
                "EMM00166466": 1.85,
                "EMG00001310": 4.40,
            }
        ]
        unavailable_payload = china_rates_module.fetch_china_rates_data("2026-04-13")
        assert unavailable_payload["data"]["china_10y"]["status"] == "unavailable"
        assert unavailable_payload["data"]["cn_us_10y_spread"]["status"] == "unavailable"
    finally:
        hk_local_module._collect_turnover_history = original_turnover
        hk_local_module._fetch_hkma_record = original_hkma
        hk_local_module._fetch_short_sell_snapshot = original_short
        china_rates_module._fetch_rows = original_rows

    print("Hong Kong local foundation test passed")


if __name__ == "__main__":
    main()
