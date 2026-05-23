import os
import sys
import tempfile

from _bootstrap import ROOT  # noqa: F401

from professional.daily_one_chart import generate_daily_one_chart


def main() -> None:
    bundle = {
        "meta": {"briefing_date": "2026-04-14"},
        "market_summary": {
            "Equities": {
                "Hang Seng TECH ETF": {"Pct Change": "-1.2%"},
                "China Large-Cap (FXI)": {"Pct Change": "0.5%"},
            },
            "FX": {"DXY": {"Pct Change": "0.1%"}, "USD/CNH": {"Pct Change": "0.1%"}},
            "Commodities": {"Brent Crude": {"Pct Change": "0.4%"}, "Crude Oil": {"Pct Change": "0.5%"}},
        },
        "hk_local": {
            "short_selling_ratio": {"value": 18.5},
            "turnover_vs_20d": {"value": 1.05},
        },
        "flow_tracker": {
            "short_sell_top_ratio": [
                {
                    "ticker": "00388.HK",
                    "code": "00388",
                    "name": "HKEX",
                    "short_ratio_pct": 26.2,
                    "short_turnover_hkd": 300_000_000,
                    "total_turnover_hkd": 1_300_000_000,
                }
            ]
        },
        "attribution": {
            "dominant_drivers": [],
            "risk_dashboard": {"score": 52.0, "bucket": "Mixed", "components": []},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        chart_dir = os.path.join(tmpdir, "charts")
        output_path = os.path.join(chart_dir, "test_daily_one_chart_unit.png")
        meta = generate_daily_one_chart(bundle, output_path)

        assert os.path.exists(output_path)
        assert meta["kind"] == "short_selling"
        assert meta["path"] == "test_daily_one_chart_unit.png"

        stock_connect_bundle = {
            "meta": {"briefing_date": "2026-04-14"},
            "market_summary": {"Equities": {}, "FX": {}, "Commodities": {}},
            "hk_local": {"short_selling_ratio": {"value": 10.0}, "turnover_vs_20d": {"value": 1.0}},
            "flow_tracker": {
                "stock_connect": {
                    "data": {
                        "southbound": {
                            "top_active": [
                                {"ticker": "03690.HK", "name": "MEITUAN-W", "net_buy": 500.0, "total_turnover": 1300.0},
                                {"ticker": "00700.HK", "name": "TENCENT", "net_buy": 400.0, "total_turnover": 2000.0},
                            ]
                        }
                    }
                }
            },
            "attribution": {"dominant_drivers": [], "risk_dashboard": {"score": 52.0, "bucket": "Mixed", "components": []}},
        }
        stock_connect_path = os.path.join(chart_dir, "test_daily_one_chart_stockconnect.png")
        stock_connect_meta = generate_daily_one_chart(stock_connect_bundle, stock_connect_path)
        assert os.path.exists(stock_connect_path)
        assert stock_connect_meta["kind"] == "stock_connect"

        moderate_short_bundle = {
            "meta": {"briefing_date": "2026-04-14"},
            "market_summary": {"Equities": {}, "FX": {}, "Commodities": {}},
            "hk_local": {"short_selling_ratio": {"value": 14.5}, "turnover_vs_20d": {"value": 1.0}},
            "flow_tracker": {
                "short_sell_top_ratio": [
                    {
                        "ticker": "09988.HK",
                        "code": "09988",
                        "name": "BABA-W",
                        "short_ratio_pct": 18.8,
                        "short_turnover_hkd": 500_000_000,
                    }
                ],
                "stock_connect": {
                    "data": {
                        "southbound": {
                            "top_active": [
                                {"ticker": "00700.HK", "name": "TENCENT", "net_buy": 700.0, "total_turnover": 2300.0}
                            ]
                        }
                    }
                },
            },
            "attribution": {"dominant_drivers": [], "risk_dashboard": {"score": 52.0, "bucket": "Mixed", "components": []}},
        }
        moderate_short_path = os.path.join(chart_dir, "test_daily_one_chart_moderate_short.png")
        moderate_short_meta = generate_daily_one_chart(moderate_short_bundle, moderate_short_path)
        assert os.path.exists(moderate_short_path)
        assert moderate_short_meta["kind"] == "stock_connect"

        ah_bundle = {
            "meta": {"briefing_date": "2026-04-14"},
            "market_summary": {"Equities": {}, "FX": {}, "Commodities": {}},
            "hk_local": {"short_selling_ratio": {"value": 10.0}, "turnover_vs_20d": {"value": 1.0}},
            "flow_tracker": {
                "ah_premium": {
                    "data": {
                        "top_premium": [
                            {"name": "CRRC", "premium_pct": 82.5},
                            {"name": "China Railway", "premium_pct": 64.2},
                        ]
                    }
                }
            },
            "attribution": {"dominant_drivers": [], "risk_dashboard": {"score": 52.0, "bucket": "Mixed", "components": []}},
        }
        ah_path = os.path.join(chart_dir, "test_daily_one_chart_ah.png")
        ah_meta = generate_daily_one_chart(ah_bundle, ah_path)
        assert os.path.exists(ah_path)
        assert ah_meta["kind"] == "ah_premium"
    print("Daily One Chart test passed")


if __name__ == "__main__":
    main()
