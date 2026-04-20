import os
import shutil

from _bootstrap import ROOT  # noqa: F401

from professional import trend_pack
from professional.trend_pack import collect_hk_trend_pack_data, generate_hk_trend_pack


def main() -> None:
    bundle = {"meta": {"report_date": "2026-04-13", "briefing_date": "2026-04-14"}}
    trend_data = {
        "southbound": [
            {"date": "2026-04-08", "net_buy_hkd_bn": 1.2, "cumulative_hkd_bn": 1.2},
            {"date": "2026-04-09", "net_buy_hkd_bn": -0.6, "cumulative_hkd_bn": 0.6},
            {"date": "2026-04-10", "net_buy_hkd_bn": 2.0, "cumulative_hkd_bn": 2.6},
            {"date": "2026-04-13", "net_buy_hkd_bn": 1.4, "cumulative_hkd_bn": 4.0},
        ],
        "liquidity": [
            {"date": "2026-04-08", "hibor_1m": 2.05, "aggregate_balance_bn": 55.1},
            {"date": "2026-04-09", "hibor_1m": 2.09, "aggregate_balance_bn": 54.8},
            {"date": "2026-04-10", "hibor_1m": 2.16, "aggregate_balance_bn": 54.5},
            {"date": "2026-04-13", "hibor_1m": 2.23, "aggregate_balance_bn": 54.4},
        ],
        "leadership": {
            "dates": ["2026-04-08", "2026-04-09", "2026-04-10", "2026-04-13"],
            "series": {
                "HSI": [100.0, 100.6, 101.1, 101.4],
                "HSCEI": [100.0, 100.4, 100.8, 100.9],
                "HSTECH": [100.0, 101.0, 101.9, 102.2],
            },
        },
        "ah_heatmap": {
            "dates": ["2026-04-08", "2026-04-09", "2026-04-10", "2026-04-13"],
            "names": ["CRRC", "China Railway"],
            "matrix": [[80.0, 81.0, 82.0, 82.5], [62.5, 63.0, 63.8, 64.2]],
        },
    }

    output_path = os.path.join("reports_professional", "charts", "test_hk_trend_pack_unit.png")
    meta = generate_hk_trend_pack(bundle, output_path, trend_data=trend_data)

    assert os.path.exists(output_path)
    assert meta["title"] == "Hong Kong Trend Pack"
    assert meta["path"] == "test_hk_trend_pack_unit.png"
    assert meta["rel_path"] == "charts/test_hk_trend_pack_unit.png"

    cache_path = ROOT / "tmp_trend_pack_cache"
    if cache_path.exists():
        shutil.rmtree(cache_path, ignore_errors=True)
    cache_path.mkdir()
    cache_dir = str(cache_path)
    cached_bundle = {"meta": {"report_date": "2026-04-13", "hk_data_date": "2026-04-13"}}
    calls = {"southbound": 0, "liquidity": 0, "leadership": 0, "ah_heatmap": 0}

    original_southbound = trend_pack._collect_southbound_history
    original_liquidity = trend_pack._collect_hkma_history
    original_leadership = trend_pack._collect_leadership_history
    original_ah = trend_pack._collect_ah_heatmap_history

    try:
        def fake_southbound(report_date: str, sessions: int = 20, cache_dir: str | None = None):
            calls["southbound"] += 1
            return trend_data["southbound"]

        def fake_liquidity(report_date: str, sessions: int = 30, cache_dir: str | None = None):
            calls["liquidity"] += 1
            return trend_data["liquidity"]

        def fake_leadership(report_date: str, sessions: int = 30, cache_dir: str | None = None):
            calls["leadership"] += 1
            return trend_data["leadership"]

        def fake_ah(bundle_arg, report_date: str, row_limit: int = 8, sessions: int = 5, cache_dir: str | None = None):
            calls["ah_heatmap"] += 1
            return trend_data["ah_heatmap"]

        trend_pack._collect_southbound_history = fake_southbound
        trend_pack._collect_hkma_history = fake_liquidity
        trend_pack._collect_leadership_history = fake_leadership
        trend_pack._collect_ah_heatmap_history = fake_ah

        first = collect_hk_trend_pack_data(cached_bundle, sessions=4, cache_dir=cache_dir)
        second = collect_hk_trend_pack_data(cached_bundle, sessions=4, cache_dir=cache_dir)

        assert first == second
        assert calls == {"southbound": 1, "liquidity": 1, "leadership": 1, "ah_heatmap": 1}
    finally:
        trend_pack._collect_southbound_history = original_southbound
        trend_pack._collect_hkma_history = original_liquidity
        trend_pack._collect_leadership_history = original_leadership
        trend_pack._collect_ah_heatmap_history = original_ah
        shutil.rmtree(cache_path, ignore_errors=True)

    print("Trend Pack test passed")


if __name__ == "__main__":
    main()
