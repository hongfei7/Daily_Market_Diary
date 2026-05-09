from __future__ import annotations

from types import SimpleNamespace

from _bootstrap import ROOT  # noqa: F401

import main_professional


def _args(output_dir: str, **overrides) -> SimpleNamespace:
    values = {
        "date": "",
        "review_date": "",
        "global_date": "",
        "hk_date": "",
        "briefing_date": "",
        "output_dir": output_dir,
        "config": "",
        "skip_charts": False,
        "skip_dashboard": False,
        "skip_daily_chart": False,
        "skip_trend_pack": False,
        "no_llm": True,
        "debug": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _install_main_stubs(monkeypatch, tmp_path, *, args: SimpleNamespace, mode: str) -> dict:
    calls = {"dashboard": 0, "daily": 0, "trend_collect": 0, "trend_generate": 0, "appendix": 0}

    monkeypatch.setattr(main_professional, "parse_args", lambda: args)
    monkeypatch.setattr(
        main_professional,
        "load_professional_config",
        lambda path=None: {"system": {"timezone": "Asia/Shanghai"}},
    )
    monkeypatch.setattr(
        main_professional,
        "resolve_report_dates",
        lambda parsed_args, config: {
            "briefing_date": "2026-05-08",
            "review_date": "2026-05-07",
            "global_market_date": "2026-05-07",
            "hk_data_date": "2026-05-07",
        },
    )
    monkeypatch.setattr(main_professional, "_configure_market_data_cache", lambda output_dir: None)
    monkeypatch.setattr(
        main_professional,
        "fetch_all_data",
        lambda **kwargs: {
            "market": {"timeseries": []},
            "macro": {},
            "sector": {},
            "movers": {},
            "stock_connect": {},
            "ah_premium": {},
            "risk": {},
            "news": [],
            "hk_local": {},
            "china_rates": {},
        },
    )
    monkeypatch.setattr(main_professional, "extract_chart_features", lambda timeseries, tz="Asia/Shanghai": {})
    monkeypatch.setattr(main_professional, "_save_chart_features", lambda *args, **kwargs: None)
    monkeypatch.setattr(main_professional, "_save_bundle", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        main_professional,
        "build_professional_bundle",
        lambda **kwargs: {
            "day_mode": {"mode": mode},
            "meta": {
                "briefing_date": kwargs.get("briefing_date", "2026-05-08"),
                "report_date": kwargs.get("report_date", "2026-05-07"),
                "hk_data_date": kwargs.get("hk_data_date", "2026-05-07"),
            },
        },
    )
    monkeypatch.setattr(main_professional, "run_fact_check", lambda bundle: {"status": "ok"})
    monkeypatch.setattr(main_professional, "build_report_quality", lambda bundle: {"status": "ok"})
    monkeypatch.setattr(main_professional, "render_professional_report", lambda **kwargs: "# report")

    def fake_dashboard(bundle, output_path):
        calls["dashboard"] += 1
        return "dashboard.png"

    def fake_daily_chart(bundle, output_path):
        calls["daily"] += 1
        return {"path": "daily_one_chart.png", "title": "Daily One Chart"}

    def fake_collect(bundle, cache_dir=None):
        calls["trend_collect"] += 1
        return {"southbound": [], "liquidity": [], "leadership": {"dates": [], "series": {}}, "ah_heatmap": {"dates": [], "names": [], "matrix": []}}

    def fake_trend_pack(bundle, output_path, trend_data=None, cache_dir=None):
        calls["trend_generate"] += 1
        return {"path": "hk_trend_pack.png", "title": "Hong Kong Trend Pack", "weekly_summary": {"status": "ok", "rows": []}}

    def fake_appendix(**kwargs):
        calls["appendix"] += 1
        return "_appendix_"

    monkeypatch.setattr(main_professional, "generate_dashboard", fake_dashboard)
    monkeypatch.setattr(main_professional, "generate_daily_one_chart", fake_daily_chart)
    monkeypatch.setattr(main_professional, "collect_hk_trend_pack_data", fake_collect)
    monkeypatch.setattr(main_professional, "generate_hk_trend_pack", fake_trend_pack)
    monkeypatch.setattr(main_professional, "render_chart_appendix", fake_appendix)

    return calls


def test_skip_charts_skips_all_visual_generation(monkeypatch, tmp_path) -> None:
    args = _args(str(tmp_path), skip_charts=True)
    calls = _install_main_stubs(monkeypatch, tmp_path, args=args, mode="weekly_review")

    main_professional.main()

    assert calls == {"dashboard": 0, "daily": 0, "trend_collect": 0, "trend_generate": 0, "appendix": 0}


def test_non_weekly_runs_do_not_generate_trend_pack(monkeypatch, tmp_path) -> None:
    args = _args(str(tmp_path))
    calls = _install_main_stubs(monkeypatch, tmp_path, args=args, mode="trading_daily")

    main_professional.main()

    assert calls["dashboard"] == 1
    assert calls["daily"] == 1
    assert calls["trend_collect"] == 0
    assert calls["trend_generate"] == 0
    assert calls["appendix"] == 1
