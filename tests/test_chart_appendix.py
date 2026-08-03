from _bootstrap import ROOT  # noqa: F401

from market_diary.professional.chart_appendix import render_chart_appendix


def main() -> None:
    bundle = {
        "overview": {
            "chart_read": {
                "fx": ["USD composite moved higher intraday."],
                "assets": ["Gold outperformed oil on the day."],
            }
        },
        "attribution": {
            "risk_dashboard": {
                "components": [
                    {"label": "VIX", "delta": -6, "evidence": "Volatility eased versus the prior session."},
                    {"label": "Turnover", "delta": +4, "evidence": "Main board turnover recovered against its 20-day average."},
                ]
            }
        },
    }

    appendix = render_chart_appendix(
        bundle=bundle,
        dashboard_rel_path="charts/dashboard_test.png",
        catalyst_radar_rel_path="charts/catalyst_radar_test.png",
        daily_chart_rel_path="charts/daily_one_chart_test.png",
        trend_pack_rel_path="charts/hk_trend_pack_test.png",
    )

    assert "legacy intraday chart pack" in appendix
    assert "Visual Dashboard" in appendix
    assert "Catalyst & Event Radar" in appendix
    assert "Hong Kong Trend Pack" in appendix
    assert "## \U0001f4ca Charts" not in appendix
    assert "charts/dashboard_test.png" in appendix
    assert "charts/catalyst_radar_test.png" in appendix
    assert "USD composite moved higher intraday." in appendix
    print("Chart appendix test passed")


if __name__ == "__main__":
    main()
