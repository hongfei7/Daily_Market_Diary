"""
Professional morning briefing entrypoint.

The upgraded pipeline is designed around four layers:
1. Data collection
2. Deterministic research analytics
3. Visual dashboard generation
4. Markdown report rendering
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import threading
from typing import Any, Dict

from modules.adapter_ah_premium import fetch_ah_premium_data
from modules.adapter_stockconnect import fetch_stock_connect_data
from modules.chart_features import extract_chart_features
from modules.china_rates import fetch_china_rates_data
from modules.data_fetcher import fetch_market_data, fetch_news
from modules.hk_local_data import fetch_hk_local_data
from modules.macro_calendar import fetch_macro_data
from modules.market_movers import fetch_movers_data
from modules.risk_radar import fetch_risk_data
from modules.sector_news import fetch_sector_data
from professional.analytics import build_professional_bundle
from professional.chart_appendix import render_chart_appendix
from professional.config import load_professional_config
from professional.daily_one_chart import generate_daily_one_chart
from professional.dashboard import generate_dashboard
from professional.date_policy import (
    build_day_mode,
    previous_calendar_day as _previous_calendar_day,
    previous_hk_trading_day as _previous_hk_trading_day,
    previous_weekday as _previous_weekday,
    resolve_report_dates,
)
from professional.fact_checker import run_fact_check
from professional.llm_enhancer import generate_llm_sections
from professional.report_quality import build_report_quality
from professional.report_builder import render_professional_report
from professional.trend_pack import collect_hk_trend_pack_data, generate_hk_trend_pack, summarize_hk_trend_pack_data


DEFAULT_DATA_STEP_TIMEOUT_SECONDS = float(os.environ.get("DMD_DATA_STEP_TIMEOUT_SECONDS", "90"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the professional morning research briefing.")
    parser.add_argument("--date", type=str, default="", help="Compatibility override: use the same completed date for review, global, and Hong Kong local data.")
    parser.add_argument("--review-date", type=str, default="", help="Calendar date being reviewed in YYYY-MM-DD format. Defaults to the previous calendar day.")
    parser.add_argument("--global-date", type=str, default="", help="Last completed global market date for US/Europe/FX/commodities in YYYY-MM-DD format.")
    parser.add_argument("--hk-date", type=str, default="", help="Last completed Hong Kong / China local data date in YYYY-MM-DD format.")
    parser.add_argument("--briefing-date", type=str, default="", help="Morning briefing date in YYYY-MM-DD format. Defaults to today in the configured timezone.")
    parser.add_argument("--output-dir", type=str, default="", help="Override the output directory.")
    parser.add_argument("--config", type=str, default="", help="Optional JSON config path.")
    parser.add_argument("--skip-charts", action="store_true", help="Skip chart generation.")
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip dashboard image generation.")
    parser.add_argument("--skip-daily-chart", action="store_true", help="Skip the dedicated Daily One Chart image.")
    parser.add_argument("--skip-trend-pack", action="store_true", help="Skip the Hong Kong Trend Pack image.")
    parser.add_argument("--no-llm", action="store_true", help="Disable the optional LLM overlay.")
    parser.add_argument("--debug", action="store_true", help="Persist intermediate raw payloads.")
    return parser.parse_args()


def _configure_console_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def _configure_market_data_cache(output_dir: str) -> None:
    """Keep third-party quote caches inside the project output tree."""
    cache_dir = os.path.join(output_dir, "raw", "runtime_cache", "yfinance")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        import yfinance as yf

        yf.set_tz_cache_location(cache_dir)
    except Exception as exc:
        print(f"[runtime] yfinance cache setup skipped: {type(exc).__name__}: {exc}")


def _run_external_step(
    label: str,
    func,
    fallback: Dict[str, Any],
    timeout_seconds: float = DEFAULT_DATA_STEP_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Run network-bound adapters with a bounded wait.

    A daemon worker lets the report degrade cleanly when a third-party library
    ignores its own socket timeout.  This is intentionally used only around
    data collection, where a structured fallback is safer than blocking the
    entire morning run.
    """
    result_queue: "queue.Queue[tuple[str, Any]]" = queue.Queue(maxsize=1)

    def _target() -> None:
        try:
            result_queue.put(("ok", func()), block=False)
        except Exception as exc:
            result_queue.put(("error", exc), block=False)

    worker = threading.Thread(target=_target, name=f"dmd-{label}", daemon=True)
    worker.start()
    worker.join(timeout=max(float(timeout_seconds), 1.0))

    if worker.is_alive():
        print(f"[runtime] {label} timed out after {timeout_seconds:.0f}s; using structured fallback.")
        return {**fallback, "status": "timeout", "error": f"{label} timed out after {timeout_seconds:.0f}s"}

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        return {**fallback, "status": "error", "error": f"{label} returned no payload"}

    if status == "ok" and isinstance(payload, dict):
        return payload
    if status == "ok":
        return {"status": "ok", "data": payload}

    print(f"[runtime] {label} failed: {type(payload).__name__}: {payload}")
    return {**fallback, "status": "error", "error": str(payload)}


def _fallback_market_payload(date_value: str) -> Dict[str, Any]:
    return {
        "requested_date": date_value,
        "effective_date": date_value,
        "summary_date": date_value,
        "timeseries": [],
        "summary": {},
        "quality": {"coverage": "0/0", "missing": 0, "fallback": 0, "status": "timeout"},
    }


def _fallback_sector_payload() -> Dict[str, Any]:
    return {
        "sector_news": {},
        "earnings_calendar": [],
        "analyst_changes": [],
        "hkex_announcements": {"status": "timeout", "items": [], "data": []},
        "formatted_text": "",
    }


def _fallback_movers_payload() -> Dict[str, Any]:
    return {
        "premarket_movers": {"gainers": [], "losers": [], "most_active": []},
        "etf_flows": [],
        "block_trades": [],
        "unusual_options": [],
        "short_sell": {"status": "timeout", "data": {}, "meta": {}},
        "formatted_text": "",
    }


def _fallback_trend_pack_payload() -> Dict[str, Any]:
    return {
        "southbound": [],
        "liquidity": [],
        "leadership": {"dates": [], "series": {}},
        "ah_heatmap": {"dates": [], "names": [], "matrix": []},
    }


def _attach_weekly_trend_summary(bundle: Dict[str, Any], trend_summary: Dict[str, Any]) -> None:
    if (bundle.get("day_mode", {}) or {}).get("mode") != "weekly_review" or not trend_summary:
        return
    weekly_review = bundle.setdefault("weekly_review", {})
    weekly_review["trend_summary"] = trend_summary
    if trend_summary.get("method_note"):
        weekly_review["method_note"] = trend_summary["method_note"]


def _extract_current_prices(market_data: Dict[str, Any]) -> Dict[str, float]:
    prices: Dict[str, float] = {}
    summary = market_data.get("summary", {}) or {}

    if "Equities" in summary:
        equities = summary["Equities"]
        if isinstance(equities.get("S&P 500"), dict):
            prices["SPX"] = equities["S&P 500"].get("Price", 0)
        if isinstance(equities.get("Nasdaq 100"), dict):
            prices["NDX"] = equities["Nasdaq 100"].get("Price", 0)
        if isinstance(equities.get("Hang Seng Index"), dict):
            prices["HSI"] = equities["Hang Seng Index"].get("Price", 0)
        if isinstance(equities.get("Hang Seng China Enterprises"), dict):
            prices["HSCEI"] = equities["Hang Seng China Enterprises"].get("Price", 0)

    if "FX" in summary:
        fx = summary["FX"]
        if isinstance(fx.get("DXY"), dict):
            prices["DXY"] = fx["DXY"].get("Price", 0)
        if isinstance(fx.get("USD/HKD"), dict):
            prices["USD/HKD"] = fx["USD/HKD"].get("Price", 0)

    if "Rates" in summary:
        rates = summary["Rates"]
        if isinstance(rates.get("10Y Treasury"), dict):
            prices["US10Y"] = rates["10Y Treasury"].get("Price", 0)

    return prices


def fetch_all_data(
    global_market_date: str,
    hk_data_date: str,
    briefing_date: str,
    review_date: str,
    config: Dict[str, Any],
    debug: bool = False,
    debug_dir: str = "",
) -> Dict[str, Any]:
    print(f"\n{'=' * 72}")
    print(
        "Collecting morning-briefing inputs | "
        f"briefing={briefing_date} | review={review_date} | global-request={global_market_date} | hk-request={hk_data_date}"
    )
    print(f"{'=' * 72}\n")

    payload: Dict[str, Any] = {}
    runtime_cache_dir = os.path.join((debug_dir or os.path.join("reports_professional", "debug")), "..", "raw", "runtime_cache")
    runtime_cache_dir = os.path.normpath(runtime_cache_dir)

    print("[1/7] Market data")
    day_mode = build_day_mode(review_date, config)
    payload["market"] = _run_external_step(
        "market-data",
        lambda: fetch_market_data(
            global_market_date,
            prefer_weekend_active_assets=not bool(day_mode.get("is_trading_day", True)),
        ),
        fallback=_fallback_market_payload(global_market_date),
        timeout_seconds=float(os.environ.get("DMD_MARKET_STEP_TIMEOUT_SECONDS", DEFAULT_DATA_STEP_TIMEOUT_SECONDS)),
    )

    print("[2/7] Sector news and HKEX announcements")
    payload["sector"] = _run_external_step(
        "sector-news",
        lambda: fetch_sector_data(hk_data_date, config=config, cache_dir=runtime_cache_dir),
        fallback=_fallback_sector_payload(),
        timeout_seconds=float(os.environ.get("DMD_NEWS_STEP_TIMEOUT_SECONDS", "45")),
    )

    print("[3/7] Movers, Stock Connect, AH premium, and HKEX short selling")
    payload["movers"] = _run_external_step(
        "market-movers",
        lambda: fetch_movers_data(hk_data_date, watchlists=(config or {}).get("watchlists", {})),
        fallback=_fallback_movers_payload(),
        timeout_seconds=float(os.environ.get("DMD_MOVER_STEP_TIMEOUT_SECONDS", "60")),
    )
    payload["stock_connect"] = _run_external_step(
        "stock-connect",
        lambda: fetch_stock_connect_data(hk_data_date),
        fallback={"status": "timeout", "data": {}, "meta": {}},
        timeout_seconds=float(os.environ.get("DMD_PUBLIC_STEP_TIMEOUT_SECONDS", "45")),
    )
    payload["ah_premium"] = _run_external_step(
        "ah-premium",
        lambda: fetch_ah_premium_data(hk_data_date),
        fallback={"status": "timeout", "data": {}, "meta": {}},
        timeout_seconds=float(os.environ.get("DMD_MOVER_STEP_TIMEOUT_SECONDS", "60")),
    )

    print("[4/7] Hong Kong local market data")
    payload["hk_local"] = _run_external_step(
        "hk-local",
        lambda: fetch_hk_local_data(
            hk_data_date,
            short_sell_data=(payload.get("movers", {}) or {}).get("short_sell"),
            stock_connect_data=payload.get("stock_connect", {}) or {},
            ah_premium_data=payload.get("ah_premium", {}) or {},
        ),
        fallback={"status": "timeout", "data": {}, "meta": {}},
        timeout_seconds=float(os.environ.get("DMD_PUBLIC_STEP_TIMEOUT_SECONDS", "45")),
    )

    print("[5/7] China local rates")
    payload["china_rates"] = _run_external_step(
        "china-rates",
        lambda: fetch_china_rates_data(hk_data_date),
        fallback={"status": "timeout", "data": {}, "meta": {}},
        timeout_seconds=float(os.environ.get("DMD_PUBLIC_STEP_TIMEOUT_SECONDS", "45")),
    )

    print("[6/7] Macro calendar")
    payload["macro"] = fetch_macro_data(briefing_date)

    print("[7/7] Risk radar + headlines")
    payload["risk"] = fetch_risk_data(_extract_current_prices(payload.get("market", {})))
    news_payload = _run_external_step(
        "rss-headlines",
        lambda: {"items": fetch_news(max_per_feed=10, cache_dir=runtime_cache_dir, cache_key=briefing_date)},
        fallback={"items": []},
        timeout_seconds=float(os.environ.get("DMD_NEWS_STEP_TIMEOUT_SECONDS", "45")),
    )
    payload["news"] = news_payload.get("items", []) if isinstance(news_payload, dict) else []

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"raw_inputs_{briefing_date}.json")
        with open(debug_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        print(f"Saved raw input payload: {debug_path}")

    return payload


def _save_chart_features(report_date: str, output_dir: str, chart_features: Dict[str, Any]) -> None:
    chart_dir = os.path.join(output_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)
    feature_path = os.path.join(chart_dir, f"features_{report_date}.json")
    with open(feature_path, "w", encoding="utf-8") as handle:
        json.dump(chart_features, handle, ensure_ascii=False, indent=2, default=str)


def _save_bundle(report_date: str, output_dir: str, bundle: Dict[str, Any]) -> None:
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    bundle_path = os.path.join(raw_dir, f"{report_date}_bundle.json")
    with open(bundle_path, "w", encoding="utf-8") as handle:
        json.dump(bundle, handle, ensure_ascii=False, indent=2, default=str)


def main() -> None:
    _configure_console_output()
    args = parse_args()
    config = load_professional_config(args.config or None)

    resolved_dates = resolve_report_dates(args, config)
    briefing_date = resolved_dates["briefing_date"]
    review_date = resolved_dates["review_date"]
    global_market_date = resolved_dates["global_market_date"]
    hk_data_date = resolved_dates["hk_data_date"]
    output_dir = args.output_dir or config.get("system", {}).get("output_dir", "reports_professional")
    os.makedirs(output_dir, exist_ok=True)
    _configure_market_data_cache(output_dir)

    debug_dir = os.path.join(output_dir, "debug")
    payload = fetch_all_data(
        global_market_date=global_market_date,
        hk_data_date=hk_data_date,
        briefing_date=briefing_date,
        review_date=review_date,
        config=config,
        debug=args.debug,
        debug_dir=debug_dir,
    )

    timeseries = (payload.get("market", {}) or {}).get("timeseries", []) or []
    chart_features = extract_chart_features(timeseries, tz=config.get("system", {}).get("timezone", "Asia/Shanghai"))
    output_label = briefing_date
    _save_chart_features(output_label, output_dir, chart_features)

    bundle = build_professional_bundle(
        report_date=review_date,
        briefing_date=briefing_date,
        global_market_date=global_market_date,
        hk_data_date=hk_data_date,
        config=config,
        market_data=payload.get("market", {}) or {},
        chart_features=chart_features,
        macro_data=payload.get("macro", {}) or {},
        sector_data=payload.get("sector", {}) or {},
        movers_data=payload.get("movers", {}) or {},
        stock_connect_data=payload.get("stock_connect", {}) or {},
        ah_premium_data=payload.get("ah_premium", {}) or {},
        risk_data=payload.get("risk", {}) or {},
        news_headlines=payload.get("news", []) or [],
        hk_local_data=payload.get("hk_local", {}) or {},
        china_rates_data=payload.get("china_rates", {}) or {},
    )

    trend_pack_data: Dict[str, Any] | None = None
    trend_cache_dir = os.path.join(output_dir, "raw", "trend_cache")
    if not args.skip_trend_pack and (bundle.get("day_mode", {}) or {}).get("mode") == "weekly_review":
        trend_pack_data = _run_external_step(
            "trend-pack-data",
            lambda: collect_hk_trend_pack_data(bundle, cache_dir=trend_cache_dir),
            fallback=_fallback_trend_pack_payload(),
            timeout_seconds=float(os.environ.get("DMD_TREND_PACK_STEP_TIMEOUT_SECONDS", "45")),
        )
        _attach_weekly_trend_summary(bundle, summarize_hk_trend_pack_data(trend_pack_data))

    llm_cache_dir = os.path.join(output_dir, "raw", "llm_cache")
    bundle["llm_sections"] = (
        {}
        if args.no_llm
        else generate_llm_sections(bundle=bundle, config=config, cache_dir=llm_cache_dir)
    )
    bundle["fact_check"] = run_fact_check(bundle)
    bundle["report_quality"] = build_report_quality(bundle)

    dashboard_rel_path = ""
    if not args.skip_dashboard:
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        dashboard_name = generate_dashboard(bundle, os.path.join(chart_dir, f"dashboard_{output_label}.png"))
        dashboard_rel_path = f"charts/{dashboard_name}"

    daily_chart_rel_path = ""
    if not args.skip_daily_chart:
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        daily_chart_name = generate_daily_one_chart(bundle, os.path.join(chart_dir, f"daily_one_chart_{output_label}.png"))
        bundle["daily_one_chart"] = {
            **daily_chart_name,
            "rel_path": f"charts/{daily_chart_name['path']}",
        }
        daily_chart_rel_path = bundle["daily_one_chart"]["rel_path"]

    trend_pack_rel_path = ""
    if not args.skip_trend_pack:
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        trend_pack_meta = generate_hk_trend_pack(
            bundle,
            os.path.join(chart_dir, f"hk_trend_pack_{output_label}.png"),
            trend_data=trend_pack_data,
            cache_dir=trend_cache_dir,
        )
        bundle["trend_pack"] = {
            **trend_pack_meta,
            "rel_path": f"charts/{trend_pack_meta['path']}",
        }
        trend_pack_rel_path = bundle["trend_pack"]["rel_path"]
        trend_summary = trend_pack_meta.get("weekly_summary", {}) if isinstance(trend_pack_meta, dict) else {}
        _attach_weekly_trend_summary(bundle, trend_summary)

    charts_section = (
        "_Supplementary visual appendix was skipped._"
        if args.skip_charts
        else render_chart_appendix(
            bundle=bundle,
            dashboard_rel_path=dashboard_rel_path,
            daily_chart_rel_path=daily_chart_rel_path,
            trend_pack_rel_path=trend_pack_rel_path,
        )
    )

    _save_bundle(output_label, output_dir, bundle)

    report = render_professional_report(
        bundle=bundle,
        charts_section=charts_section,
        dashboard_rel_path=dashboard_rel_path,
        daily_chart_rel_path=daily_chart_rel_path,
        trend_pack_rel_path=trend_pack_rel_path,
    )

    output_file = os.path.join(output_dir, f"{output_label}_morning_briefing.md")
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"\n{'=' * 72}")
    print("Morning briefing generated successfully")
    print(f"Report: {output_file}")
    if dashboard_rel_path:
        print(f"Dashboard: {os.path.join(output_dir, dashboard_rel_path)}")
    if daily_chart_rel_path:
        print(f"Daily One Chart: {os.path.join(output_dir, daily_chart_rel_path)}")
    if trend_pack_rel_path:
        print(f"Hong Kong Trend Pack: {os.path.join(output_dir, trend_pack_rel_path)}")
    print(f"{'=' * 72}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)
