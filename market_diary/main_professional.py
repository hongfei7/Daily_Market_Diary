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
from pathlib import Path
from typing import Any, Dict

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from market_diary.modules.adapter_ah_premium import fetch_ah_premium_data
from market_diary.modules.adapter_stockconnect import fetch_stock_connect_data
from market_diary.modules.chart_features import extract_chart_features
from market_diary.modules.china_rates import fetch_china_rates_data
from market_diary.modules.data_fetcher import fetch_market_data, fetch_news
from market_diary.modules.hk_local_data import fetch_hk_local_data
from market_diary.modules.macro_calendar import fetch_macro_data
from market_diary.modules.market_movers import fetch_movers_data
from market_diary.modules.provenance import (
    audit_source_provenance,
    collect_source_provenance,
    ensure_payload_provenance,
    provenance_record,
)
from market_diary.modules.risk_radar import fetch_risk_data
from market_diary.modules.sector_news import fetch_sector_data
from market_diary.professional.analytics import build_professional_bundle
from market_diary.professional.chart_appendix import render_chart_appendix
from market_diary.professional.catalyst_radar import generate_catalyst_radar
from market_diary.professional.config import load_professional_config
from market_diary.professional.daily_one_chart import generate_daily_one_chart
from market_diary.professional.dashboard import generate_dashboard
from market_diary.professional.date_policy import (
    build_day_mode,
    previous_calendar_day as _previous_calendar_day,
    previous_hk_trading_day as _previous_hk_trading_day,
    previous_weekday as _previous_weekday,
    resolve_report_dates,
)
from market_diary.professional.fact_checker import apply_fact_check_fallbacks, run_fact_check
from market_diary.professional.llm_enhancer import generate_llm_sections
from market_diary.professional.performance import refresh_performance_tracking
from market_diary.professional.skill_shadow import generate_skill_shadow
from market_diary.professional.metric_history import load_history as load_metric_history
from market_diary.professional.metric_history import record_observations as record_metric_observations
from market_diary.professional.metric_history import save_history as save_metric_history
from market_diary.professional.prose_guard import check_markdown as check_prose
from market_diary.professional.prose_guard import summarize as summarize_prose
from market_diary.professional.report_quality import build_report_quality
from market_diary.professional.report_builder import render_professional_report
from market_diary.professional.source_health import build_source_health
from market_diary.professional.trend_pack import collect_hk_trend_pack_data, generate_hk_trend_pack, summarize_hk_trend_pack_data


DEFAULT_DATA_STEP_TIMEOUT_SECONDS = float(os.environ.get("DMD_DATA_STEP_TIMEOUT_SECONDS", "90"))


def _error_summary(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


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
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip the dashboard and companion Catalyst & Event Radar images.")
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
        outcome: tuple[str, Any]
        try:
            outcome = ("ok", func())
        except BaseException as exc:
            outcome = ("error", exc)
        try:
            result_queue.put(outcome, block=False)
        except queue.Full:
            pass

    worker = threading.Thread(target=_target, name=f"dmd-{label}", daemon=True)
    worker.start()
    worker.join(timeout=max(float(timeout_seconds), 1.0))

    if worker.is_alive():
        print(f"[runtime] {label} timed out after {timeout_seconds:.0f}s; using structured fallback.")
        return {
            **fallback,
            "status": "timeout",
            "error": f"{label} timed out after {timeout_seconds:.0f}s",
            "error_type": "TimeoutError",
            "step": label,
        }

    try:
        status, payload = result_queue.get_nowait()
    except queue.Empty:
        return {
            **fallback,
            "status": "error",
            "error": f"{label} returned no payload",
            "error_type": "RuntimeError",
            "step": label,
        }

    if status == "ok" and isinstance(payload, dict):
        return payload
    if status == "ok":
        return {"status": "ok", "data": payload}

    print(f"[runtime] {label} failed: {_error_summary(payload)}")
    return {
        **fallback,
        "status": "error",
        "error": str(payload),
        "error_type": type(payload).__name__,
        "step": label,
    }


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
        "earnings_calendar_status": "unavailable",
        "analyst_changes": [],
        "analyst_changes_status": "unavailable",
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


def _attach_provenance(payload: Dict[str, Any], global_date: str, hk_date: str, briefing_date: str) -> None:
    """Normalize aggregate provenance for adapters that predate the shared schema."""
    market = payload.get("market", {}) or {}
    market_quality = (market.get("quality", {}) or market.get("meta", {}).get("market_quality", {}) or {})
    market_available = int(market_quality.get("available", 0) or 0)
    market_total = int(market_quality.get("total", 0) or 0)
    market_status = "ok" if market_available and market_available == market_total else "partial" if market_available else "unavailable"
    market_records = []
    for category, category_items in (market.get("summary", {}) or {}).items():
        if not isinstance(category_items, dict):
            continue
        for raw_name, item in category_items.items():
            if not isinstance(item, dict):
                continue
            quality = str(item.get("Quality", "fresh") or "fresh").lower()
            status = "stale_public" if quality == "stale" else "ok"
            market_records.append(
                provenance_record(
                    source_name=f"Yahoo Finance: {item.get('Display Name') or raw_name}",
                    source_url="https://finance.yahoo.com/",
                    as_of=str(item.get("As Of") or global_date),
                    source_type="public",
                    status=status,
                    confidence=0.8 if status == "ok" else 0.55,
                    note=(
                        f"{category} quote; instrument_id={item.get('Instrument ID', 'unmapped')}; "
                        f"symbol={item.get('Source Symbol', 'unknown')}; change_unit={item.get('Change Unit', 'pct')}."
                    ),
                )
            )
    if market_records:
        market["provenance"] = market_records
    payload["market"] = ensure_payload_provenance(
        market,
        source_name="Yahoo Finance market quotes",
        source_url="https://finance.yahoo.com/",
        as_of=str((market.get("meta", {}) or {}).get("summary_date") or global_date),
        source_type="public" if market_status != "unavailable" else "unavailable",
        status=market_status,
        confidence=0.8 if market_status == "ok" else 0.55 if market_status == "partial" else 0.0,
        note="Public market quotes used for cross-asset levels and changes.",
    )

    sector = payload.get("sector", {}) or {}
    sector_status = str(sector.get("status", "unavailable") or "unavailable")
    payload["sector"] = ensure_payload_provenance(
        sector,
        source_name="Public market-news feeds and HKEXnews",
        source_url="https://www1.hkexnews.hk/",
        as_of=hk_date,
        source_type="public" if sector_status in {"ok", "partial"} else "unavailable",
        status=sector_status,
        confidence=0.75 if sector_status == "ok" else 0.5 if sector_status == "partial" else 0.0,
        note="Aggregate sector-news and official-announcement input.",
    )

    movers = payload.get("movers", {}) or {}
    movers_status = str(movers.get("status", "unavailable") or "unavailable")
    payload["movers"] = ensure_payload_provenance(
        movers,
        source_name="Yahoo Finance activity proxies and HKEX short selling",
        source_url="https://www.hkex.com.hk/",
        as_of=hk_date,
        source_type="derived" if movers_status in {"ok", "partial"} else "unavailable",
        status=movers_status,
        confidence=0.7 if movers_status == "ok" else 0.5 if movers_status == "partial" else 0.0,
        note="Aggregate mover proxies and official short-selling input.",
    )

    macro = payload.get("macro", {}) or {}
    macro_status = str(macro.get("status", "unavailable") or "unavailable")
    payload["macro"] = ensure_payload_provenance(
        macro,
        source_name="Macro calendar",
        source_url="",
        as_of=briefing_date,
        source_type="unavailable" if macro_status == "unavailable" else "public",
        status=macro_status,
        confidence=0.0 if macro_status == "unavailable" else 0.75,
        note="No verified macro-calendar provider is configured." if macro_status == "unavailable" else "Verified macro-calendar input.",
    )

    risk = payload.get("risk", {}) or {}
    risk_status = str(risk.get("status", "unavailable") or "unavailable")
    payload["risk"] = ensure_payload_provenance(
        risk,
        source_name="Risk and sentiment event feed",
        source_url="",
        as_of=briefing_date,
        source_type="unavailable" if risk_status == "unavailable" else "public",
        status=risk_status,
        confidence=0.0 if risk_status == "unavailable" else 0.75,
        note="No verified risk-event provider is configured." if risk_status == "unavailable" else "Verified risk-event input.",
    )

    stock_connect = payload.get("stock_connect", {}) or {}
    stock_meta = stock_connect.get("meta", {}) or {}
    stock_status = str(stock_connect.get("status", "unavailable") or "unavailable")
    payload["stock_connect"] = ensure_payload_provenance(
        stock_connect,
        source_name=str(stock_meta.get("source") or "HKEX Stock Connect"),
        source_url=str(stock_meta.get("source_url") or "https://www.hkex.com.hk/Mutual-Market/Stock-Connect"),
        as_of=str(stock_meta.get("effective_date") or hk_date),
        source_type="official" if stock_status == "ok" else "unavailable",
        status=stock_status,
        confidence=0.95 if stock_status == "ok" else 0.0,
        note="Official Stock Connect daily disclosure.",
    )

    ah_premium = payload.get("ah_premium", {}) or {}
    ah_meta = ah_premium.get("meta", {}) or {}
    ah_status = str(ah_premium.get("status", "unavailable") or "unavailable")
    payload["ah_premium"] = ensure_payload_provenance(
        ah_premium,
        source_name=str(ah_meta.get("source") or "Yahoo Finance A/H quote calculation"),
        source_url="https://finance.yahoo.com/",
        as_of=str(ah_meta.get("effective_date") or hk_date),
        source_type="derived" if ah_status == "ok" else "unavailable",
        status="derived" if ah_status == "ok" else ah_status,
        confidence=0.7 if ah_status == "ok" else 0.0,
        note="Calculated A/H premium using public A-share, H-share, and FX closes.",
    )

    hk_local = payload.get("hk_local", {}) or {}
    hk_meta = hk_local.get("meta", {}) or {}
    hk_status = str(hk_local.get("status", "unavailable") or "unavailable")
    payload["hk_local"] = ensure_payload_provenance(
        hk_local,
        source_name="HKEX and HKMA official public data",
        source_url="https://api.hkma.gov.hk/public/market-data-and-statistics/",
        as_of=str(hk_meta.get("turnover_effective_date") or hk_meta.get("hkma_effective_date") or hk_date),
        source_type="official" if hk_status in {"ok", "partial"} else "unavailable",
        status=hk_status,
        confidence=0.9 if hk_status == "ok" else 0.65 if hk_status == "partial" else 0.0,
        note="Aggregate provenance for official Hong Kong turnover and liquidity metrics.",
    )

    china_rates = payload.get("china_rates", {}) or {}
    rates_meta = china_rates.get("meta", {}) or {}
    rates_status = str(china_rates.get("status", "unavailable") or "unavailable")
    payload["china_rates"] = ensure_payload_provenance(
        china_rates,
        source_name="Eastmoney Treasury Yield History",
        source_url="https://datacenter-web.eastmoney.com/api/data/v1/get",
        as_of=str(rates_meta.get("effective_date") or hk_date),
        source_type="public" if rates_status == "ok" else "unavailable",
        status=rates_status,
        confidence=0.75 if rates_status == "ok" else 0.0,
        note="Public cross-market treasury-yield history.",
    )

    news_source = payload.get("news_source", {}) or {}
    news_items = news_source.get("items", []) or []
    payload["news_source"] = ensure_payload_provenance(
        news_source,
        source_name="Public market-news RSS feeds",
        source_url="https://www.cnbc.com/id/100003114/device/rss/rss.html",
        as_of=briefing_date,
        source_type="public" if news_items else "unavailable",
        status="ok" if news_items else "unavailable",
        confidence=0.65 if news_items else 0.0,
        note="Headline discovery only; a headline is not sufficient evidence for a numeric or company-event claim.",
    )


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
    payload["news_source"] = news_payload if isinstance(news_payload, dict) else {"items": []}

    _attach_provenance(payload, global_market_date, hk_data_date, briefing_date)

    if debug and debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        debug_path = os.path.join(debug_dir, f"raw_inputs_{briefing_date}.json")
        with open(debug_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        print(f"Saved raw input payload: {debug_path}")

    return payload


def _metric_history_path(output_dir: str) -> str:
    """Sit beside the signal ledger; both are append-only run histories."""
    return os.path.join(output_dir, "performance", "metric_history.json")


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


def _save_diagnostic(report_date: str, output_dir: str, name: str, payload: Dict[str, Any]) -> None:
    raw_dir = os.path.join(output_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    path = os.path.join(raw_dir, f"{report_date}_{name}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str, sort_keys=True)


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

    # Trailing history turns a raw level into a percentile, so the report can say
    # whether a reading is actually unusual rather than just clearing a constant.
    metric_history_path = _metric_history_path(output_dir)
    try:
        metric_history = load_metric_history(metric_history_path)
    except Exception as exc:
        print(f"[runtime] Metric history load failed (non-fatal): {_error_summary(exc)}")
        metric_history = {"observations": {}}

    bundle = build_professional_bundle(
        metric_history=metric_history,
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
    # Append after the bundle is built so today's value never feeds its own
    # percentile; the store is append-only, so a rerun cannot reshape history.
    try:
        record_metric_observations(
            metric_history,
            hk_data_date,
            (bundle.get("hk_local", {}) or {}),
        )
        save_metric_history(metric_history, metric_history_path)
    except Exception as exc:
        print(f"[runtime] Metric history update failed (non-fatal): {_error_summary(exc)}")

    source_payloads = {
        "market_data": payload.get("market", {}) or {},
        "sector_news": payload.get("sector", {}) or {},
        "movers": payload.get("movers", {}) or {},
        "stock_connect": payload.get("stock_connect", {}) or {},
        "ah_premium": payload.get("ah_premium", {}) or {},
        "hk_local": payload.get("hk_local", {}) or {},
        "china_rates": payload.get("china_rates", {}) or {},
        "macro_calendar": payload.get("macro", {}) or {},
        "risk_feed": payload.get("risk", {}) or {},
        "rss_headlines": payload.get("news_source", {}) or {},
    }
    bundle["source_provenance"] = collect_source_provenance(source_payloads)
    bundle["provenance_audit"] = audit_source_provenance(source_payloads)
    bundle["source_health"] = build_source_health(
        bundle["source_provenance"],
        reference_date=briefing_date,
        policies=config.get("source_health", {}) or {},
    )

    day_mode = (bundle.get("day_mode", {}) or {})
    skip_all_charts = bool(args.skip_charts)
    should_render_dashboard = not skip_all_charts and not args.skip_dashboard
    should_render_catalyst_radar = should_render_dashboard
    should_render_daily_chart = not skip_all_charts and not args.skip_daily_chart
    should_render_trend_pack = (
        not skip_all_charts
        and not args.skip_trend_pack
        and day_mode.get("mode") == "weekly_review"
    )

    trend_pack_data: Dict[str, Any] | None = None
    trend_cache_dir = os.path.join(output_dir, "raw", "trend_cache")
    if should_render_trend_pack:
        trend_pack_data = _run_external_step(
            "trend-pack-data",
            lambda: collect_hk_trend_pack_data(bundle, cache_dir=trend_cache_dir),
            fallback=_fallback_trend_pack_payload(),
            timeout_seconds=float(os.environ.get("DMD_TREND_PACK_STEP_TIMEOUT_SECONDS", "45")),
        )
        _attach_weekly_trend_summary(bundle, summarize_hk_trend_pack_data(trend_pack_data))

    llm_cache_dir = os.path.join(output_dir, "raw", "llm_cache")
    try:
        bundle["llm_sections"] = (
            {}
            if args.no_llm
            else generate_llm_sections(bundle=bundle, config=config, cache_dir=llm_cache_dir)
        )
    except Exception as exc:
        print(f"[runtime] LLM overlay crashed (non-fatal, using empty sections): {_error_summary(exc)}")
        bundle["llm_sections"] = {"task_meta": {"status": "error", "error": _error_summary(exc)}}
    try:
        bundle["skill_shadow"] = (
            {"status": "disabled", "mode": "shadow", "publish": False, "skills": {}}
            if args.no_llm
            else generate_skill_shadow(bundle=bundle, config=config, cache_dir=llm_cache_dir)
        )
    except Exception as exc:
        print(f"[runtime] Skill shadow run crashed (non-fatal): {_error_summary(exc)}")
        bundle["skill_shadow"] = {
            "status": "error",
            "mode": "shadow",
            "publish": False,
            "error": _error_summary(exc),
            "skills": {},
        }
    try:
        initial_fact_check = run_fact_check(bundle)
        degraded_fields = apply_fact_check_fallbacks(bundle, initial_fact_check)
        bundle["fact_check"] = run_fact_check(bundle)
        bundle["fact_check"]["degraded_fields"] = degraded_fields
        bundle["fact_check"]["initial_summary"] = initial_fact_check.get("summary", "")
        if degraded_fields and bundle["fact_check"].get("status") == "ok":
            bundle["fact_check"]["status"] = "warning"
            bundle["fact_check"]["summary"] = (
                f"{bundle['fact_check'].get('summary', '')} "
                f"Replaced {len(degraded_fields)} unsafe narrative field(s) with deterministic fallback copy."
            ).strip()
    except Exception as exc:
        print(f"[runtime] Fact-check guardrail failed (non-fatal): {_error_summary(exc)}")
        bundle["fact_check"] = {"status": "error", "error": _error_summary(exc)}
    try:
        bundle["report_quality"] = build_report_quality(bundle)
    except Exception as exc:
        print(f"[runtime] Report quality scoring failed (non-fatal): {_error_summary(exc)}")
        bundle["report_quality"] = {"status": "error", "error": _error_summary(exc)}

    performance_config = config.get("performance", {}) or {}
    if performance_config.get("enabled", True):
        try:
            performance_chart_path = None
            if not skip_all_charts:
                chart_dir = os.path.join(output_dir, "charts")
                os.makedirs(chart_dir, exist_ok=True)
                performance_chart_path = os.path.join(chart_dir, f"signal_performance_{output_label}.png")
            bundle["performance"] = refresh_performance_tracking(
                bundle,
                output_dir=output_dir,
                archive_root=os.path.join(output_dir, "archive"),
                chart_path=performance_chart_path,
                benchmarks=tuple(performance_config.get("benchmarks", ["Hang Seng Index", "Hang Seng TECH ETF (3033.HK)"])),
                horizons=tuple(performance_config.get("horizons_sessions", [1, 5, 20])),
                cost_bps=float(performance_config.get("transaction_cost_bps", 10.0)),
            )
            if bundle["performance"].get("chart_path"):
                bundle["performance"]["rel_path"] = f"charts/{bundle['performance']['chart_path']}"
        except Exception as exc:
            print(f"[runtime] Performance tracking failed (non-fatal): {_error_summary(exc)}")
            bundle["performance"] = {
                "status": "error",
                "error": _error_summary(exc),
                "methodology": {"look_ahead_guard": True},
            }
    else:
        bundle["performance"] = {"status": "disabled", "methodology": {"look_ahead_guard": True}}

    dashboard_rel_path = ""
    if should_render_dashboard:
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        dashboard_name = generate_dashboard(bundle, os.path.join(chart_dir, f"dashboard_{output_label}.png"))
        dashboard_rel_path = f"charts/{dashboard_name}"

    catalyst_radar_rel_path = ""
    if should_render_catalyst_radar:
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        catalyst_radar_meta = generate_catalyst_radar(
            bundle,
            os.path.join(chart_dir, f"catalyst_radar_{output_label}.png"),
        )
        bundle["catalyst_radar"] = {
            **catalyst_radar_meta,
            "rel_path": f"charts/{catalyst_radar_meta['path']}",
        }
        catalyst_radar_rel_path = bundle["catalyst_radar"]["rel_path"]

    daily_chart_rel_path = ""
    if should_render_daily_chart:
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        daily_chart_name = generate_daily_one_chart(bundle, os.path.join(chart_dir, f"daily_one_chart_{output_label}.png"))
        bundle["daily_one_chart"] = {
            **daily_chart_name,
            "rel_path": f"charts/{daily_chart_name['path']}",
        }
        daily_chart_rel_path = bundle["daily_one_chart"]["rel_path"]

    trend_pack_rel_path = ""
    if should_render_trend_pack:
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
            catalyst_radar_rel_path=catalyst_radar_rel_path,
            daily_chart_rel_path=daily_chart_rel_path,
            trend_pack_rel_path=trend_pack_rel_path,
        )
    )

    _save_diagnostic(
        output_label,
        output_dir,
        "llm_health",
        ((bundle.get("llm_sections", {}) or {}).get("task_meta", {}) or {}).get("health", {}) or {},
    )
    _save_diagnostic(output_label, output_dir, "source_health", bundle.get("source_health", {}) or {})
    _save_diagnostic(output_label, output_dir, "performance_summary", bundle.get("performance", {}) or {})
    _save_bundle(output_label, output_dir, bundle)

    report = render_professional_report(
        bundle=bundle,
        charts_section=charts_section,
        dashboard_rel_path=dashboard_rel_path,
        catalyst_radar_rel_path=catalyst_radar_rel_path,
        daily_chart_rel_path=daily_chart_rel_path,
        trend_pack_rel_path=trend_pack_rel_path,
    )

    # Prose defects only exist in the rendered text, so the guard runs on the
    # exact markdown that ships. Findings feed back into the quality score, which
    # means the report is rendered a second time to carry its own verdict.
    try:
        prose_findings = check_prose(report)
        bundle["prose_guard"] = summarize_prose(prose_findings)
        if prose_findings:
            print(f"[runtime] Prose guard flagged {len(prose_findings)} defect(s) in the rendered report.")
            for item in prose_findings[:5]:
                print(f"          L{item['line']} [{item['rule']}] {item['text'][:100]}")
            bundle["report_quality"] = build_report_quality(bundle)
            report = render_professional_report(
                bundle=bundle,
                charts_section=charts_section,
                dashboard_rel_path=dashboard_rel_path,
                catalyst_radar_rel_path=catalyst_radar_rel_path,
                daily_chart_rel_path=daily_chart_rel_path,
                trend_pack_rel_path=trend_pack_rel_path,
            )
    except Exception as exc:
        print(f"[runtime] Prose guard failed (non-fatal): {_error_summary(exc)}")
        bundle["prose_guard"] = {"status": "error", "error": _error_summary(exc)}

    _save_diagnostic(output_label, output_dir, "prose_guard", bundle.get("prose_guard", {}))

    output_file = os.path.join(output_dir, f"{output_label}_morning_briefing.md")
    with open(output_file, "w", encoding="utf-8") as handle:
        handle.write(report)

    print(f"\n{'=' * 72}")
    print("Morning briefing generated successfully")
    print(f"Report: {output_file}")
    if dashboard_rel_path:
        print(f"Dashboard: {os.path.join(output_dir, dashboard_rel_path)}")
    if catalyst_radar_rel_path:
        print(f"Catalyst Radar: {os.path.join(output_dir, catalyst_radar_rel_path)}")
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
