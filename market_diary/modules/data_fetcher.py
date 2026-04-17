"""Market and headline data adapters for the professional morning briefing."""

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import feedparser
import pandas as pd
import requests
import yfinance as yf

from modules.text_normalizer import normalize_news_text


NEWS_REQUEST_TIMEOUT = (5, 12)
NEWS_USER_AGENT = "DailyMarketDiary/1.0"


TICKERS = {
    "Rates": {
        "13W T-Bill": "^IRX",
        "5Y Treasury": "^FVX",
        "10Y Treasury": "^TNX",
        "30Y Treasury": "^TYX",
    },
    "FX": {
        "DXY": "DX-Y.NYB",
        "USD/JPY": "JPY=X",
        "EUR/USD": "EURUSD=X",
        "USD/CNH": "CNH=F",
        "USD/HKD": "HKD=X",
    },
    "Commodities": {
        "Crude Oil": "CL=F",
        "Brent Crude": "BZ=F",
        "WTI 6M": "CL4=F",
        "Gold": "GC=F",
        "Copper": "HG=F",
    },
    "Equities": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Dow Jones": "^DJI",
        "Euro Stoxx 50": "^STOXX50E",
        "Hang Seng Index": "^HSI",
        "Hang Seng China Enterprises": "^HSCE",
        "Hang Seng TECH ETF": "3033.HK",
        "CSI 300": "000300.SS",
        "ChiNext Index": "399006.SZ",
        "Nikkei 225": "^N225",
        "Shanghai Composite": "000001.SS",
        "China Large-Cap (FXI)": "FXI",
    },
    "Vol": {
        "VIX": "^VIX",
        "MOVE": "^MOVE",
    },
    "Credit": {
        "IG (LQD)": "LQD",
        "HY (HYG)": "HYG",
    },
    "Crypto": {
        "Bitcoin": "BTC-USD",
        "Ethereum": "ETH-USD",
    },
}

# Stable aliases used by the charting layer and feature extraction map.
PLOT_ALIASES = {
    ("Rates", "13W T-Bill"): "UST3M",
    ("Rates", "5Y Treasury"): "UST5Y",
    ("Rates", "10Y Treasury"): "UST10Y",
    ("Rates", "30Y Treasury"): "UST30Y",
    ("FX", "DXY"): "DXY",
    ("FX", "USD/JPY"): "USD/JPY",
    ("FX", "EUR/USD"): "EUR/USD",
    ("FX", "USD/CNH"): "USD/CNH",
    ("FX", "USD/HKD"): "USD/HKD",
    ("Commodities", "Crude Oil"): "WTI",
    ("Commodities", "Brent Crude"): "BRENT",
    ("Commodities", "WTI 6M"): "WTI_6M",
    ("Commodities", "Gold"): "GOLD",
    ("Commodities", "Copper"): "COPPER",
    ("Equities", "S&P 500"): "SPX",
    ("Equities", "Nasdaq 100"): "NDX",
    ("Equities", "Dow Jones"): "DJI",
    ("Equities", "Euro Stoxx 50"): "STOXX50E",
    ("Equities", "Hang Seng Index"): "HSI",
    ("Equities", "Hang Seng China Enterprises"): "HSCEI",
    ("Equities", "Hang Seng TECH ETF"): "HSTECH",
    ("Equities", "CSI 300"): "CSI300",
    ("Equities", "ChiNext Index"): "CHINEXT",
    ("Equities", "Nikkei 225"): "NIKKEI225",
    ("Equities", "Shanghai Composite"): "SHCOMP",
    ("Equities", "China Large-Cap (FXI)"): "FXI",
    ("Vol", "VIX"): "VIX",
    ("Vol", "MOVE"): "MOVE",
    ("Credit", "IG (LQD)"): "LQD",
    ("Credit", "HY (HYG)"): "HYG",
    ("Crypto", "Bitcoin"): "BTC",
    ("Crypto", "Ethereum"): "ETH",
}

RSS_FEEDS = [
    "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "http://feeds.reuters.com/reuters/businessNews",
]

# Expanding this list increases request volume. Trim it if rate limits become an issue.
INTRADAY_WHITELIST = {
    ("FX", "DXY"),
    ("FX", "USD/JPY"),
    ("FX", "EUR/USD"),
    ("FX", "USD/CNH"),
    ("FX", "USD/HKD"),
    ("Commodities", "Crude Oil"),
    ("Commodities", "Brent Crude"),
    ("Commodities", "WTI 6M"),
    ("Commodities", "Gold"),
    ("Crypto", "Bitcoin"),
    ("Rates", "13W T-Bill"),
    ("Rates", "5Y Treasury"),
    ("Rates", "10Y Treasury"),
    ("Rates", "30Y Treasury"),
    ("Equities", "S&P 500"),
    ("Equities", "Nasdaq 100"),
    ("Equities", "Dow Jones"),
    ("Equities", "Euro Stoxx 50"),
    ("Equities", "Hang Seng Index"),
    ("Equities", "Hang Seng China Enterprises"),
    ("Equities", "Hang Seng TECH ETF"),
    ("Equities", "Nikkei 225"),
    ("Equities", "China Large-Cap (FXI)"),
    ("Vol", "VIX"),
    ("Vol", "MOVE"),
    ("Credit", "IG (LQD)"),
    ("Credit", "HY (HYG)"),
}

DEFAULT_INTRADAY_INTERVAL = "5m"
ALWAYS_OPEN_CATEGORIES = {"Crypto"}
SUMMARY_LOOKBACK_DAYS = 14
SUMMARY_STALE_AFTER_DAYS = 1
SUMMARY_FETCH_ATTEMPTS = 2


def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _flatten_download_df(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.empty:
        return df

    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
        out = out.loc[:, ~out.columns.duplicated()]
    return out


def _request_download(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    sink = io.StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            df = yf.download(
                symbol,
                start=start,
                end=end,
                interval=interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
    except Exception:
        return pd.DataFrame()
    return _flatten_download_df(df)


def _request_history(
    symbol: str,
    start: Optional[datetime],
    end: Optional[datetime],
    interval: str,
    period: Optional[str] = None,
) -> pd.DataFrame:
    sink = io.StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            ticker = yf.Ticker(symbol)
            kwargs: Dict[str, Any] = {
                "interval": interval,
                "auto_adjust": False,
            }
            if period:
                kwargs["period"] = period
            else:
                kwargs["start"] = start
                kwargs["end"] = end
            df = ticker.history(**kwargs)
    except Exception:
        return pd.DataFrame()
    return _flatten_download_df(df)


def _filter_history_window(df: pd.DataFrame, start: datetime, end: datetime) -> pd.DataFrame:
    out = _flatten_download_df(df)
    if out.empty:
        return out

    index = pd.to_datetime(out.index, errors="coerce")
    if getattr(index, "tz", None) is not None:
        start_ts = pd.Timestamp(start).tz_localize(index.tz)
        end_ts = pd.Timestamp(end).tz_localize(index.tz)
    else:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

    mask = (index >= start_ts) & (index < end_ts)
    filtered = out.loc[mask]
    if filtered.empty:
        return pd.DataFrame()
    filtered.index = index[mask]
    return filtered


def _candidate_periods(interval: str) -> List[str]:
    if interval == "1d":
        return ["1mo", "3mo"]
    if interval.endswith("m"):
        return ["5d", "1mo"]
    return ["1mo"]


def _safe_download_with_source(
    symbol: str,
    start: datetime,
    end: datetime,
    interval: str,
) -> Tuple[pd.DataFrame, str]:
    for attempt in range(1, SUMMARY_FETCH_ATTEMPTS + 1):
        methods = [
            ("download_window", lambda: _request_download(symbol, start, end, interval)),
            ("history_window", lambda: _request_history(symbol, start, end, interval)),
        ]

        for period in _candidate_periods(interval):
            methods.append(
                (
                    f"history_period:{period}",
                    lambda period=period: _request_history(symbol, None, None, interval, period=period),
                )
            )

        for source, getter in methods:
            df = _filter_history_window(getter(), start, end)
            if not df.empty:
                suffix = "" if attempt == 1 else f":retry{attempt}"
                return df, f"{source}{suffix}"

    return pd.DataFrame(), "unavailable"


def _safe_download(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    df, _ = _safe_download_with_source(symbol, start, end, interval)
    return df


def _normalize_intraday_df(hist: pd.DataFrame, name: str, category: str, ticker: str) -> pd.DataFrame:
    """Normalize intraday downloads for charting and feature extraction."""
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["time", "symbol", "name", "ticker", "price", "Category"])

    df = _flatten_download_df(hist).reset_index()

    if "Datetime" in df.columns:
        time_col = "Datetime"
    elif "Date" in df.columns:
        time_col = "Date"
    else:
        time_col = df.columns[0]

    price_col = None
    for candidate in ["Close", "close", "Adj Close", "adjclose"]:
        if candidate in df.columns:
            price_col = candidate
            break

    if price_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return pd.DataFrame(columns=["time", "symbol", "name", "ticker", "price", "Category"])
        price_col = numeric_cols[0]

    alias = PLOT_ALIASES.get((category, name), name)

    return (
        pd.DataFrame(
            {
                "time": pd.to_datetime(df[time_col], errors="coerce"),
                "price": pd.to_numeric(df[price_col], errors="coerce"),
                "symbol": alias,
                "name": name,
                "ticker": ticker,
                "Category": category,
            }
        )
        .dropna(subset=["time", "price"])
    )


def _get_effective_intraday_date(
    requested_date: str,
    interval: str,
    max_lookback_days: int = 4,
) -> Tuple[str, List[pd.DataFrame]]:
    base = _parse_date(requested_date)
    crypto_only_candidate: Optional[Tuple[str, List[pd.DataFrame]]] = None

    for offset in range(max_lookback_days + 1):
        day = base - timedelta(days=offset)
        start = day
        end = day + timedelta(days=1)

        timeseries: List[pd.DataFrame] = []
        market_rows = 0
        crypto_rows = 0

        for category, items in TICKERS.items():
            if category not in ALWAYS_OPEN_CATEGORIES and day.weekday() >= 5:
                continue

            for name, ticker in items.items():
                if (category, name) not in INTRADAY_WHITELIST:
                    continue

                hist = _safe_download(ticker, start=start, end=end, interval=interval)
                df_plot = _normalize_intraday_df(hist, name=name, category=category, ticker=ticker)
                if df_plot.empty:
                    continue

                rows = len(df_plot)
                if category in ALWAYS_OPEN_CATEGORIES:
                    crypto_rows += rows
                else:
                    market_rows += rows
                timeseries.append(df_plot)

        if market_rows > 0:
            return day.strftime("%Y-%m-%d"), timeseries
        if crypto_rows > 0 and crypto_only_candidate is None:
            crypto_only_candidate = (day.strftime("%Y-%m-%d"), timeseries)

    if crypto_only_candidate is not None:
        return crypto_only_candidate
    return requested_date, []


def _extract_price_series(frame: pd.DataFrame) -> pd.Series:
    if frame is None or frame.empty:
        return pd.Series(dtype="float64")

    for candidate in ["Close", "Adj Close", "close", "adjclose"]:
        if candidate in frame.columns:
            return pd.to_numeric(frame[candidate], errors="coerce").dropna()

    numeric_cols = frame.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return pd.Series(dtype="float64")
    return pd.to_numeric(frame[numeric_cols[0]], errors="coerce").dropna()


def _normalize_timestamp(value: Any) -> Optional[pd.Timestamp]:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None

    if pd.isna(timestamp):
        return None

    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert(None)
    return timestamp


def _build_summary_payload(
    price: float,
    reference: float,
    target_date: str,
    as_of: Optional[pd.Timestamp],
    source: str,
    quality: str,
    basis: str,
) -> Dict[str, Any]:
    target_day = _parse_date(target_date).date()
    as_of_date = as_of.date() if as_of is not None else target_day
    freshness_days = max((target_day - as_of_date).days, 0)
    change = price - reference
    pct = (change / reference) * 100 if reference else 0.0

    as_of_str = as_of.strftime("%Y-%m-%d %H:%M") if as_of is not None and basis == "intraday_session" else as_of_date.isoformat()
    summary_quality = quality
    if quality == "fresh" and freshness_days > SUMMARY_STALE_AFTER_DAYS:
        summary_quality = "stale"

    return {
        "Price": round(price, 4),
        "Change": round(change, 4),
        "Pct Change": f"{pct:.2f}%",
        "As Of": as_of_str,
        "Freshness Days": freshness_days,
        "Quality": summary_quality,
        "Source": source,
        "Basis": basis,
    }


def _build_daily_summary(frame: pd.DataFrame, target_date: str, source: str) -> Optional[Dict[str, Any]]:
    closes = _extract_price_series(frame)
    if closes.empty:
        return None

    price = float(closes.iloc[-1])
    reference = float(closes.iloc[-2]) if len(closes) >= 2 else price
    as_of = _normalize_timestamp(closes.index[-1])
    return _build_summary_payload(
        price=price,
        reference=reference,
        target_date=target_date,
        as_of=as_of,
        source=source,
        quality="fresh",
        basis="daily_close",
    )


def _build_intraday_summary_from_cache(
    intraday_cache: Optional[pd.DataFrame],
    target_date: str,
    source: str,
) -> Optional[Dict[str, Any]]:
    if intraday_cache is None or intraday_cache.empty:
        return None

    frame = intraday_cache.copy()
    frame["time"] = pd.to_datetime(frame.get("time"), errors="coerce")
    frame["price"] = pd.to_numeric(frame.get("price"), errors="coerce")
    frame = frame.dropna(subset=["time", "price"]).sort_values("time")
    if frame.empty:
        return None

    price = float(frame["price"].iloc[-1])
    reference = float(frame["price"].iloc[0]) if len(frame) >= 1 else price
    as_of = _normalize_timestamp(frame["time"].iloc[-1])
    return _build_summary_payload(
        price=price,
        reference=reference,
        target_date=target_date,
        as_of=as_of,
        source=source,
        quality="intraday_fallback",
        basis="intraday_session",
    )


def _build_summary_quality(summary_data: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    available = 0
    total = 0
    missing: List[str] = []
    stale: List[str] = []
    fallback: List[str] = []

    for category, items in summary_data.items():
        for name, value in items.items():
            total += 1
            label = f"{category} / {name}"
            if not isinstance(value, dict):
                missing.append(label)
                continue

            available += 1

            freshness_days = value.get("Freshness Days")
            try:
                freshness_days = int(freshness_days)
            except (TypeError, ValueError):
                freshness_days = None

            quality = str(value.get("Quality", "")).strip().lower()
            source = str(value.get("Source", "")).strip().lower()

            if freshness_days is not None and freshness_days > SUMMARY_STALE_AFTER_DAYS:
                stale.append(label)
            if quality == "intraday_fallback" or (source and not source.startswith("download_window")):
                fallback.append(label)

    ratio = round(available / total, 3) if total else 1.0
    return {
        "available": available,
        "total": total,
        "ratio": ratio,
        "missing": missing,
        "stale": stale,
        "fallback": fallback,
    }


def _calc_summary_for_symbol(
    ticker: str,
    target_date: str,
    intraday_cache: Optional[pd.DataFrame] = None,
) -> Optional[Dict[str, Any]]:
    day = _parse_date(target_date)
    start = day - timedelta(days=SUMMARY_LOOKBACK_DAYS)
    end = day + timedelta(days=1)

    daily, source = _safe_download_with_source(ticker, start=start, end=end, interval="1d")
    summary = _build_daily_summary(daily, target_date, source)
    if summary is not None:
        return summary

    summary = _build_intraday_summary_from_cache(intraday_cache, target_date, "intraday_cache")
    if summary is not None:
        return summary

    intraday, intraday_source = _safe_download_with_source(
        ticker,
        start=day,
        end=day + timedelta(days=1),
        interval=DEFAULT_INTRADAY_INTERVAL,
    )
    intraday_frame = _normalize_intraday_df(intraday, name=ticker, category="Fallback", ticker=ticker)
    return _build_intraday_summary_from_cache(intraday_frame, target_date, intraday_source)


def fetch_market_data(
    report_date: Optional[str] = None,
    intraday_interval: str = DEFAULT_INTRADAY_INTERVAL,
    intraday_fallback_days: int = 4,
) -> Dict[str, Any]:
    """Fetch summary market data and intraday chart series."""
    if report_date is None:
        report_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    print(f"[data] requested report_date={report_date}")
    print(f"[data] fetching intraday ({intraday_interval}) for charts...")

    effective_date, timeseries_data = _get_effective_intraday_date(
        requested_date=report_date,
        interval=intraday_interval,
        max_lookback_days=intraday_fallback_days,
    )

    if timeseries_data:
        print(f"[data] intraday effective_date={effective_date} series={len(timeseries_data)}")
        unique_symbols: List[str] = []
        for df in timeseries_data:
            symbol = df["symbol"].iloc[0] if "symbol" in df.columns and not df.empty else None
            if symbol and symbol not in unique_symbols:
                unique_symbols.append(symbol)
        print("[data] intraday symbols (first 30):", unique_symbols[:30])
    else:
        print(f"[data] intraday empty (effective_date tried back {intraday_fallback_days}d)")

    summary_date = effective_date if timeseries_data else report_date
    print(f"[data] fetching daily summary for {summary_date}...")

    intraday_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    for df in timeseries_data:
        if df.empty or "Category" not in df.columns or "name" not in df.columns:
            continue
        key = (str(df["Category"].iloc[0]), str(df["name"].iloc[0]))
        intraday_cache[key] = df.copy()

    summary_data: Dict[str, Dict[str, Any]] = {}
    for category, items in TICKERS.items():
        summary_data[category] = {}
        for name, ticker in items.items():
            try:
                result = _calc_summary_for_symbol(
                    ticker,
                    summary_date,
                    intraday_cache=intraday_cache.get((category, name)),
                )
                if result is None:
                    summary_data[category][name] = "No Data"
                else:
                    summary_data[category][name] = result
            except Exception as exc:
                print(f"[data] summary error: {name} ({ticker}) -> {type(exc).__name__}: {exc}")
                summary_data[category][name] = "Error"

    quality = _build_summary_quality(summary_data)
    print(
        "[data] summary coverage:",
        f"{quality['available']}/{quality['total']}",
        f"fallback={len(quality['fallback'])}",
        f"stale={len(quality['stale'])}",
        f"missing={len(quality['missing'])}",
    )

    return {
        "summary": summary_data,
        "timeseries": timeseries_data,
        "meta": {
            "requested_date": report_date,
            "effective_date": effective_date,
            "summary_date": summary_date,
            "intraday_interval": intraday_interval,
            "market_quality": quality,
        },
    }


def _news_cache_path(cache_dir: str, cache_key: str, max_per_feed: int) -> str:
    safe_key = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (cache_key or "news"))
    return os.path.join(cache_dir, f"headlines_{safe_key}_{max_per_feed}.json")


def _load_news_cache(cache_dir: str, cache_key: str, max_per_feed: int) -> Optional[List[str]]:
    if not cache_dir:
        return None
    path = _news_cache_path(cache_dir, cache_key, max_per_feed)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return [str(item) for item in payload]
    except Exception:
        return None
    return None


def _save_news_cache(cache_dir: str, cache_key: str, max_per_feed: int, headlines: List[str]) -> None:
    if not cache_dir:
        return
    os.makedirs(cache_dir, exist_ok=True)
    path = _news_cache_path(cache_dir, cache_key, max_per_feed)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(headlines, handle, ensure_ascii=False, indent=2)


def _fetch_rss_feed(url: str):
    response = requests.get(url, headers={"User-Agent": NEWS_USER_AGENT}, timeout=NEWS_REQUEST_TIMEOUT)
    response.raise_for_status()
    return feedparser.parse(response.content)


def fetch_news(max_per_feed: int = 5, cache_dir: str = "", cache_key: str = "") -> List[str]:
    """Fetch and deduplicate RSS headlines."""
    cached = _load_news_cache(cache_dir, cache_key, max_per_feed)
    if cached is not None:
        print(f"[news] using cached RSS headlines ({len(cached)})")
        return cached

    headlines: List[str] = []
    print("[news] fetching RSS headlines...")

    for url in RSS_FEEDS:
        try:
            feed = _fetch_rss_feed(url)
            for entry in feed.entries[:max_per_feed]:
                title = getattr(entry, "title", None)
                if title:
                    cleaned = normalize_news_text(title, strip_html_tags=True)
                    if cleaned:
                        headlines.append(f"- {cleaned}")
        except Exception as exc:
            print(f"[news] RSS error: {url} -> {type(exc).__name__}: {exc}")

    seen = set()
    deduped: List[str] = []
    for headline in headlines:
        if headline not in seen:
            seen.add(headline)
            deduped.append(headline)
    _save_news_cache(cache_dir, cache_key, max_per_feed, deduped)
    return deduped
