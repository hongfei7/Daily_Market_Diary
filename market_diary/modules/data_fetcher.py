import yfinance as yf
import feedparser
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any

# -----------------------------
# Config
# -----------------------------
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
    },
    "Commodities": {
        "Crude Oil": "CL=F",
        # 这个 back-month 在 Yahoo/yfinance 有时会取不到：取不到会自动跳过
        "WTI 6M": "CL4=F",
        "Gold": "GC=F",
        "Copper": "HG=F",
    },
    "Equities": {
        "S&P 500": "^GSPC",
        "Nasdaq 100": "^NDX",
        "Euro Stoxx 50": "^STOXX50E",
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

# 画图时用的“稳定别名”（让你的 KEYMAP 永远命中）
PLOT_ALIASES = {
    ("Rates", "13W T-Bill"): "UST3M",
    ("Rates", "5Y Treasury"): "UST5Y",
    ("Rates", "10Y Treasury"): "UST10Y",
    ("Rates", "30Y Treasury"): "UST30Y",

    ("FX", "DXY"): "DXY",
    ("FX", "USD/JPY"): "USD/JPY",
    ("FX", "EUR/USD"): "EUR/USD",
    ("FX", "USD/CNH"): "USD/CNH",

    ("Commodities", "Crude Oil"): "WTI",
    ("Commodities", "WTI 6M"): "WTI_6M",
    ("Commodities", "Gold"): "GOLD",
    ("Commodities", "Copper"): "COPPER",

    ("Equities", "S&P 500"): "SPX",
    ("Equities", "Nasdaq 100"): "NDX",
    ("Equities", "Euro Stoxx 50"): "STOXX50E",
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

# intraday 图表白名单（会显著增加请求量；如果你遇到 rate-limit，可以删减）
INTRADAY_WHITELIST = {
    # FX
    ("FX", "DXY"),
    ("FX", "USD/JPY"),
    ("FX", "EUR/USD"),
    ("FX", "USD/CNH"),

    # Commodities + Crypto
    ("Commodities", "Crude Oil"),
    ("Commodities", "WTI 6M"),
    ("Commodities", "Gold"),
    ("Crypto", "Bitcoin"),

    # Rates
    ("Rates", "13W T-Bill"),
    ("Rates", "5Y Treasury"),
    ("Rates", "10Y Treasury"),
    ("Rates", "30Y Treasury"),

    # Equities
    ("Equities", "S&P 500"),
    ("Equities", "Nasdaq 100"),
    ("Equities", "Euro Stoxx 50"),
    ("Equities", "China Large-Cap (FXI)"),

    # Vol + Credit
    ("Vol", "VIX"),
    ("Vol", "MOVE"),
    ("Credit", "IG (LQD)"),
    ("Credit", "HY (HYG)"),
}

DEFAULT_INTRADAY_INTERVAL = "5m"


# -----------------------------
# Helpers
# -----------------------------
def _parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def _safe_download(symbol: str, start: datetime, end: datetime, interval: str) -> pd.DataFrame:
    df = yf.download(
        symbol,
        start=start,
        end=end,
        interval=interval,
        progress=False,
        auto_adjust=False,
        threads=False,
    )
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def _normalize_intraday_df(hist: pd.DataFrame, name: str, category: str, ticker: str) -> pd.DataFrame:
    """
    Output columns:
      time | price | symbol(稳定别名) | name(可读) | ticker(yfinance) | Category
    """
    if hist is None or hist.empty:
        return pd.DataFrame(columns=["time", "symbol", "name", "ticker", "price", "Category"])

    df = hist.reset_index()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if c[0] else c[1] for c in df.columns]

    if "Datetime" in df.columns:
        time_col = "Datetime"
    elif "Date" in df.columns:
        time_col = "Date"
    else:
        time_col = df.columns[0]

    price_col = None
    for cand in ["Close", "close", "Adj Close", "adjclose"]:
        if cand in df.columns:
            price_col = cand
            break
    if price_col is None:
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            return pd.DataFrame(columns=["time", "symbol", "name", "ticker", "price", "Category"])
        price_col = numeric_cols[0]

    alias = PLOT_ALIASES.get((category, name), name)

    out = pd.DataFrame({
        "time": pd.to_datetime(df[time_col], errors="coerce"),
        "price": pd.to_numeric(df[price_col], errors="coerce"),
        "symbol": alias,     # ✅ 给画图/KEYMAP 用
        "name": name,        # ✅ 给人读
        "ticker": ticker,    # ✅ debug 用
        "Category": category,
    }).dropna(subset=["time", "price"])

    return out


def _get_effective_intraday_date(requested_date: str, interval: str, max_lookback_days: int = 4) -> Tuple[str, List[pd.DataFrame]]:
    base = _parse_date(requested_date)
    for k in range(max_lookback_days + 1):
        day = base - timedelta(days=k)
        start = day
        end = day + timedelta(days=1)

        timeseries: List[pd.DataFrame] = []
        any_rows = 0

        for category, items in TICKERS.items():
            for name, ticker in items.items():
                if (category, name) not in INTRADAY_WHITELIST:
                    continue

                hist = _safe_download(ticker, start=start, end=end, interval=interval)
                df_plot = _normalize_intraday_df(hist, name=name, category=category, ticker=ticker)
                if not df_plot.empty:
                    any_rows += len(df_plot)
                    timeseries.append(df_plot)

        if any_rows > 0:
            return day.strftime("%Y-%m-%d"), timeseries

    return requested_date, []


def _calc_summary_for_symbol(ticker: str, target_date: str) -> Optional[Tuple[float, float, str]]:
    day = _parse_date(target_date)
    start = day - timedelta(days=10)
    end = day + timedelta(days=1)

    daily = _safe_download(ticker, start=start, end=end, interval="1d")
    if daily is None or daily.empty or "Close" not in daily.columns:
        return None

    closes = daily["Close"].dropna()
    if len(closes) == 0:
        return None

    price = float(closes.iloc[-1])
    prev = float(closes.iloc[-2]) if len(closes) >= 2 else price

    change = price - prev
    pct = (change / prev) * 100 if prev != 0 else 0.0
    return round(price, 4), round(change, 4), f"{pct:.2f}%"


# -----------------------------
# Public APIs
# -----------------------------
def fetch_market_data(
    report_date: Optional[str] = None,
    intraday_interval: str = DEFAULT_INTRADAY_INTERVAL,
    intraday_fallback_days: int = 4,
) -> Dict[str, Any]:
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
        # debug：打印前 30 个 symbol
        uniq = []
        for df in timeseries_data:
            s = df["symbol"].iloc[0] if "symbol" in df.columns and not df.empty else None
            if s and s not in uniq:
                uniq.append(s)
        print("[data] intraday symbols (first 30):", uniq[:30])
    else:
        print(f"[data] intraday empty (effective_date tried back {intraday_fallback_days}d)")

    summary_date = effective_date if timeseries_data else report_date
    print(f"[data] fetching daily summary for {summary_date}...")

    summary_data: Dict[str, Dict[str, Any]] = {}
    for category, items in TICKERS.items():
        summary_data[category] = {}
        for name, ticker in items.items():
            try:
                out = _calc_summary_for_symbol(ticker, summary_date)
                if out is None:
                    summary_data[category][name] = "No Data"
                else:
                    price, change, pct = out
                    summary_data[category][name] = {"Price": price, "Change": change, "Pct Change": pct}
            except Exception as e:
                print(f"[data] summary error: {name} ({ticker}) -> {type(e).__name__}: {e}")
                summary_data[category][name] = "Error"

    return {
        "summary": summary_data,
        "timeseries": timeseries_data,
        "meta": {
            "requested_date": report_date,
            "effective_date": effective_date,
            "summary_date": summary_date,
            "intraday_interval": intraday_interval,
        },
    }


def fetch_news(max_per_feed: int = 5) -> List[str]:
    headlines: List[str] = []
    print("[news] fetching RSS headlines...")

    for url in RSS_FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                title = getattr(entry, "title", None)
                if title:
                    headlines.append(f"- {title}")
        except Exception as e:
            print(f"[news] RSS error: {url} -> {type(e).__name__}: {e}")

    seen = set()
    deduped = []
    for h in headlines:
        if h not in seen:
            seen.add(h)
            deduped.append(h)
    return deduped
