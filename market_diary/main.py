import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from modules.data_fetcher import fetch_market_data, fetch_news
from modules.llm_client import generate_report
from modules.chart_features import extract_chart_features, features_to_prompt_block


# -----------------------------
# Helpers
# -----------------------------
def _safe_first(df: pd.DataFrame, col: str, default=None):
    if df is None or df.empty or col not in df.columns:
        return default
    return df[col].iloc[0]


def _get_category(df: pd.DataFrame) -> Optional[str]:
    cat = _safe_first(df, "Category", None)
    if cat is None:
        cat = _safe_first(df, "category", None)
    return cat


def _ensure_datetime(s: pd.Series) -> pd.Series:
    try:
        if np.issubdtype(s.dtype, np.datetime64):
            return s
    except Exception:
        pass
    return pd.to_datetime(s, errors="coerce")


def _prep_timeseries(
    df: pd.DataFrame,
    tz: str = "Asia/Shanghai",
    time_col: str = "time",
    value_col: str = "price",
) -> pd.DataFrame:
    """Sort, parse time, convert to Beijing time (tz-naive), drop invalid rows."""
    if df is None or df.empty:
        return pd.DataFrame()

    if time_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    d = df.sort_values(time_col).copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    # Convert timezone
    if pd.api.types.is_datetime64tz_dtype(d[time_col]):
        d[time_col] = d[time_col].dt.tz_convert(tz).dt.tz_localize(None)
    else:
        # If tz-naive, try treat as UTC then convert to Beijing
        try:
            d[time_col] = d[time_col].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
        except Exception:
            pass

    d = d.dropna(subset=[time_col, value_col])
    return d


def _symbol_str(df: pd.DataFrame) -> str:
    return str(_safe_first(df, "symbol", "") or "")


def _match_df(
    timeseries_list: List[pd.DataFrame],
    keywords: List[str],
    category: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """
    Find first DF whose symbol contains ANY keyword (case-insensitive).
    If category is provided, require df Category matches (case-insensitive substring).
    """
    kws = [k.upper() for k in keywords]
    for df in timeseries_list:
        if df is None or df.empty:
            continue

        if category:
            cat = (_get_category(df) or "").upper()
            if category.upper() not in cat:
                continue

        sym = _symbol_str(df).upper()
        if any(k in sym for k in kws):
            return df
    return None


def _plot_time_axis(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.grid(True, linestyle="--", alpha=0.6)


def _save_fig(fig, save_path: str) -> str:
    plt.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)
    return os.path.basename(save_path)


# -----------------------------
# Plotting: existing charts
# -----------------------------
def plot_usd_trend(timeseries_list, report_date, save_path, tz="Asia/Shanghai"):
    """
    FX intraday chart using % change from open.
    Positive = USD strength.

    Input DF expected columns: time, price, symbol, Category
    """
    if not timeseries_list:
        return None

    fx_dfs = []
    for df in timeseries_list:
        if df is None or df.empty:
            continue
        cat = _get_category(df)
        if (cat or "").upper() == "FX":
            fx_dfs.append(df)

    if not fx_dfs:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(fx_dfs))))

    for i, df in enumerate(fx_dfs):
        d = _prep_timeseries(df, tz=tz)
        if d.empty:
            continue

        first_price = d["price"].iloc[0]
        if first_price == 0 or pd.isna(first_price):
            continue

        d["pct"] = (d["price"] / first_price - 1.0) * 100.0

        label = str(_safe_first(d, "symbol", f"FX_{i}") or f"FX_{i}")

        # If pair is XXX/USD and not USD/XXX, rising means USD weaker -> invert
        if "/USD" in label and not label.startswith("USD/"):
            d["pct"] = -d["pct"]
            label = f"{label} (inverted)"

        ax.plot(d["time"], d["pct"], label=label, color=colors[i % len(colors)], linewidth=2)

    ax.set_title(f"USD Strength vs Major FX ({report_date})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (Beijing)")
    ax.set_ylabel("Change from Open (%)  (Positive = USD stronger)")
    ax.axhline(0, linewidth=1, linestyle="-")
    ax.legend(loc="best")
    _plot_time_axis(ax)
    fig.autofmt_xdate()
    return _save_fig(fig, save_path)


def plot_multi_asset_trend(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    """
    Plot intraday performance (% from open) for Gold / Oil / Bitcoin.
    """
    if not timeseries_list:
        return None

    target_keywords = {
        "Gold": ["GOLD", "XAU"],
        "Oil": ["CRUDE", "WTI", "BRENT", "OIL", "CL1", "CL=F"],
        "Bitcoin": ["BTC", "BITCOIN"],
    }

    plot_data: Dict[str, pd.DataFrame] = {}

    for name, keywords in target_keywords.items():
        matched_df = _match_df(timeseries_list, keywords)
        if matched_df is None or matched_df.empty:
            continue

        d = _prep_timeseries(matched_df)
        if d.empty:
            continue

        first_price = d["price"].iloc[0]
        if first_price == 0 or pd.isna(first_price):
            continue

        d["pct_change"] = ((d["price"] - first_price) / first_price) * 100
        plot_data[name] = d[["time", "pct_change"]]

    if not plot_data:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    for name, d in plot_data.items():
        ax.plot(d["time"], d["pct_change"], label=name, linewidth=2.5)

    ax.set_title(
        f"Intraday Performance: Gold vs Oil vs Bitcoin ({report_date})\n"
        f"(Base = Open Price, Unit: % Change)",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time (Beijing)")
    ax.set_ylabel("Change from Open (%)")
    ax.axhline(0, linewidth=1, linestyle="-")
    ax.legend(loc="best")
    _plot_time_axis(ax)
    fig.autofmt_xdate()
    return _save_fig(fig, save_path)


# -----------------------------
# Plotting: NEW charts (rates / equities / vol / credit / curve / oil curve)
# -----------------------------
KEYMAP = {
    # Rates
    "UST2Y": ["US2Y", "UST2", "2Y"],
    "UST10Y": ["US10Y", "UST10", "10Y"],
    "UST30Y": ["US30Y", "UST30", "30Y"],
    "REAL10Y": ["REAL10", "US10Y REAL", "TIPS10", "10Y REAL"],
    "BE10Y": ["BE10", "BREAKEVEN10", "10Y BE", "10Y BREAKEVEN"],

    # Equities (indices)
    "SPX": ["SPX", "S&P", "SP500", "^GSPC"],
    "NDX": ["NDX", "NASDAQ100", "NASDAQ 100", "^NDX"],
    "STOXX": ["STOXX", "SX5E", "EURO STOXX"],
    "HSI": ["HSI", "HANG SENG", "^HSI"],
    "CSI300": ["CSI300", "000300", "SH000300"],

    # Vol
    "VIX": ["VIX"],
    "MOVE": ["MOVE"],

    # Credit (spreads)
    "IG": ["IG", "CDX IG", "LQD", "US IG"],
    "HY": ["HY", "CDX HY", "HYG", "US HY"],

    # Oil curve (front/back)
    "WTI_FRONT": ["CL1", "WTI1", "WTI FRONT", "WTI 1M", "CL=F"],
    "WTI_BACK": ["CL4", "WTI6", "WTI BACK", "WTI 6M"],
}


def _plot_change_from_open(
    series: List[Tuple[str, pd.DataFrame]],
    report_date: str,
    save_path: str,
    unit: str = "%",
    title: str = "",
    ylabel: str = "",
    value_transform=None,
) -> Optional[str]:
    """
    Plot multiple series, change from open.
    unit:
      - "%" uses % change
      - "bps" uses (price - open)*100 bps assuming price is yield in %
      - "abs" uses absolute change (price - open)
    """
    if not series:
        return None

    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = 0

    for name, df in series:
        d = _prep_timeseries(df)
        if d.empty:
            continue

        open_v = d["price"].iloc[0]
        if pd.isna(open_v):
            continue

        if unit == "%":
            if open_v == 0:
                continue
            y = (d["price"] / open_v - 1.0) * 100.0
            ax.plot(d["time"], y, label=name, linewidth=2.2)
        elif unit == "bps":
            # Yield in % -> delta% * 100 = bps
            y = (d["price"] - open_v) * 100.0
            ax.plot(d["time"], y, label=name, linewidth=2.2)
        else:  # abs
            y = (d["price"] - open_v)
            ax.plot(d["time"], y, label=name, linewidth=2.2)

        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return None

    ax.set_title(title or f"Change from Open ({report_date})", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (Beijing)")
    ax.set_ylabel(ylabel or ("Change from Open" + (f" ({unit})" if unit else "")))
    ax.axhline(0, linewidth=1, linestyle="-")
    ax.legend(loc="best")
    _plot_time_axis(ax)
    fig.autofmt_xdate()
    return _save_fig(fig, save_path)


def plot_rates_2y10y30y(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    d2 = _match_df(timeseries_list, KEYMAP["UST2Y"])
    d10 = _match_df(timeseries_list, KEYMAP["UST10Y"])
    d30 = _match_df(timeseries_list, KEYMAP["UST30Y"])
    series = []
    if d2 is not None: series.append(("UST 2Y", d2))
    if d10 is not None: series.append(("UST 10Y", d10))
    if d30 is not None: series.append(("UST 30Y", d30))

    return _plot_change_from_open(
        series,
        report_date,
        save_path,
        unit="bps",
        title=f"Rates: UST 2Y / 10Y / 30Y (Intraday, bps from open) — {report_date}",
        ylabel="bps from open",
    )


def plot_curve_2s10s(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    """
    Prefer direct 2s10s series if present; else compute from 10Y - 2Y.
    """
    direct = _match_df(timeseries_list, ["2S10S", "2-10", "2Y10Y", "2S/10S", "2S_10S", "2s10s"])
    if direct is not None:
        return _plot_change_from_open(
            [("2s10s", direct)],
            report_date,
            save_path,
            unit="bps",
            title=f"Curve Slope: 2s10s (Intraday, bps from open) — {report_date}",
            ylabel="bps from open",
        )

    d2 = _match_df(timeseries_list, KEYMAP["UST2Y"])
    d10 = _match_df(timeseries_list, KEYMAP["UST10Y"])
    if d2 is None or d10 is None:
        return None

    a = _prep_timeseries(d2)
    b = _prep_timeseries(d10)
    if a.empty or b.empty:
        return None

    # Merge on nearest timestamps (asof) to compute slope
    a = a[["time", "price"]].rename(columns={"price": "y2"}).sort_values("time")
    b = b[["time", "price"]].rename(columns={"price": "y10"}).sort_values("time")
    merged = pd.merge_asof(a, b, on="time", direction="nearest", tolerance=pd.Timedelta("3min"))
    merged = merged.dropna(subset=["y2", "y10"])
    if merged.empty:
        return None

    merged["slope_bps"] = (merged["y10"] - merged["y2"]) * 100.0
    open_slope = merged["slope_bps"].iloc[0]
    merged["chg_bps"] = merged["slope_bps"] - open_slope

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(merged["time"], merged["chg_bps"], label="2s10s (10Y-2Y)", linewidth=2.2)
    ax.set_title(f"Curve Slope: 2s10s (Computed, bps from open) — {report_date}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (Beijing)")
    ax.set_ylabel("bps from open")
    ax.axhline(0, linewidth=1, linestyle="-")
    ax.legend(loc="best")
    _plot_time_axis(ax)
    fig.autofmt_xdate()
    return _save_fig(fig, save_path)


def plot_real_vs_breakeven(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    real = _match_df(timeseries_list, KEYMAP["REAL10Y"])
    be = _match_df(timeseries_list, KEYMAP["BE10Y"])
    if real is None or be is None:
        return None

    r = _prep_timeseries(real)
    b = _prep_timeseries(be)
    if r.empty or b.empty:
        return None

    # Plot absolute levels (not change) because interpretation is level-based
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(r["time"], r["price"], label="10Y Real Yield", linewidth=2.2)
    ax.plot(b["time"], b["price"], label="10Y Breakeven", linewidth=2.2)
    ax.set_title(f"Inflation Decomposition: 10Y Real Yield vs 10Y Breakeven — {report_date}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (Beijing)")
    ax.set_ylabel("Level")
    ax.legend(loc="best")
    _plot_time_axis(ax)
    fig.autofmt_xdate()
    return _save_fig(fig, save_path)


def plot_equity_global(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    picks = []
    for label, key in [("US: SPX", "SPX"), ("US: NDX", "NDX"), ("EU: STOXX", "STOXX"), ("CN/HK: HSI", "HSI"), ("CN: CSI300", "CSI300")]:
        df = _match_df(timeseries_list, KEYMAP[key])
        if df is not None:
            picks.append((label, df))

    return _plot_change_from_open(
        picks,
        report_date,
        save_path,
        unit="%",
        title=f"Equities: US / Europe / China (Intraday, % from open) — {report_date}",
        ylabel="% from open",
    )


def plot_vol_vix_move(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    vix = _match_df(timeseries_list, KEYMAP["VIX"])
    move = _match_df(timeseries_list, KEYMAP["MOVE"])
    series = []
    if vix is not None: series.append(("VIX", vix))
    if move is not None: series.append(("MOVE", move))

    # Use % from open (not bps)
    return _plot_change_from_open(
        series,
        report_date,
        save_path,
        unit="%",
        title=f"Vol: VIX vs MOVE (Intraday, % from open) — {report_date}",
        ylabel="% from open",
    )


def plot_credit_ig_hy(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    ig = _match_df(timeseries_list, KEYMAP["IG"], category=None)
    hy = _match_df(timeseries_list, KEYMAP["HY"], category=None)
    series = []
    if ig is not None: series.append(("Credit IG", ig))
    if hy is not None: series.append(("Credit HY", hy))

    # Spreads are better as absolute change, but we only have "price".
    # We'll plot absolute change from open.
    if not series:
        return None

    return _plot_change_from_open(
        series,
        report_date,
        save_path,
        unit="abs",
        title=f"Credit: IG vs HY (Intraday, abs change from open) — {report_date}",
        ylabel="abs change from open",
    )


def plot_wti_curve_front_back(timeseries_list: List[pd.DataFrame], report_date: str, save_path: str) -> Optional[str]:
    front = _match_df(timeseries_list, KEYMAP["WTI_FRONT"])
    back = _match_df(timeseries_list, KEYMAP["WTI_BACK"])
    if front is None or back is None:
        return None

    f = _prep_timeseries(front)
    b = _prep_timeseries(back)
    if f.empty or b.empty:
        return None

    f = f[["time", "price"]].rename(columns={"price": "front"}).sort_values("time")
    b = b[["time", "price"]].rename(columns={"price": "back"}).sort_values("time")
    merged = pd.merge_asof(f, b, on="time", direction="nearest", tolerance=pd.Timedelta("5min"))
    merged = merged.dropna(subset=["front", "back"])
    if merged.empty:
        return None

    merged["spread"] = merged["front"] - merged["back"]
    open_spread = merged["spread"].iloc[0]
    merged["chg"] = merged["spread"] - open_spread

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(merged["time"], merged["chg"], label="Front-Back (Front minus Back)", linewidth=2.2)
    ax.set_title(f"Oil Curve: WTI Front–Back Spread (Change from open) — {report_date}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (Beijing)")
    ax.set_ylabel("Spread change from open")
    ax.axhline(0, linewidth=1, linestyle="-")
    ax.legend(loc="best")
    _plot_time_axis(ax)
    fig.autofmt_xdate()
    return _save_fig(fig, save_path)


# -----------------------------
# Charts builder
# -----------------------------
def create_charts(report_date: str, market_data_dict: dict, output_dir: str) -> str:
    """Create chart images and return the Markdown section embedding them."""
    chart_dir = os.path.join(output_dir, "charts")
    os.makedirs(chart_dir, exist_ok=True)

    timeseries_list = market_data_dict.get("timeseries", [])
    if not timeseries_list:
        return "\n*(No intraday data available for charts)*\n"

    charts_md = "## 📊 Charts\n\n"

    # 1) USD / Multi assets
    fx_img = plot_usd_trend(timeseries_list, report_date, os.path.join(chart_dir, f"fx_{report_date}.png"))
    if fx_img:
        charts_md += f"### 💵 USD Strength (FX, Intraday %)\n![FX Chart](charts/{fx_img})\n\n"

    multi_img = plot_multi_asset_trend(timeseries_list, report_date, os.path.join(chart_dir, f"multi_{report_date}.png"))
    if multi_img:
        charts_md += f"### 🟡🛢️₿ Gold vs Oil vs Bitcoin (Intraday %)\n![Multi Asset](charts/{multi_img})\n\n"

    # 2) Rates pack
    rates_img = plot_rates_2y10y30y(timeseries_list, report_date, os.path.join(chart_dir, f"rates_{report_date}.png"))
    if rates_img:
        charts_md += f"### 🏦 Rates: UST 2Y/10Y/30Y (bps from open)\n![Rates](charts/{rates_img})\n\n"

    curve_img = plot_curve_2s10s(timeseries_list, report_date, os.path.join(chart_dir, f"curve_2s10s_{report_date}.png"))
    if curve_img:
        charts_md += f"### 📈 Curve: 2s10s (bps from open)\n![2s10s](charts/{curve_img})\n\n"

    realbe_img = plot_real_vs_breakeven(timeseries_list, report_date, os.path.join(chart_dir, f"real_be_{report_date}.png"))
    if realbe_img:
        charts_md += f"### 🧊🔥 Real Yield vs Breakeven (levels)\n![Real vs BE](charts/{realbe_img})\n\n"

    # 3) Risk assets global
    eq_img = plot_equity_global(timeseries_list, report_date, os.path.join(chart_dir, f"equities_{report_date}.png"))
    if eq_img:
        charts_md += f"### 📉 Equities: US/EU/CN (Intraday %)\n![Equities](charts/{eq_img})\n\n"

    # 4) Vol & Credit
    vol_img = plot_vol_vix_move(timeseries_list, report_date, os.path.join(chart_dir, f"vol_{report_date}.png"))
    if vol_img:
        charts_md += f"### 🌪️ Vol: VIX vs MOVE (Intraday %)\n![Vol](charts/{vol_img})\n\n"

    credit_img = plot_credit_ig_hy(timeseries_list, report_date, os.path.join(chart_dir, f"credit_{report_date}.png"))
    if credit_img:
        charts_md += f"### 🧱 Credit: IG vs HY (abs change)\n![Credit](charts/{credit_img})\n\n"

    # 5) Commodity curve (optional)
    oilcurve_img = plot_wti_curve_front_back(timeseries_list, report_date, os.path.join(chart_dir, f"wti_curve_{report_date}.png"))
    if oilcurve_img:
        charts_md += f"### 🛢️ Oil Curve: WTI Front–Back (change from open)\n![WTI Curve](charts/{oilcurve_img})\n\n"

    # If nothing got plotted:
    if charts_md.strip() == "## 📊 Charts":
        return "\n*(Charts unavailable: no matching series found)*\n"

    return charts_md


# -----------------------------
# Main
# -----------------------------
def _parse_args() -> argparse.Namespace:
    default_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    parser = argparse.ArgumentParser(description="Generate Daily Market Diary with Charts")
    parser.add_argument(
        "--date",
        type=str,
        default=default_date,
        help="Report date (YYYY-MM-DD). Defaults to yesterday.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report_date = args.date

    print(f"🚀 Generating Market Diary for {report_date}...")

    print("📡 Fetching market data and news...")
    raw_data = fetch_market_data(report_date)
    news_headlines = fetch_news()

    if not raw_data:
        print("⚠️ Market data fetch failed; continuing with empty payload.")
        raw_data = {"summary": {}, "timeseries": [], "meta": {"effective_date": report_date}}

    meta = raw_data.get("meta", {}) or {}
    effective_date = meta.get("effective_date", report_date)
    if effective_date != report_date:
        print(f"⚠️ No intraday on {report_date}; fallback to trading day {effective_date}")
        report_date = effective_date

    market_summary = raw_data.get("summary", {}) or {}
    timeseries_data = raw_data.get("timeseries", []) or []

    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    print("📊 Generating charts...")
    if timeseries_data:
        try:
            charts_section = create_charts(report_date, raw_data, output_dir=output_dir)
        except Exception as e:
            print(f"❌ Chart generation failed: {type(e).__name__}: {e}")
            charts_section = "\n*(Charts generation failed due to an error)*\n"
    else:
        charts_section = "\n*(No intraday data available to generate charts)*\n"

    # ---- Extract chart features for LLM ----
    print("🔬 Extracting chart features...")
    try:
        chart_features = extract_chart_features(timeseries_data, tz="Asia/Shanghai")
        chart_dir = os.path.join(output_dir, "charts")
        os.makedirs(chart_dir, exist_ok=True)
        feat_path = os.path.join(chart_dir, f"features_{report_date}.json")
        with open(feat_path, "w", encoding="utf-8") as _f:
            json.dump(chart_features, _f, ensure_ascii=False, indent=2, default=str)
        print(f"💾 Chart features saved: {feat_path}")
        chart_features_block = features_to_prompt_block(chart_features)
    except Exception as e:
        print(f"⚠️ Chart feature extraction failed: {type(e).__name__}: {e}")
        chart_features_block = "[Chart features extraction failed — LLM should acknowledge missing data]"

    print("🧠 Generating AI analysis...")
    report_content = generate_report(report_date, market_summary, news_headlines, chart_features_block)
    if isinstance(report_content, str) and report_content.startswith("Error"):
        print(f"❌ LLM analysis failed: {report_content}")
        report_content = f"*AI Analysis failed: {report_content}*"

    # ✅ IMPORTANT: charts moved to the END of the report
    final_report = f"""# 📅 Market Diary: {report_date}

---

## 🧠 AI Macro Analysis

{report_content}

---

{charts_section}

---

*Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""

    output_file = os.path.join(output_dir, f"{report_date}.md")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(final_report)
        print(f"✅ Report saved: {output_file}")
    except Exception as e:
        print(f"❌ Failed to save report: {type(e).__name__}: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
