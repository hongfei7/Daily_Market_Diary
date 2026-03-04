"""
chart_features.py — Extract LLM-readable features from intraday timeseries.

Public API:
    extract_chart_features(timeseries_list, tz="Asia/Shanghai") -> Dict
    features_to_prompt_block(features) -> str
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Internal helpers (lightweight replicas; no import from main.py)
# ---------------------------------------------------------------------------

def _prep_ts(
    df: pd.DataFrame,
    tz: str = "Asia/Shanghai",
    time_col: str = "time",
    value_col: str = "price",
) -> pd.DataFrame:
    """Sort, parse time, convert to tz-naive Beijing time, drop bad rows."""
    if df is None or df.empty:
        return pd.DataFrame()
    if time_col not in df.columns or value_col not in df.columns:
        return pd.DataFrame()

    d = df.sort_values(time_col).copy()
    d[time_col] = pd.to_datetime(d[time_col], errors="coerce")

    if pd.api.types.is_datetime64tz_dtype(d[time_col]):
        d[time_col] = d[time_col].dt.tz_convert(tz).dt.tz_localize(None)
    else:
        try:
            d[time_col] = (
                d[time_col].dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
            )
        except Exception:
            pass

    d = d.dropna(subset=[time_col, value_col])
    return d


def _safe_first(df: pd.DataFrame, col: str, default=None):
    if df is None or df.empty or col not in df.columns:
        return default
    return df[col].iloc[0]


def _get_category(df: pd.DataFrame) -> Optional[str]:
    cat = _safe_first(df, "Category", None)
    if cat is None:
        cat = _safe_first(df, "category", None)
    return cat


def _symbol_str(df: pd.DataFrame) -> str:
    return str(_safe_first(df, "symbol", "") or "")


def _match_df(
    timeseries_list: List[pd.DataFrame],
    keywords: List[str],
    category: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Find first DF whose symbol contains ANY keyword (case-insensitive)."""
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


# ---------------------------------------------------------------------------
# Turning-point detection
# ---------------------------------------------------------------------------

def _turning_points(
    times: pd.Series,
    values: pd.Series,
    n_max: int = 3,
    min_gap_bars: int = 3,
) -> List[Dict]:
    """
    Return up to n_max turning points (slope sign changes).
    Each entry: {"time": str, "value": float, "direction": "peak"|"trough"}
    """
    if len(values) < 5:
        return []

    arr = np.array(values, dtype=float)
    # Remove NaNs for diff
    mask = ~np.isnan(arr)
    if mask.sum() < 5:
        return []

    deltas = np.diff(arr)
    signs = np.sign(deltas)

    turning = []
    last_idx = -min_gap_bars

    for i in range(1, len(signs)):
        if (signs[i - 1] > 0 and signs[i] < 0) or (signs[i - 1] < 0 and signs[i] > 0):
            bar_idx = i  # the bar where slope flips
            if bar_idx - last_idx < min_gap_bars:
                continue
            last_idx = bar_idx
            direction = "peak" if signs[i - 1] > 0 else "trough"
            t = times.iloc[bar_idx] if bar_idx < len(times) else None
            v = float(arr[bar_idx]) if not math.isnan(float(arr[bar_idx])) else None
            if t is not None and v is not None:
                turning.append({
                    "time": str(t)[:16],
                    "value": round(v, 4),
                    "direction": direction,
                })
            if len(turning) >= n_max:
                break

    return turning


# ---------------------------------------------------------------------------
# Series stats
# ---------------------------------------------------------------------------

def _series_stats(
    d: pd.DataFrame,
    pct_col: str = "pct",
    time_col: str = "time",
    n_turning: int = 3,
) -> Dict:
    """
    Given a prepped DF with a pct_col (% from open), compute summary stats.
    """
    if d.empty or pct_col not in d.columns:
        return {"available": False}

    pct = d[pct_col].dropna()
    if pct.empty:
        return {"available": False}

    net = float(pct.iloc[-1])
    rng = float(pct.max() - pct.min())
    hi_idx = pct.idxmax()
    lo_idx = pct.idxmin()

    hi_time = str(d.loc[hi_idx, time_col])[:16] if hi_idx in d.index else "?"
    lo_time = str(d.loc[lo_idx, time_col])[:16] if lo_idx in d.index else "?"
    hi_val = float(pct.max())
    lo_val = float(pct.min())

    tps = _turning_points(d[time_col].reset_index(drop=True), pct.reset_index(drop=True), n_max=n_turning)

    return {
        "available": True,
        "net_pp": round(net, 3),
        "range_pp": round(rng, 3),
        "high": {"time": hi_time, "value": round(hi_val, 3)},
        "low": {"time": lo_time, "value": round(lo_val, 3)},
        "turning_points": tps,
    }


# ---------------------------------------------------------------------------
# FX / USD strength extraction
# ---------------------------------------------------------------------------

def _extract_fx_features(timeseries_list: List[pd.DataFrame], tz: str) -> Dict:
    """
    For each FX pair (Category=FX), compute % from open (inverted for XXX/USD
    so positive = USD stronger). Then build a 5-min composite.
    """
    fx_dfs = [df for df in timeseries_list if (_get_category(df) or "").upper() == "FX"]

    pair_results = []
    composite_series: List[pd.Series] = []

    for df in fx_dfs:
        d = _prep_ts(df, tz=tz)
        if d.empty:
            continue

        sym = _symbol_str(df)
        first_price = d["price"].iloc[0]
        if first_price == 0 or pd.isna(first_price):
            continue

        d["pct"] = (d["price"] / first_price - 1.0) * 100.0

        # Invert XXX/USD (e.g. EUR/USD) so positive = USD stronger
        inverted = False
        if "/USD" in sym.upper() and not sym.upper().startswith("USD/"):
            d["pct"] = -d["pct"]
            inverted = True

        stats = _series_stats(d)
        stats["symbol"] = sym
        stats["inverted"] = inverted
        pair_results.append(stats)

        # Collect pct series for composite (resampled to 5-min)
        ts_pct = d.set_index("time")["pct"]
        ts_pct.index = pd.to_datetime(ts_pct.index)
        ts_5m = ts_pct.resample("5min").last().ffill()
        composite_series.append(ts_5m)

    # Composite: mean across FX pairs at each 5-min bar
    composite_result: Dict = {"available": False}
    if composite_series:
        try:
            comp_df = pd.concat(composite_series, axis=1)
            comp_mean = comp_df.mean(axis=1).dropna()
            if not comp_mean.empty:
                comp_frame = comp_mean.reset_index()
                comp_frame.columns = ["time", "pct"]
                composite_result = _series_stats(comp_frame)
        except Exception:
            pass

    return {
        "pairs": pair_results,
        "composite": composite_result,
    }


# ---------------------------------------------------------------------------
# Gold / Oil / Bitcoin extraction
# ---------------------------------------------------------------------------

_ASSET_KEYWORDS = {
    "Gold": ["GOLD", "XAU"],
    "Oil": ["CRUDE", "WTI", "BRENT", "OIL", "CL1", "CL=F"],
    "Bitcoin": ["BTC", "BITCOIN"],
}


def _extract_asset_features(timeseries_list: List[pd.DataFrame], tz: str) -> Dict:
    """
    For Gold, Oil, Bitcoin: compute % from open stats.
    Add divergence summary and pairwise 5-min correlations.
    """
    asset_stats: Dict[str, Dict] = {}
    asset_series_5m: Dict[str, pd.Series] = {}

    for asset_name, keywords in _ASSET_KEYWORDS.items():
        df = _match_df(timeseries_list, keywords)
        if df is None or df.empty:
            asset_stats[asset_name] = {"available": False}
            continue

        d = _prep_ts(df, tz=tz)
        if d.empty:
            asset_stats[asset_name] = {"available": False}
            continue

        first_price = d["price"].iloc[0]
        if first_price == 0 or pd.isna(first_price):
            asset_stats[asset_name] = {"available": False}
            continue

        d["pct"] = (d["price"] / first_price - 1.0) * 100.0
        stats = _series_stats(d)
        stats["symbol"] = _symbol_str(df)
        asset_stats[asset_name] = stats

        ts_pct = d.set_index("time")["pct"]
        ts_pct.index = pd.to_datetime(ts_pct.index)
        ts_5m = ts_pct.resample("5min").last().ffill()
        asset_series_5m[asset_name] = ts_5m

    # Divergence
    available_nets = {
        name: stats["net_pp"]
        for name, stats in asset_stats.items()
        if stats.get("available")
    }
    divergence: Dict = {}
    if available_nets:
        best = max(available_nets, key=available_nets.get)
        worst = min(available_nets, key=available_nets.get)
        divergence = {
            "best_asset": best,
            "best_net_pp": round(available_nets[best], 3),
            "worst_asset": worst,
            "worst_net_pp": round(available_nets[worst], 3),
            "spread_pp": round(available_nets[best] - available_nets[worst], 3),
        }

    # Pairwise 5-min correlations
    corr_pairs = [("Gold", "Oil"), ("Gold", "Bitcoin"), ("Oil", "Bitcoin")]
    correlations: Dict = {}
    for a, b in corr_pairs:
        key = f"{a}_{b}"
        if a in asset_series_5m and b in asset_series_5m:
            try:
                merged = pd.concat([asset_series_5m[a], asset_series_5m[b]], axis=1).dropna()
                if len(merged) >= 5:
                    c = float(merged.iloc[:, 0].corr(merged.iloc[:, 1]))
                    correlations[key] = round(c, 3) if not math.isnan(c) else None
                else:
                    correlations[key] = None
            except Exception:
                correlations[key] = None
        else:
            correlations[key] = None

    return {
        "assets": asset_stats,
        "divergence": divergence,
        "correlations": correlations,
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_chart_features(
    timeseries_list: List[pd.DataFrame],
    tz: str = "Asia/Shanghai",
) -> Dict:
    """
    Extract LLM-readable chart features from intraday timeseries data.

    Returns a dict with:
        fx_pairs    : list of per-pair stats dicts
        fx_composite: composite USD strength stats dict
        assets      : dict of Gold/Oil/Bitcoin stats
        divergence  : best/worst asset + spread
        correlations: pairwise 5-min correlations
    """
    if not timeseries_list:
        return {
            "fx_pairs": [],
            "fx_composite": {"available": False},
            "assets": {name: {"available": False} for name in _ASSET_KEYWORDS},
            "divergence": {},
            "correlations": {},
            "error": "No timeseries data provided",
        }

    try:
        fx = _extract_fx_features(timeseries_list, tz)
    except Exception as e:
        fx = {"pairs": [], "composite": {"available": False}, "error": str(e)}

    try:
        assets_data = _extract_asset_features(timeseries_list, tz)
    except Exception as e:
        assets_data = {
            "assets": {name: {"available": False} for name in _ASSET_KEYWORDS},
            "divergence": {},
            "correlations": {},
            "error": str(e),
        }

    return {
        "fx_pairs": fx.get("pairs", []),
        "fx_composite": fx.get("composite", {"available": False}),
        "assets": assets_data.get("assets", {}),
        "divergence": assets_data.get("divergence", {}),
        "correlations": assets_data.get("correlations", {}),
    }


# ---------------------------------------------------------------------------
# Prompt block formatting
# ---------------------------------------------------------------------------

def features_to_prompt_block(features: Dict) -> str:
    """
    Convert extracted chart features into a compact, LLM-readable text block.
    Uses "pp" (percentage points from open) notation.
    """
    lines: List[str] = []
    lines.append("### Chart Features (extracted from intraday timeseries)")
    lines.append("")

    # ---- FX / USD block ----
    lines.append("#### Chart 1 — USD Strength (FX Composite)")

    fx_pairs: List[Dict] = features.get("fx_pairs", [])
    fx_comp: Dict = features.get("fx_composite", {})

    if not fx_pairs:
        lines.append("- [FX data unavailable]")
    else:
        for p in fx_pairs:
            if not p.get("available"):
                sym = p.get("symbol", "?")
                lines.append(f"- {sym}: [data unavailable]")
                continue
            sym = p.get("symbol", "?")
            inv_tag = " (inverted)" if p.get("inverted") else ""
            net = p["net_pp"]
            rng = p["range_pp"]
            hi = p["high"]
            lo = p["low"]
            lines.append(
                f"- **{sym}{inv_tag}**: net {_fmt_pp(net)}, "
                f"range {rng:+.2f}pp, "
                f"hi {_fmt_pp(hi['value'])} @ {hi['time']}, "
                f"lo {_fmt_pp(lo['value'])} @ {lo['time']}"
            )
            tps = p.get("turning_points", [])
            if tps:
                tp_strs = [f"{t['direction']} {_fmt_pp(t['value'])} @ {t['time']}" for t in tps]
                lines.append(f"  → Turning pts: {'; '.join(tp_strs)}")

    if fx_comp.get("available"):
        net = fx_comp["net_pp"]
        rng = fx_comp["range_pp"]
        hi = fx_comp["high"]
        lo = fx_comp["low"]
        lines.append(
            f"- **FX Composite (5-min mean)**: net {_fmt_pp(net)}, "
            f"range {rng:+.2f}pp, "
            f"hi {_fmt_pp(hi['value'])} @ {hi['time']}, "
            f"lo {_fmt_pp(lo['value'])} @ {lo['time']}"
        )
        tps = fx_comp.get("turning_points", [])
        if tps:
            tp_strs = [f"{t['direction']} {_fmt_pp(t['value'])} @ {t['time']}" for t in tps]
            lines.append(f"  → Composite turning pts: {'; '.join(tp_strs)}")
    else:
        lines.append("- FX Composite: [unavailable]")

    lines.append("")

    # ---- Gold / Oil / Bitcoin block ----
    lines.append("#### Chart 2 — Gold vs Oil vs Bitcoin")

    assets: Dict = features.get("assets", {})
    for asset_name in ["Gold", "Oil", "Bitcoin"]:
        stats = assets.get(asset_name, {})
        if not stats.get("available"):
            lines.append(f"- **{asset_name}**: [data unavailable]")
            continue
        sym = stats.get("symbol", asset_name)
        net = stats["net_pp"]
        rng = stats["range_pp"]
        hi = stats["high"]
        lo = stats["low"]
        lines.append(
            f"- **{asset_name}** ({sym}): net {_fmt_pp(net)}, "
            f"range {rng:+.2f}pp, "
            f"hi {_fmt_pp(hi['value'])} @ {hi['time']}, "
            f"lo {_fmt_pp(lo['value'])} @ {lo['time']}"
        )
        tps = stats.get("turning_points", [])
        if tps:
            tp_strs = [f"{t['direction']} {_fmt_pp(t['value'])} @ {t['time']}" for t in tps]
            lines.append(f"  → Turning pts: {'; '.join(tp_strs)}")

    # Divergence
    div: Dict = features.get("divergence", {})
    if div:
        best = div.get("best_asset", "?")
        worst = div.get("worst_asset", "?")
        spread = div.get("spread_pp", 0.0)
        best_net = div.get("best_net_pp", 0.0)
        worst_net = div.get("worst_net_pp", 0.0)
        lines.append(
            f"- **Divergence**: Best={best} ({_fmt_pp(best_net)}), "
            f"Worst={worst} ({_fmt_pp(worst_net)}), "
            f"Spread={spread:+.2f}pp"
        )
    else:
        lines.append("- Divergence: [insufficient data]")

    # Correlations
    corr: Dict = features.get("correlations", {})
    corr_parts = []
    for key, val in corr.items():
        label = key.replace("_", "–")
        if val is not None:
            corr_parts.append(f"{label} r={val:+.2f}")
        else:
            corr_parts.append(f"{label} r=[n/a]")
    if corr_parts:
        lines.append(f"- **5-min Correlations**: {', '.join(corr_parts)}")
    else:
        lines.append("- Correlations: [unavailable]")

    lines.append("")

    return "\n".join(lines)


def _fmt_pp(val: float) -> str:
    """Format a percentage-point value with sign and 2 decimal places."""
    return f"{val:+.2f}pp"
