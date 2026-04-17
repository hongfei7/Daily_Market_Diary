from __future__ import annotations

import io
import json
import os
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import requests
import yfinance as yf

from modules.adapter_ah_premium import AH_UNIVERSE
from modules.adapter_stockconnect import fetch_stock_connect_data
from modules.hk_local_data import HKMA_LIQUIDITY_URL, REQUEST_TIMEOUT, USER_AGENT


INK = "#102a43"
SLATE = "#486581"
LINE = "#d9e2ec"
FIG_BG = "#f8fafc"
PANEL_BG = "#ffffff"
GREEN = "#1f7a3e"
RED = "#b42318"
AMBER = "#d97706"
BLUE = "#0b4f71"
DEFAULT_CACHE_ROOT = os.path.join("reports_professional", "raw", "trend_cache")
TREND_PACK_CACHE_VERSION = 1


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _cache_root(cache_dir: Optional[str]) -> str:
    root = cache_dir or DEFAULT_CACHE_ROOT
    os.makedirs(root, exist_ok=True)
    return root


def _safe_cache_key(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value or "cache"))


def _cache_path(cache_dir: Optional[str], bucket: str, key: str) -> str:
    folder = os.path.join(_cache_root(cache_dir), bucket)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"{_safe_cache_key(key)}.json")


def _load_json_cache(path: str) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _write_json_cache(path: str, payload: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).replace("%", "").replace(",", "").replace("HK$", "").replace("bn", "").strip()
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _download_daily_close_uncached(symbol: str, start: date, end: date) -> Dict[str, float]:
    sink = io.StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            hist = yf.download(
                symbol,
                start=start,
                end=end,
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )
    except Exception:
        return {}
    if hist is None or hist.empty or "Close" not in hist:
        return {}
    close = hist["Close"].dropna()
    if close.empty:
        return {}
    if hasattr(close, "columns"):
        if len(close.columns) == 0:
            return {}
        close = close.iloc[:, 0]
    output: Dict[str, float] = {}
    for idx, value in close.items():
        try:
            output[pd_timestamp_to_date(idx)] = float(value)
        except (TypeError, ValueError):
            continue
    return output


def _series_covers_window(series: Dict[str, Any], start: date, end: date) -> bool:
    if not series:
        return False
    requested_start = start.isoformat()
    requested_end = (end - timedelta(days=1)).isoformat()
    dates = sorted(str(key) for key in series.keys())
    return bool(dates and dates[0] <= requested_start and dates[-1] >= requested_end)


def _slice_series(series: Dict[str, Any], start: date, end: date) -> Dict[str, float]:
    start_key = start.isoformat()
    end_key = end.isoformat()
    output: Dict[str, float] = {}
    for day, value in series.items():
        day_key = str(day)
        if start_key <= day_key < end_key:
            try:
                output[day_key] = float(value)
            except (TypeError, ValueError):
                continue
    return output


def _download_daily_close(symbol: str, start: date, end: date, cache_dir: Optional[str] = None) -> Dict[str, float]:
    if not cache_dir:
        return _download_daily_close_uncached(symbol, start, end)

    path = _cache_path(cache_dir, "yahoo_daily", symbol)
    cached = _load_json_cache(path) or {}
    series = cached.get("series", {}) if isinstance(cached.get("series"), dict) else {}

    if not _series_covers_window(series, start, end):
        fetch_start = start
        cached_dates = sorted(str(key) for key in series.keys())
        requested_end = (end - timedelta(days=1)).isoformat()
        if cached_dates and cached_dates[0] <= start.isoformat() and cached_dates[-1] < requested_end:
            fetch_start = _parse_date(cached_dates[-1]) + timedelta(days=1)

        if fetch_start < end:
            fresh = _download_daily_close_uncached(symbol, fetch_start, end)
            if fresh:
                series.update(fresh)
                _write_json_cache(
                    path,
                    {
                        "symbol": symbol,
                        "updated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "series": {key: float(value) for key, value in sorted(series.items())},
                    },
                )

    return _slice_series(series, start, end)


def pd_timestamp_to_date(value: Any) -> str:
    if hasattr(value, "date"):
        return value.date().isoformat()
    return str(value)[:10]


def _fetch_stock_connect_day_cached(report_date: str, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    if not cache_dir:
        return fetch_stock_connect_data(report_date, lookback_days=0)

    path = _cache_path(cache_dir, "stock_connect_daily", report_date)
    cached = _load_json_cache(path)
    if cached:
        return cached

    payload = fetch_stock_connect_data(report_date, lookback_days=0)
    if payload.get("status") in {"ok", "partial"}:
        _write_json_cache(path, payload)
    return payload


def _collect_southbound_history(report_date: str, sessions: int = 20, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    target = _parse_date(report_date)
    rows: List[Dict[str, Any]] = []
    seen_dates = set()
    running = 0.0

    for offset in range(sessions * 3):
        day = target - timedelta(days=offset)
        payload = _fetch_stock_connect_day_cached(day.isoformat(), cache_dir=cache_dir)
        if payload.get("status") != "ok":
            continue
        effective_date = str((payload.get("meta", {}) or {}).get("effective_date", "") or "")
        if not effective_date or effective_date in seen_dates or effective_date > report_date:
            continue
        southbound = (payload.get("data", {}) or {}).get("southbound", {}) or {}
        net_buy = _safe_float(southbound.get("net_buy"))
        turnover = _safe_float(southbound.get("total_turnover"))
        if net_buy is None and turnover is None:
            continue
        seen_dates.add(effective_date)
        rows.append(
            {
                "date": effective_date,
                "net_buy_hkd_bn": (net_buy / 1000.0) if net_buy is not None else None,
                "turnover_hkd_bn": (turnover / 1000.0) if turnover is not None else None,
            }
        )
        if len(rows) >= sessions:
            break

    rows.sort(key=lambda item: item["date"])
    for item in rows:
        running += item.get("net_buy_hkd_bn") or 0.0
        item["cumulative_hkd_bn"] = running
    return rows


def _collect_hkma_history(report_date: str, sessions: int = 30, cache_dir: Optional[str] = None) -> List[Dict[str, Any]]:
    target = _parse_date(report_date)
    path = _cache_path(cache_dir, "hkma", "liquidity_latest") if cache_dir else ""
    cached = _load_json_cache(path) if path else None
    records = ((cached.get("result", {}) or {}).get("records", []) or []) if cached else []

    if not records or str(((records[0] or {}).get("end_of_date", "") or "")) < report_date:
        try:
            response = _session().get(HKMA_LIQUIDITY_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            payload = response.json()
            records = ((payload.get("result", {}) or {}).get("records", []) or [])
            if path and records:
                _write_json_cache(
                    path,
                    {
                        "fetched_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "result": {"records": records},
                    },
                )
        except Exception:
            return []

    rows: List[Dict[str, Any]] = []
    for record in records:
        end_of_date = str(record.get("end_of_date", "") or "")
        if not end_of_date or end_of_date > report_date:
            continue
        rows.append(
            {
                "date": end_of_date,
                "hibor_1m": _safe_float(record.get("hibor_fixing_1m")),
                "aggregate_balance_bn": (_safe_float(record.get("closing_balance")) or 0.0) / 1000.0 if record.get("closing_balance") is not None else None,
            }
        )
    rows.sort(key=lambda item: item["date"])
    return rows[-sessions:]


def _collect_leadership_history(report_date: str, sessions: int = 30, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    target = _parse_date(report_date)
    start = target - timedelta(days=75)
    end = target + timedelta(days=1)
    series_map = {
        "HSI": _download_daily_close("^HSI", start, end, cache_dir=cache_dir),
        "HSCEI": _download_daily_close("^HSCE", start, end, cache_dir=cache_dir),
        "HSTECH": _download_daily_close("3033.HK", start, end, cache_dir=cache_dir),
    }
    common_dates = sorted(set.intersection(*(set(series.keys()) for series in series_map.values() if series)))
    if not common_dates:
        return {"dates": [], "series": {}}
    selected_dates = common_dates[-sessions:]
    output_series: Dict[str, List[float]] = {}
    for label, series in series_map.items():
        values = [series.get(day) for day in selected_dates]
        if not values or values[0] in {None, 0}:
            continue
        base = float(values[0])
        output_series[label] = [float(value) / base * 100.0 if value is not None else np.nan for value in values]
    return {"dates": selected_dates, "series": output_series}


def _fx_history(report_date: str, day_window: int = 14, cache_dir: Optional[str] = None) -> Dict[str, float]:
    target = _parse_date(report_date)
    start = target - timedelta(days=day_window)
    end = target + timedelta(days=1)
    direct = _download_daily_close("CNYHKD=X", start, end, cache_dir=cache_dir)
    if direct:
        return direct
    usd_hkd = _download_daily_close("HKD=X", start, end, cache_dir=cache_dir)
    usd_cnh = _download_daily_close("CNH=F", start, end, cache_dir=cache_dir)
    common_dates = sorted(set(usd_hkd.keys()) & set(usd_cnh.keys()))
    output: Dict[str, float] = {}
    for day in common_dates:
        left = usd_hkd.get(day)
        right = usd_cnh.get(day)
        if left and right:
            output[day] = float(left) / float(right)
    return output


def _collect_ah_heatmap_history(
    bundle: Dict[str, Any],
    report_date: str,
    row_limit: int = 8,
    sessions: int = 5,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    current_rows = (((bundle.get("ah_premium", {}) or {}).get("data", {}) or {}).get("top_premium", []) or [])
    selected = current_rows[:row_limit]
    if not selected:
        selected = [
            {
                "name": item["name"],
                "a_ticker": item["a"],
                "h_ticker": item["h"],
            }
            for item in AH_UNIVERSE[:row_limit]
        ]

    target = _parse_date(report_date)
    start = target - timedelta(days=20)
    end = target + timedelta(days=1)
    fx_history = _fx_history(report_date, day_window=20, cache_dir=cache_dir)
    if not fx_history:
        return {"dates": [], "names": [], "matrix": []}

    matrix: List[List[float]] = []
    names: List[str] = []
    common_date_list: Optional[List[str]] = None

    for item in selected:
        a_ticker = str(item.get("a_ticker") or item.get("a") or "")
        h_ticker = str(item.get("h_ticker") or item.get("h") or "")
        if not a_ticker or not h_ticker:
            continue
        a_hist = _download_daily_close(a_ticker, start, end, cache_dir=cache_dir)
        h_hist = _download_daily_close(h_ticker, start, end, cache_dir=cache_dir)
        common_dates = sorted(set(a_hist.keys()) & set(h_hist.keys()) & set(fx_history.keys()))
        if not common_dates:
            continue
        recent_dates = common_dates[-sessions:]
        if common_date_list is None or len(recent_dates) < len(common_date_list):
            common_date_list = recent_dates
        premiums: Dict[str, float] = {}
        for day in recent_dates:
            a_price = a_hist.get(day)
            h_price = h_hist.get(day)
            fx_value = fx_history.get(day)
            if not a_price or not h_price or not fx_value or h_price <= 0:
                continue
            premium = ((float(a_price) * float(fx_value) / float(h_price)) - 1.0) * 100.0
            premiums[day] = premium
        if premiums:
            names.append(str(item.get("name", "") or h_ticker))
            matrix.append(premiums)

    if not common_date_list or not matrix:
        return {"dates": [], "names": [], "matrix": []}

    aligned_matrix: List[List[float]] = []
    aligned_names: List[str] = []
    for name, row in zip(names, matrix):
        values = [row.get(day, np.nan) for day in common_date_list]
        if all(np.isnan(value) for value in values):
            continue
        aligned_names.append(name)
        aligned_matrix.append(values)
    return {"dates": common_date_list, "names": aligned_names, "matrix": aligned_matrix}


def collect_hk_trend_pack_data(bundle: Dict[str, Any], sessions: int = 20, cache_dir: Optional[str] = None) -> Dict[str, Any]:
    report_date = str(((bundle.get("meta", {}) or {}).get("hk_data_date") or (bundle.get("meta", {}) or {}).get("report_date") or ""))
    if not report_date:
        return {"southbound": [], "liquidity": [], "leadership": {"dates": [], "series": {}}, "ah_heatmap": {"dates": [], "names": [], "matrix": []}}

    assembled_path = _cache_path(cache_dir, "assembled", f"hk_trend_pack_{report_date}_{sessions}") if cache_dir else ""
    cached = _load_json_cache(assembled_path) if assembled_path else None
    if cached and cached.get("version") == TREND_PACK_CACHE_VERSION and cached.get("report_date") == report_date:
        data = cached.get("data", {})
        if isinstance(data, dict):
            return data

    data = {
        "southbound": _collect_southbound_history(report_date, sessions=sessions, cache_dir=cache_dir),
        "liquidity": _collect_hkma_history(report_date, sessions=max(sessions, 30), cache_dir=cache_dir),
        "leadership": _collect_leadership_history(report_date, sessions=max(20, sessions), cache_dir=cache_dir),
        "ah_heatmap": _collect_ah_heatmap_history(bundle, report_date, row_limit=8, sessions=5, cache_dir=cache_dir),
    }
    if assembled_path:
        _write_json_cache(
            assembled_path,
            {
                "version": TREND_PACK_CACHE_VERSION,
                "report_date": report_date,
                "sessions": sessions,
                "generated_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
                "data": data,
            },
        )
    return data


def _style_axis(ax, title: str, subtitle: str = "") -> None:
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(0.01, 1.08, title, transform=ax.transAxes, fontsize=13.2, fontweight="bold", color=INK, va="bottom")
    if subtitle:
        ax.text(0.01, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=SLATE, va="bottom")
    ax.grid(color="#e4e7ec", linewidth=0.8, alpha=0.75)


def _plot_southbound(ax, data: List[Dict[str, Any]]) -> None:
    _style_axis(ax, "Southbound cumulative flow", "20-session cumulative net buy with daily bars")
    if not data:
        ax.axis("off")
        _style_axis(ax, "Southbound cumulative flow", "Public history was unavailable")
        return
    dates = [item["date"][5:] for item in data]
    daily = [item.get("net_buy_hkd_bn") or 0.0 for item in data]
    cumulative = [item.get("cumulative_hkd_bn") or 0.0 for item in data]
    colors = [GREEN if value >= 0 else RED for value in daily]
    ax.bar(dates, daily, color=colors, alpha=0.75)
    ax.axhline(0, color="#98a2b3", linewidth=1.0)
    ax2 = ax.twinx()
    ax2.plot(dates, cumulative, color=BLUE, linewidth=2.2, marker="o", markersize=3)
    ax.set_ylabel("Daily net buy (HK$ bn)", color=SLATE)
    ax2.set_ylabel("Cumulative (HK$ bn)", color=BLUE)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax2.tick_params(axis="y", labelsize=8, colors=BLUE)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    latest = cumulative[-1]
    ax.text(0.99, 0.93, f"Latest cumulative: {latest:+.1f}bn", transform=ax.transAxes, ha="right", fontsize=9.5, color=INK)


def _plot_liquidity(ax, data: List[Dict[str, Any]]) -> None:
    _style_axis(ax, "HIBOR and Aggregate Balance", "Funding cost versus linked-system liquidity")
    if not data:
        ax.axis("off")
        _style_axis(ax, "HIBOR and Aggregate Balance", "HKMA history was unavailable")
        return
    dates = [item["date"][5:] for item in data]
    hibor = [item.get("hibor_1m") if item.get("hibor_1m") is not None else np.nan for item in data]
    balance = [item.get("aggregate_balance_bn") if item.get("aggregate_balance_bn") is not None else np.nan for item in data]
    ax.plot(dates, hibor, color=RED, linewidth=2.2, label="HIBOR 1M")
    ax.set_ylabel("HIBOR 1M (%)", color=RED)
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8, colors=RED)
    ax2 = ax.twinx()
    ax2.fill_between(dates, balance, color=BLUE, alpha=0.18)
    ax2.plot(dates, balance, color=BLUE, linewidth=2.0, label="Aggregate Balance")
    ax2.set_ylabel("Aggregate Balance (HK$ bn)", color=BLUE)
    ax2.tick_params(axis="y", labelsize=8, colors=BLUE)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    lines = ax.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax.legend(lines, labels, loc="upper left", frameon=False, fontsize=8.5)


def _plot_leadership(ax, leadership: Dict[str, Any]) -> None:
    _style_axis(ax, "Relative leadership", "30-session indexed performance: HSI vs HSCEI vs HSTECH")
    dates = leadership.get("dates", []) or []
    series_map = leadership.get("series", {}) or {}
    if not dates or not series_map:
        ax.axis("off")
        _style_axis(ax, "Relative leadership", "Index history was unavailable")
        return
    colors = {"HSI": BLUE, "HSCEI": GREEN, "HSTECH": RED}
    for label, values in series_map.items():
        ax.plot([day[5:] for day in dates], values, linewidth=2.2, label=label, color=colors.get(label, AMBER))
    ax.axhline(100, color="#98a2b3", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Indexed to 100")
    ax.tick_params(axis="x", rotation=45, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper left", frameon=False, fontsize=8.5)


def _plot_ah_heatmap(ax, heatmap: Dict[str, Any]) -> None:
    _style_axis(ax, "A/H premium heatmap", "Last 5 sessions for the widest covered pairs")
    dates = heatmap.get("dates", []) or []
    names = heatmap.get("names", []) or []
    matrix = heatmap.get("matrix", []) or []
    if not dates or not names or not matrix:
        ax.axis("off")
        _style_axis(ax, "A/H premium heatmap", "Premium history was unavailable")
        return
    values = np.array(matrix, dtype=float)
    im = ax.imshow(values, aspect="auto", cmap="RdYlGn_r")
    ax.set_xticks(range(len(dates)))
    ax.set_xticklabels([day[5:] for day in dates], fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color=INK)
    colorbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=8)
    colorbar.outline.set_visible(False)
    colorbar.set_label("Premium (%)", fontsize=8.5, color=SLATE)
    ax.grid(False)


def generate_hk_trend_pack(
    bundle: Dict[str, Any],
    output_path: str,
    trend_data: Optional[Dict[str, Any]] = None,
    cache_dir: Optional[str] = None,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = trend_data or collect_hk_trend_pack_data(bundle, cache_dir=cache_dir)

    plt.style.use("default")
    fig = plt.figure(figsize=(16, 10.5), facecolor=FIG_BG)
    grid = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.18)

    report_date = str(((bundle.get("meta", {}) or {}).get("report_date") or ""))
    fig.suptitle(f"Hong Kong Trend Pack | {report_date}", fontsize=21, fontweight="bold", x=0.05, ha="left", color=INK, y=0.98)
    fig.text(0.05, 0.945, "Historical context for flows, funding, style leadership, and relative-value pressure", fontsize=11.5, color=SLATE)

    ax1 = fig.add_subplot(grid[0, 0])
    _plot_southbound(ax1, data.get("southbound", []) or [])

    ax2 = fig.add_subplot(grid[0, 1])
    _plot_liquidity(ax2, data.get("liquidity", []) or [])

    ax3 = fig.add_subplot(grid[1, 0])
    _plot_leadership(ax3, data.get("leadership", {}) or {})

    ax4 = fig.add_subplot(grid[1, 1])
    _plot_ah_heatmap(ax4, data.get("ah_heatmap", {}) or {})

    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)

    return {
        "path": os.path.basename(output_path),
        "title": "Hong Kong Trend Pack",
        "caption": "Four historical lenses: Southbound cumulative flow, HKMA funding and liquidity, HSI/HSCEI/HSTECH relative leadership, and A/H premium dispersion.",
        "source": "HKEX Stock Connect, HKMA liquidity data, and public Yahoo Finance quotes",
        "rel_path": f"charts/{os.path.basename(output_path)}",
    }
