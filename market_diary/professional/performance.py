"""Append-only signal ledger and look-ahead-safe research backtest."""

from __future__ import annotations

import json
import math
import re
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


SCHEMA_VERSION = "signal-performance-v1"
HSTECH_ETF_BENCHMARK = "Hang Seng TECH ETF (3033.HK)"
DEFAULT_BENCHMARKS = ("Hang Seng Index", HSTECH_ETF_BENCHMARK)
DEFAULT_HORIZONS = (1, 5, 20)
REGIME_POSITION = {"risk-on": 1, "risk on": 1, "neutral": 0, "risk-off": -1, "risk off": -1}


def _parse_date(value: Any) -> str:
    text = str(value or "").strip()[:10]
    try:
        return datetime.strptime(text, "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError):
        return ""


def _number(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        return number if math.isfinite(number) else None
    cleaned = re.sub(r"[^0-9.+-]", "", str(value or "").replace(",", ""))
    try:
        number = float(cleaned)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _position(regime: Any) -> int:
    normalized = re.sub(r"\s+", " ", str(regime or "").strip().lower())
    return REGIME_POSITION.get(normalized, 0)


def _summary_price(bundle: Mapping[str, Any], category: str, name: str) -> float | None:
    value = ((bundle.get("market_summary", {}) or {}).get(category, {}) or {}).get(name)
    if isinstance(value, Mapping):
        return _number(value.get("Price"))
    return None


def _bundle_prices(bundle: Mapping[str, Any]) -> Dict[str, float]:
    candidates = {
        "Hang Seng Index": ("Equities", "Hang Seng Index"),
        HSTECH_ETF_BENCHMARK: ("Equities", "Hang Seng TECH ETF"),
    }
    prices: Dict[str, float] = {}
    for label, (category, name) in candidates.items():
        value = _summary_price(bundle, category, name)
        if value is not None and value > 0:
            prices[label] = value
    return prices


def observation_from_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any] | None:
    meta = bundle.get("meta", {}) or {}
    report_date = _parse_date(meta.get("briefing_date") or meta.get("report_date"))
    as_of = _parse_date(meta.get("effective_date") or meta.get("market_effective_date") or meta.get("data_through"))
    prices = _bundle_prices(bundle)
    if not as_of or not prices:
        return None
    return {"as_of": as_of, "report_date": report_date, "prices": prices, "source": "current_bundle"}


def signal_from_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any] | None:
    meta = bundle.get("meta", {}) or {}
    report_date = _parse_date(meta.get("briefing_date") or meta.get("report_date"))
    market_as_of = _parse_date(meta.get("effective_date") or meta.get("data_through"))
    regime = str((bundle.get("overview", {}) or {}).get("risk_regime") or "Neutral")
    if not report_date or not market_as_of:
        return None
    release = ((bundle.get("report_quality", {}) or {}).get("release_recommendation", {}) or {}).get("action", "")
    blocked = release == "manual_review" or (bundle.get("source_health", {}) or {}).get("status") == "failed"
    position = 0 if blocked else _position(regime)
    return {
        "signal_id": f"hk-beta:{market_as_of}:{report_date}",
        "report_date": report_date,
        "market_as_of": market_as_of,
        "signal": "blocked" if blocked else regime,
        "position": position,
        "benchmark_scope": list(DEFAULT_BENCHMARKS),
        "entry_policy": "next_available_close",
        "status": "blocked" if blocked else "active",
        "evidence": {
            "risk_score": _number((bundle.get("risk", {}) or {}).get("score")),
            "report_quality": (bundle.get("report_quality", {}) or {}).get("score"),
            "source_health": (bundle.get("source_health", {}) or {}).get("status", "unknown"),
            "market_pulse": str((bundle.get("overview", {}) or {}).get("theme", ""))[:240],
        },
        "source": "current_bundle",
    }


def _markdown_cells(line: str) -> List[str]:
    return [cell.strip().replace("**", "") for cell in line.strip().strip("|").split("|")]


def parse_archived_report(path: Path) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
    """Recover the minimum auditable signal/price fields from legacy Markdown archives."""
    text = path.read_text(encoding="utf-8", errors="replace")
    heading = re.search(r"^# Morning Research Workbench \| (\d{4}-\d{2}-\d{2})", text, re.MULTILINE)
    report_date = _parse_date(heading.group(1) if heading else path.parent.name)
    effective = re.search(r"Market effective date:\s*`([^`]+)`", text)
    market_as_of = _parse_date(effective.group(1) if effective else "")
    risk = re.search(r"Composite risk score:\*\*\s*`?([0-9.]+)/100`?\s*\|\s*\*\*Regime:\*\*\s*`?([^`\n]+?)`?\s*$", text, re.MULTILINE)
    if not risk:
        risk = re.search(r"Composite risk score:.*?([0-9.]+)/100.*?Regime:.*?`?([A-Za-z -]+)`?", text)
    regime = risk.group(2).strip().strip("`") if risk else "Neutral"
    risk_score = _number(risk.group(1)) if risk else None

    dashboard_match = re.search(
        r"### 1\.2 Global Asset Price Dashboard\s*(.*?)(?=\n### 1\.3 )",
        text,
        flags=re.DOTALL,
    )
    dashboard_text = dashboard_match.group(1) if dashboard_match else ""
    prices: Dict[str, float] = {}
    for line in dashboard_text.splitlines():
        if not line.startswith("|"):
            continue
        cells = _markdown_cells(line)
        if len(cells) < 2:
            continue
        label = cells[0]
        if label not in {"Hang Seng Index", "Hang Seng TECH", HSTECH_ETF_BENCHMARK}:
            continue
        price = _number(cells[1].split("/")[0])
        if price is not None and price > 0:
            prices[HSTECH_ETF_BENCHMARK if label == "Hang Seng TECH" else label] = price

    observation = None
    if market_as_of and prices:
        observation = {
            "as_of": market_as_of,
            "report_date": report_date,
            "prices": prices,
            "source": path.as_posix(),
        }
    signal = None
    if report_date and market_as_of and risk:
        signal = {
            "signal_id": f"hk-beta:{market_as_of}:{report_date}",
            "report_date": report_date,
            "market_as_of": market_as_of,
            "signal": regime,
            "position": _position(regime),
            "benchmark_scope": list(DEFAULT_BENCHMARKS),
            "entry_policy": "next_available_close",
            "status": "active",
            "evidence": {"risk_score": risk_score, "source_health": "legacy_unavailable"},
            "source": path.as_posix(),
        }
    return observation, signal


def _merge_observations(observations: Iterable[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    by_date: Dict[str, Dict[str, Any]] = {}
    conflicts: List[str] = []
    conflicted_keys: set[tuple[str, str]] = set()
    for raw in observations:
        as_of = _parse_date(raw.get("as_of"))
        prices: Dict[str, float] = {}
        for name, value in (raw.get("prices", {}) or {}).items():
            parsed = _number(value)
            if parsed is not None and parsed > 0:
                canonical_name = HSTECH_ETF_BENCHMARK if str(name) == "Hang Seng TECH" else str(name)
                prices[canonical_name] = parsed
        if not as_of or not prices:
            continue
        if datetime.strptime(as_of, "%Y-%m-%d").weekday() >= 5:
            conflicts.append(f"{as_of}: excluded non-session weekend effective date")
            continue
        report_date = _parse_date(raw.get("report_date"))
        if as_of not in by_date:
            by_date[as_of] = {
                "as_of": as_of,
                "report_date": report_date,
                "prices": prices,
                "source": raw.get("source", ""),
            }
            continue
        existing = by_date[as_of]["prices"]
        for name, value in prices.items():
            conflict_key = (as_of, name)
            if conflict_key in conflicted_keys:
                continue
            if name in existing and not math.isclose(existing[name], value, rel_tol=1e-6, abs_tol=1e-6):
                conflicts.append(
                    f"{as_of} {name}: {existing[name]} versus {value}; excluded conflicted observation"
                )
                existing.pop(name, None)
                conflicted_keys.add(conflict_key)
                continue
            existing.setdefault(name, value)
    return [by_date[key] for key in sorted(by_date)], conflicts


def _merge_signals(signals: Iterable[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    for raw in signals:
        signal_id = str(raw.get("signal_id", "") or "")
        report_date = _parse_date(raw.get("report_date"))
        market_as_of = _parse_date(raw.get("market_as_of"))
        if not signal_id or not report_date or not market_as_of:
            continue
        if signal_id not in by_id:
            by_id[signal_id] = dict(raw)
            by_id[signal_id]["report_date"] = report_date
            by_id[signal_id]["market_as_of"] = market_as_of
    return sorted(by_id.values(), key=lambda item: (item["market_as_of"], item["report_date"], item["signal_id"]))


def _active_signals(signals: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Use the last published view for each market close before the next observation."""
    latest: Dict[str, Dict[str, Any]] = {}
    for signal in signals:
        key = str(signal.get("market_as_of", ""))
        if key and (key not in latest or str(signal.get("report_date", "")) >= str(latest[key].get("report_date", ""))):
            latest[key] = dict(signal)
    return [latest[key] for key in sorted(latest)]


def _drawdown(returns: Sequence[float]) -> float:
    equity = 1.0
    peak = 1.0
    worst = 0.0
    for value in returns:
        equity *= 1.0 + value
        peak = max(peak, equity)
        worst = min(worst, equity / peak - 1.0)
    return worst


def _portfolio_series(
    observations: Sequence[Mapping[str, Any]],
    signals: Sequence[Mapping[str, Any]],
    benchmark: str,
    cost_bps: float,
) -> List[Dict[str, Any]]:
    prices = [(item["as_of"], _number((item.get("prices", {}) or {}).get(benchmark))) for item in observations]
    prices = [(date_value, price) for date_value, price in prices if price is not None and price > 0]
    active = _active_signals(signals)
    rows: List[Dict[str, Any]] = []
    previous_position = 0
    strategy_equity = 1.0
    benchmark_equity = 1.0
    for index in range(len(prices) - 1):
        entry_date, entry_price = prices[index]
        exit_date, exit_price = prices[index + 1]
        eligible = [signal for signal in active if signal.get("market_as_of", "") < entry_date and signal.get("report_date", "") <= entry_date]
        selected = eligible[-1] if eligible else None
        position = int(selected.get("position", 0) or 0) if selected else 0
        underlying_return = exit_price / entry_price - 1.0
        turnover = abs(position - previous_position)
        cost = turnover * float(cost_bps) / 10000.0
        strategy_return = position * underlying_return - cost
        strategy_equity *= 1.0 + strategy_return
        benchmark_equity *= 1.0 + underlying_return
        rows.append(
            {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "position": position,
                "signal_id": selected.get("signal_id", "") if selected else "",
                "underlying_return": round(underlying_return, 8),
                "strategy_return_net": round(strategy_return, 8),
                "turnover": turnover,
                "strategy_equity": round(strategy_equity, 8),
                "benchmark_equity": round(benchmark_equity, 8),
            }
        )
        previous_position = position
    return rows


def _event_outcomes(
    observations: Sequence[Mapping[str, Any]],
    signals: Sequence[Mapping[str, Any]],
    benchmark: str,
    horizons: Sequence[int],
    cost_bps: float,
) -> List[Dict[str, Any]]:
    prices = [(item["as_of"], _number((item.get("prices", {}) or {}).get(benchmark))) for item in observations]
    prices = [(date_value, price) for date_value, price in prices if price is not None and price > 0]
    outcomes: List[Dict[str, Any]] = []
    for signal in _active_signals(signals):
        entry_candidates = [
            index
            for index, (date_value, _) in enumerate(prices)
            if date_value > signal.get("market_as_of", "") and date_value >= signal.get("report_date", "")
        ]
        if not entry_candidates:
            continue
        entry_index = entry_candidates[0]
        entry_date, entry_price = prices[entry_index]
        row: Dict[str, Any] = {
            "signal_id": signal.get("signal_id", ""),
            "report_date": signal.get("report_date", ""),
            "market_as_of": signal.get("market_as_of", ""),
            "entry_date": entry_date,
            "position": int(signal.get("position", 0) or 0),
            "horizons": {},
        }
        for horizon in horizons:
            exit_index = entry_index + int(horizon)
            if exit_index >= len(prices):
                continue
            exit_date, exit_price = prices[exit_index]
            underlying = exit_price / entry_price - 1.0
            round_trip_cost = (2.0 * float(cost_bps) / 10000.0) if row["position"] else 0.0
            directional = row["position"] * underlying - round_trip_cost
            row["horizons"][str(horizon)] = {
                "exit_date": exit_date,
                "underlying_return": round(underlying, 8),
                "directional_return_net": round(directional, 8),
                "hit": bool(directional > 0) if row["position"] else None,
            }
        outcomes.append(row)
    return outcomes


def _metrics(series: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    returns = [float(row.get("strategy_return_net", 0.0) or 0.0) for row in series]
    benchmark_returns = [float(row.get("underlying_return", 0.0) or 0.0) for row in series]
    active_returns = [value for value, row in zip(returns, series) if int(row.get("position", 0) or 0) != 0]
    sessions = len(returns)
    active_sessions = len(active_returns)
    decision_grade = sessions >= 252 and active_sessions >= 100
    strategy_total = math.prod(1.0 + value for value in returns) - 1.0 if returns else 0.0
    benchmark_total = math.prod(1.0 + value for value in benchmark_returns) - 1.0 if benchmark_returns else 0.0
    volatility = pstdev(returns) * math.sqrt(252) if sessions >= 20 and pstdev(returns) > 0 else None
    annualized = (1.0 + strategy_total) ** (252.0 / sessions) - 1.0 if sessions >= 20 and strategy_total > -1 else None
    sharpe = mean(returns) / pstdev(returns) * math.sqrt(252) if sessions >= 20 and pstdev(returns) > 0 else None
    return {
        "status": "decision_grade" if decision_grade else "exploratory",
        "sessions": sessions,
        "active_sessions": active_sessions,
        "active_exposure_ratio": round(active_sessions / sessions, 4) if sessions else 0.0,
        "sample_gate": {"minimum_sessions": 252, "minimum_active_sessions": 100},
        "cumulative_return_net": round(strategy_total, 6),
        "benchmark_return": round(benchmark_total, 6),
        "excess_return": round(strategy_total - benchmark_total, 6),
        "annualized_return": round(annualized, 6) if annualized is not None else None,
        "annualized_volatility": round(volatility, 6) if volatility is not None else None,
        "sharpe_zero_rf": round(sharpe, 3) if sharpe is not None else None,
        "max_drawdown": round(_drawdown(returns), 6),
        "hit_rate_active_sessions": round(sum(1 for value in active_returns if value > 0) / len(active_returns), 4) if active_returns else None,
        "average_daily_return": round(mean(returns), 8) if returns else None,
        "median_daily_return": round(median(returns), 8) if returns else None,
    }


def build_performance_ledger(
    *,
    observations: Iterable[Mapping[str, Any]],
    signals: Iterable[Mapping[str, Any]],
    benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    cost_bps: float = 10.0,
) -> Dict[str, Any]:
    merged_observations, conflicts = _merge_observations(observations)
    merged_signals = _merge_signals(signals)
    results: Dict[str, Any] = {}
    for benchmark in benchmarks:
        series = _portfolio_series(merged_observations, merged_signals, benchmark, cost_bps)
        outcomes = _event_outcomes(merged_observations, merged_signals, benchmark, horizons, cost_bps)
        event_stats: Dict[str, Any] = {}
        for horizon in horizons:
            resolved = [
                row["horizons"][str(horizon)]
                for row in outcomes
                if str(horizon) in row.get("horizons", {}) and row.get("position")
            ]
            event_stats[str(horizon)] = {
                "resolved_signals": len(resolved),
                "hit_rate": round(sum(1 for item in resolved if item.get("hit")) / len(resolved), 4) if resolved else None,
                "average_directional_return_net": round(mean(item["directional_return_net"] for item in resolved), 6) if resolved else None,
            }
        results[benchmark] = {
            "metrics": _metrics(series),
            "event_horizons": event_stats,
            "series": series,
            "outcomes": outcomes,
        }

    price_conflicts = [item for item in conflicts if " versus " in item]
    excluded_non_sessions = [item for item in conflicts if "excluded non-session" in item]
    any_decision_grade = any(value["metrics"]["status"] == "decision_grade" for value in results.values())
    overall_status = "decision_grade_with_caveats" if any_decision_grade and conflicts else "decision_grade" if any_decision_grade else "exploratory_with_caveats" if conflicts else "exploratory"
    return {
        "schema_version": SCHEMA_VERSION,
        "status": overall_status,
        "methodology": {
            "signal": "Published deterministic risk regime mapped to long (+1), neutral (0), or short (-1).",
            "execution": "A signal can enter only at the first available benchmark close on or after publication and strictly after its market as-of date.",
            "portfolio": "Latest published signal per market as-of date; close-to-close returns with turnover costs.",
            "transaction_cost_bps": float(cost_bps),
            "horizons_sessions": [int(value) for value in horizons],
            "look_ahead_guard": True,
            "limitations": [
                "Close-to-close research diagnostic; it is not an executable intraday strategy or investment recommendation.",
                "Legacy signals are reconstructed from immutable published Markdown and may lack current source-health fields.",
                "No dividends, financing, borrow, market impact, taxes, or benchmark reconstitution effects are modeled.",
                "Results remain exploratory until a benchmark has at least 252 sessions and 100 active-signal sessions.",
            ],
        },
        "data_quality": {
            "observations": len(merged_observations),
            "signals": len(merged_signals),
            "active_signal_dates": len(_active_signals(merged_signals)),
            "conflicts": price_conflicts,
            "excluded_non_session_observations": excluded_non_sessions,
            "first_observation": merged_observations[0]["as_of"] if merged_observations else None,
            "last_observation": merged_observations[-1]["as_of"] if merged_observations else None,
        },
        "observations": merged_observations,
        "signals": merged_signals,
        "benchmarks": results,
    }


def _load_existing(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _performance_summary(ledger: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": ledger.get("schema_version", SCHEMA_VERSION),
        "status": ledger.get("status", "insufficient_history"),
        "methodology": ledger.get("methodology", {}),
        "data_quality": ledger.get("data_quality", {}),
        "benchmarks": {
            name: {
                "metrics": value.get("metrics", {}),
                "event_horizons": value.get("event_horizons", {}),
            }
            for name, value in (ledger.get("benchmarks", {}) or {}).items()
        },
    }


def render_performance_chart(ledger: Mapping[str, Any], output_path: Path) -> str:
    import matplotlib.pyplot as plt

    benchmark_payload = (ledger.get("benchmarks", {}) or {}).get("Hang Seng Index", {}) or {}
    series = benchmark_payload.get("series", []) or []
    if not series:
        return ""
    dates = [datetime.strptime(row["exit_date"], "%Y-%m-%d") for row in series]
    strategy = [float(row["strategy_equity"]) for row in series]
    benchmark = [float(row["benchmark_equity"]) for row in series]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.plot(dates, strategy, color="#102A43", linewidth=2.2, label="Regime signal, net")
    ax.plot(dates, benchmark, color="#5B8DB8", linewidth=1.6, label="Hang Seng buy-and-hold")
    ax.axhline(1.0, color="#CBD5E1", linewidth=0.9)
    ax.set_title("Published Signal Performance | Next-close execution", loc="left", color="#102A43", fontsize=14, weight="bold")
    ax.set_ylabel("Growth of 1.00")
    ax.grid(axis="y", color="#E6EBF0", linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.tick_params(colors="#52616B", labelsize=9)
    fig.text(
        0.01,
        0.01,
        f"Research diagnostic; {ledger.get('methodology', {}).get('transaction_cost_bps', 0):g} bps turnover cost. No dividends or market impact.",
        fontsize=8,
        color="#687782",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path.name


def refresh_performance_tracking(
    bundle: Mapping[str, Any],
    *,
    output_dir: str | Path,
    archive_root: str | Path,
    chart_path: str | Path | None = None,
    benchmarks: Sequence[str] = DEFAULT_BENCHMARKS,
    horizons: Sequence[int] = DEFAULT_HORIZONS,
    cost_bps: float = 10.0,
) -> Dict[str, Any]:
    """Merge legacy archives and the current run into a reproducible tracked ledger."""
    output_root = Path(output_dir)
    performance_root = output_root / "performance"
    ledger_path = performance_root / "signal_ledger.json"
    existing = _load_existing(ledger_path)
    observations: List[Mapping[str, Any]] = []
    signals: List[Mapping[str, Any]] = list(existing.get("signals", []) or [])

    for report_path in sorted(Path(archive_root).glob("*/morning_briefing.md")):
        observation, signal = parse_archived_report(report_path)
        if observation:
            observations.append(observation)
        if signal:
            signals.append(signal)

    current_observation = observation_from_bundle(bundle)
    current_signal = signal_from_bundle(bundle)
    if current_observation:
        observations.append(current_observation)
    if current_signal:
        signals.append(current_signal)

    ledger = build_performance_ledger(
        observations=observations,
        signals=signals,
        benchmarks=benchmarks,
        horizons=horizons,
        cost_bps=cost_bps,
    )
    summary = _performance_summary(ledger)
    _write_json(ledger_path, ledger)
    _write_json(performance_root / "performance_summary.json", summary)
    methodology = """# Signal Performance Ledger

This folder is generated by the daily GitHub Actions workflow and is intentionally tracked.

- Signals are the risk regimes actually published by the report, not retrofitted labels.
- Entry is the next available benchmark close after the report's market as-of date.
- Signal history is append-only; outcomes are filled as later closes become available.
- Legacy rows reconstructed from archived Markdown are explicitly labeled in the ledger.
- Results are research diagnostics, not investment advice or executable portfolio returns.
"""
    chart_name = ""
    if chart_path:
        chart_name = render_performance_chart(ledger, Path(chart_path))
        if Path(chart_path).parent.resolve() == performance_root.resolve() and chart_name:
            methodology += f"\n![Published signal performance]({chart_name})\n"
    (performance_root / "README.md").write_text(methodology, encoding="utf-8")
    summary["ledger_path"] = "performance/signal_ledger.json"
    summary["summary_path"] = "performance/performance_summary.json"
    summary["chart_path"] = chart_name
    return summary
