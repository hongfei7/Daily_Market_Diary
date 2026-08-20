"""Did yesterday's call actually work?

The signal ledger has recorded a directional call for every published report
(103 of them) alongside close observations, and ``performance.py`` already
scores them in aggregate as hit rates in the appendix. What the report never did
was answer the first question a desk asks in the morning: *we said X yesterday —
did it happen?*

That retrospective is the fastest way to grasp what changed, and for an analyst
still building intuition it is the only part of the report that closes a
feedback loop.

Verdicts are deliberately blunt. A call that failed is reported as BROKEN with
the size of the miss, not softened into ambiguity, and a call that cannot be
scored is UNRESOLVED rather than quietly counted as a win.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence

# Moves smaller than this are treated as noise rather than confirmation either
# way: a +0.05% session does not validate a directional call.
FLAT_THRESHOLD_PCT = 0.25

BENCHMARK_LABEL = "Hang Seng Index"
GROWTH_LABEL = "Hang Seng TECH ETF (3033.HK)"


def normalize_signal(value: Any) -> str:
    """Collapse the ledger's inconsistent casing onto one label per state.

    The ledger holds 'Risk-off' (25), 'Risk-on' (16) and 'Risk-On' (7) as three
    spellings of two states, which breaks any grouping by signal name.
    """
    text = str(value or "").strip().lower().replace("_", "-")
    if text in {"risk-on", "riskon", "risk on"}:
        return "Risk-On"
    if text in {"risk-off", "riskoff", "risk off"}:
        return "Risk-Off"
    if text in {"neutral", "mixed"}:
        return text.capitalize()
    if text == "blocked":
        return "blocked"
    return str(value or "").strip() or "unknown"


def _prices_by_date(observations: Sequence[Mapping[str, Any]]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for item in observations:
        as_of = str(item.get("as_of") or "")
        prices = item.get("prices") or {}
        if as_of and isinstance(prices, Mapping):
            out[as_of] = {str(k): float(v) for k, v in prices.items() if isinstance(v, (int, float))}
    return out


def _previous_signal(signals: Sequence[Mapping[str, Any]], briefing_date: str) -> Optional[Mapping[str, Any]]:
    prior = [item for item in signals if str(item.get("report_date") or "") < str(briefing_date)]
    if not prior:
        return None
    return max(prior, key=lambda item: str(item.get("report_date")))


def _next_close(dates: Sequence[str], after: str, prices: Dict[str, Dict[str, float]], label: str):
    """First observation strictly after ``after`` that carries ``label``."""
    for date in dates:
        if date > after and label in prices.get(date, {}):
            return date, prices[date][label]
    return None, None


def _verdict(position: int, move_pct: Optional[float]) -> str:
    if move_pct is None:
        return "UNRESOLVED"
    if position == 0:
        return "NO CALL"
    if abs(move_pct) < FLAT_THRESHOLD_PCT:
        return "UNRESOLVED"
    return "CONFIRMED" if (move_pct > 0) == (position > 0) else "BROKEN"


def build_call_scorecard(
    ledger: Mapping[str, Any],
    briefing_date: str,
) -> Dict[str, Any]:
    """Score the most recent published call against what actually happened."""
    signals = list(ledger.get("signals", []) or [])
    observations = list(ledger.get("observations", []) or [])
    prices = _prices_by_date(observations)
    dates = sorted(prices)

    previous = _previous_signal(signals, briefing_date)
    if previous is None:
        return {
            "status": "unavailable",
            "verdict": "UNRESOLVED",
            "headline": "No previously published call is on file, so there is nothing to score yet.",
        }

    signal = normalize_signal(previous.get("signal"))
    position = int(previous.get("position") or 0)
    market_as_of = str(previous.get("market_as_of") or "")
    report_date = str(previous.get("report_date") or "")

    if signal == "blocked" or position == 0:
        return {
            "status": "no_call",
            "verdict": "NO CALL",
            "signal": signal,
            "report_date": report_date,
            "headline": (
                f"The {report_date} report published no directional call"
                + (" because release was blocked on quality." if signal == "blocked" else " (neutral regime).")
            ),
            "moves": [],
        }

    moves: List[Dict[str, Any]] = []
    for label in (BENCHMARK_LABEL, GROWTH_LABEL):
        entry = prices.get(market_as_of, {}).get(label)
        exit_date, exit_price = _next_close(dates, market_as_of, prices, label)
        if entry is None or exit_price is None or entry <= 0:
            moves.append({"label": label, "move_pct": None, "reason": "no comparable close"})
            continue
        moves.append(
            {
                "label": label,
                "move_pct": round((exit_price / entry - 1.0) * 100.0, 2),
                "from_date": market_as_of,
                "to_date": exit_date,
            }
        )

    # Prefer the Hang Seng Index, but fall back to the growth proxy when it is
    # unscoreable. A conflicting price for the same date is dropped upstream
    # rather than guessed at, so the headline benchmark can be missing on a day
    # the other benchmark is fine.
    scored = [item for item in moves if item.get("move_pct") is not None]
    headline_move = next((item for item in scored if item["label"] == BENCHMARK_LABEL), None)
    fell_back = False
    if headline_move is None and scored:
        headline_move = scored[0]
        fell_back = True
    verdict = _verdict(position, headline_move["move_pct"] if headline_move else None)
    scored_on = headline_move["label"] if headline_move else ""

    fallback_note = (
        f" Scored on {scored_on} because no comparable {BENCHMARK_LABEL} close was available."
        if fell_back
        else ""
    )
    if verdict == "UNRESOLVED":
        detail = (
            "No comparable next close is available yet, so the call is still open."
            if not headline_move
            else f"{scored_on} moved {headline_move['move_pct']:+.2f}%, inside the "
            f"{FLAT_THRESHOLD_PCT:.2f}% noise band, so the call is neither confirmed nor broken.{fallback_note}"
        )
    elif verdict == "CONFIRMED":
        detail = (
            f"{signal} was published on {report_date} and {scored_on} moved "
            f"{headline_move['move_pct']:+.2f}% into {headline_move['to_date']}, in the called direction."
            f"{fallback_note}"
        )
    else:
        detail = (
            f"{signal} was published on {report_date} but {scored_on} moved "
            f"{headline_move['move_pct']:+.2f}% into {headline_move['to_date']}, against the call. "
            f"Carry that miss into today's read rather than restating the same thesis.{fallback_note}"
        )

    return {
        "status": "ok",
        "verdict": verdict,
        "signal": signal,
        "position": position,
        "report_date": report_date,
        "market_as_of": market_as_of,
        "headline": detail,
        "scored_on": scored_on,
        "moves": moves,
        "evidence": previous.get("evidence", {}) or {},
    }


def recent_record(ledger: Mapping[str, Any], briefing_date: str, limit: int = 10) -> Dict[str, Any]:
    """Roll up the last ``limit`` scoreable calls into a simple record."""
    signals = list(ledger.get("signals", []) or [])
    prices = _prices_by_date(list(ledger.get("observations", []) or []))
    dates = sorted(prices)

    results: List[str] = []
    for item in sorted(signals, key=lambda s: str(s.get("report_date")), reverse=True):
        if str(item.get("report_date") or "") >= str(briefing_date):
            continue
        position = int(item.get("position") or 0)
        if position == 0:
            continue
        market_as_of = str(item.get("market_as_of") or "")
        entry = prices.get(market_as_of, {}).get(BENCHMARK_LABEL)
        _, exit_price = _next_close(dates, market_as_of, prices, BENCHMARK_LABEL)
        if entry is None or exit_price is None or entry <= 0:
            continue
        move = (exit_price / entry - 1.0) * 100.0
        verdict = _verdict(position, move)
        if verdict in {"CONFIRMED", "BROKEN"}:
            results.append(verdict)
        if len(results) >= limit:
            break

    confirmed = results.count("CONFIRMED")
    total = len(results)
    return {
        "scored": total,
        "confirmed": confirmed,
        "broken": total - confirmed,
        "hit_rate_pct": round(confirmed / total * 100.0, 1) if total else None,
        "sequence": results,
    }
