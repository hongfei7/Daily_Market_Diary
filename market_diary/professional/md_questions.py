"""What a senior is most likely to ask this morning, and how to answer it.

An analyst who has read the numbers can still be caught out by the question
behind them. The signals needed to anticipate those questions are already in the
bundle — divergences, tail readings, coverage gaps, a call that just failed —
but nothing assembled them into the form a morning meeting actually takes.

Generated deterministically rather than through the narrative overlay: that
layer has been running at 0-2 of 7 successful tasks, and meeting preparation
should not depend on it.

Each item carries the question, the data that answers it, and a sentence that
can be said out loud without further work.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional

from market_diary.professional.analytics_market import _fresh_change_pct, build_market_snapshot
from market_diary.professional.metric_history import describe, percentile_context

MAX_QUESTIONS = 3


def _fmt(value: Optional[float], suffix: str = "%") -> str:
    return "N/A" if value is None else f"{value:+.2f}{suffix}"


def _local_value(hk_local: Mapping[str, Any], key: str) -> Optional[float]:
    item = hk_local.get(key) if isinstance(hk_local, Mapping) else None
    if not isinstance(item, Mapping):
        return None
    try:
        return float(item.get("value"))
    except (TypeError, ValueError):
        return None


def _style_divergence(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The index and its growth component disagreeing is the classic question."""
    hsi, _ = _fresh_change_pct(rows, "Hang Seng Index")
    tech, _ = _fresh_change_pct(rows, "3033.HK ETF")
    if hsi is None or tech is None:
        return None
    spread = tech - hsi
    if abs(spread) < 1.0:
        return None
    led = "lagged" if spread < 0 else "led"
    return {
        "priority": 10 + abs(spread),
        "question": f"The index was {_fmt(hsi)} but tech {led} at {_fmt(tech)} — what drove the split?",
        "evidence": f"HSI {_fmt(hsi)} versus 3033.HK {_fmt(tech)}, a {abs(spread):.2f}pp spread.",
        "answer": (
            f"This was a style move, not a beta move: the {abs(spread):.2f}pp spread means index direction "
            f"alone does not describe the session. Old-economy and H-share names carried the index while "
            f"growth {led}."
            if spread < 0
            else f"Growth carried the session: the {abs(spread):.2f}pp spread means the move was concentrated "
            f"in platform and tech names rather than broad beta."
        ),
    }


def _flow_price_divergence(rows: List[Dict[str, Any]], hk_local: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """Price and flow pointing opposite ways is the hardest question to duck."""
    hsi, _ = _fresh_change_pct(rows, "Hang Seng Index")
    southbound = _local_value(hk_local, "southbound_net_flow")
    if hsi is None or southbound is None or abs(hsi) < 0.3:
        return None
    if (hsi > 0) == (southbound > 0):
        return None
    flow_bn = southbound / 1_000_000_000
    direction = "rose" if hsi > 0 else "fell"
    flow_side = "sold" if southbound < 0 else "bought"
    return {
        "priority": 12,
        "question": f"The index {direction} but Southbound {flow_side} — is the move real?",
        "evidence": f"HSI {_fmt(hsi)} against Southbound net {flow_bn:+.1f}bn HKD.",
        "answer": (
            "Treat it as unconfirmed. Price and mainland flow disagreed, so the price move was not "
            f"driven by Southbound participation; it needs another session of confirmation before being "
            f"read as a trend."
        ),
    }


def _tail_reading(
    hk_local: Mapping[str, Any],
    metric_history: Mapping[str, Any],
    report_date: str,
) -> Optional[Dict[str, Any]]:
    """A metric in the tail of its own distribution invites a question."""
    value = _local_value(hk_local, "short_selling_ratio")
    context = percentile_context(metric_history or {}, "short_selling_ratio", value, report_date)
    if not context.get("available") or context.get("band") not in {"very high", "very low"}:
        return None
    band = context["band"]
    return {
        "priority": 9,
        "question": f"Short selling was {band} at {value:.1f}% — who is positioned against this market?",
        "evidence": f"Short-selling ratio {value:.1f}% ({describe(context)}).",
        "answer": (
            "Check the concentration before reading it as bearish: ETF-heavy short turnover is macro hedging, "
            "while single-name pressure is company-specific. The HKEX short-selling table in this report "
            "separates the two."
        ),
    }


def _coverage_gap(hk_desk_view: Mapping[str, Any], source_health: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    """A senior will ask why a conclusion is missing before asking anything else."""
    stale = list(hk_desk_view.get("stale_inputs", []) or [])
    if not stale:
        return None
    return {
        "priority": 14,
        "question": "Why is there no style call today?",
        "evidence": "; ".join(stale) + ".",
        "answer": (
            "A required input was stale, so any relative-performance claim would have compared two different "
            "dates. The call is withheld rather than published on partial evidence, and the stale input is "
            "excluded from the risk score."
        ),
    }


def _broken_call(scorecard: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if str(scorecard.get("verdict", "")) != "BROKEN":
        return None
    return {
        "priority": 13,
        "question": "Yesterday's call did not work — what changed?",
        "evidence": str(scorecard.get("headline", "")),
        "answer": (
            "Lead with the miss rather than restating the thesis. The prior read needs new evidence before "
            "it is repeated, and today's call should say explicitly what would make it different."
        ),
    }


def _ai_tmt_amplification(chain: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not chain.get("divergence_note"):
        return None
    return {
        "priority": 11,
        "question": "Did the overnight semis move explain Hong Kong tech?",
        "evidence": str(chain.get("divergence_note", "")),
        "answer": (
            "Not on its own. Same direction is not the same as explained, so check single-name news, "
            "placements and index events before attributing the move to the global cycle."
        ),
    }


def build_md_questions(
    bundle: Mapping[str, Any],
    metric_history: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Assemble the questions most likely to be asked, highest priority first."""
    summary = bundle.get("market_summary", {}) or {}
    rows = build_market_snapshot(summary)
    hk_local = bundle.get("hk_local", {}) or {}
    report_date = str((bundle.get("meta", {}) or {}).get("hk_data_date") or "")

    candidates = [
        _coverage_gap(bundle.get("hk_desk_view", {}) or {}, bundle.get("source_health", {}) or {}),
        _broken_call(bundle.get("call_scorecard", {}) or {}),
        _flow_price_divergence(rows, hk_local),
        _ai_tmt_amplification(bundle.get("ai_tmt_chain", {}) or {}),
        _style_divergence(rows),
        _tail_reading(hk_local, metric_history or {}, report_date),
    ]
    found = [item for item in candidates if item]
    found.sort(key=lambda item: item["priority"], reverse=True)
    return found[:MAX_QUESTIONS]
