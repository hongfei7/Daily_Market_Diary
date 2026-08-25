"""AI / TMT hand-off: overnight semis into the next Hong Kong session.

For a desk covering AI and TMT the overnight semiconductor tape is the most
direct external input to Hong Kong tech, ahead of broad US beta. None of it was
tracked: the instrument registry held no SOXX, TSMC, NVDA, SMIC, Hua Hong or
Sunny Optical, so the report could describe Hong Kong growth as lagging without
ever showing what happened to the complex that drives it.

This module states the hand-off explicitly — overnight leg, the prior Hong Kong
inheritance state, and the observable test at the open — so the reasoning can be
repeated in a morning meeting rather than only the numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from market_diary.professional.analytics_market import (
    _fresh_change_pct,
    _format_signed,
    _get_row,
    _stale_note,
    build_market_snapshot,
)

# The overnight leg, in the order a desk reads it.
OVERNIGHT_LEG = [
    ("SOXX", "Semis complex"),
    ("NVDA", "AI capex narrative"),
    ("TSMC", "Foundry demand"),
]

# Hong Kong names that express the overnight move.
HK_LEG = [
    ("SMIC", "Leading-edge / domestic substitution"),
    ("Hua Hong", "Mature-node pricing cycle"),
    ("Sunny Optical", "Hardware and optics supply chain"),
    ("3033.HK ETF", "Broad HK tech beta"),
]

# A same-direction move this size or larger is treated as a real signal rather
# than noise, matching the 0.5pp threshold used for the style call.
SIGNAL_THRESHOLD_PCT = 0.5

# Hong Kong moving this many times harder than the overnight leg means the
# global cycle is not a sufficient explanation on its own.
AMPLIFICATION_THRESHOLD = 2.0

# A name this far from its own leg's average is idiosyncratic, not a cycle read.
OUTLIER_THRESHOLD_PCT = 4.0


def _leg_rows(rows: List[Dict[str, Any]], leg: List[tuple]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for label, role in leg:
        value, stale_days = _fresh_change_pct(rows, label)
        row = _get_row(rows, label)
        if not row:
            continue
        out.append(
            {
                "label": label,
                "role": role,
                "change_pct": value,
                "stale_days": stale_days,
                "display": _format_signed(value) if value is not None else f"stale {stale_days}d",
                "available": value is not None,
            }
        )
    return out


def _average(values: List[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def build_ai_tmt_chain(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Build the overnight-semis to Hong Kong-tech read-through."""
    rows = build_market_snapshot(summary)
    overnight = _leg_rows(rows, OVERNIGHT_LEG)
    hk = _leg_rows(rows, HK_LEG)

    overnight_available = [item for item in overnight if item["available"]]
    hk_available = [item for item in hk if item["available"]]

    overnight_avg = _average([item["change_pct"] for item in overnight_available])
    hk_avg = _average([item["change_pct"] for item in hk_available])

    stale_inputs = [
        _stale_note(item["label"], item["stale_days"])
        for item in overnight + hk
        if not item["available"] and item["stale_days"] is not None
    ]

    if overnight_avg is None:
        verdict = "unavailable"
        headline = "Overnight semiconductor coverage was unavailable, so no AI/TMT read-through can be formed."
        expression = "Fall back on broad beta and local flow for the tech read."
        test = "Restore SOXX, NVDA and TSMC coverage before relying on this chain."
    else:
        direction = "risk-off" if overnight_avg < 0 else "risk-on"
        if abs(overnight_avg) < SIGNAL_THRESHOLD_PCT:
            verdict = "neutral"
            headline = (
                f"The overnight semis leg was flat ({_format_signed(overnight_avg)} average), "
                "so it does not set the direction for Hong Kong tech today."
            )
            expression = "Let local flow and style leadership drive the tech read rather than the overnight tape."
            test = "Watch whether SMIC and Hua Hong trade with HSCEI rather than with the overnight semis leg."
        else:
            verdict = direction
            headline = (
                f"The overnight semis leg was {direction} at {_format_signed(overnight_avg)} average "
                f"({', '.join(f'{i['label']} {i['display']}' for i in overnight_available)})."
            )
            pressure = "under the most pressure" if overnight_avg < 0 else "best placed"
            expression = (
                f"SMIC, Hua Hong and Sunny Optical are {pressure} at the open, and 3033.HK should be "
                "tested before the Hang Seng Index."
            )
            test = (
                "Confirm if SMIC and Hua Hong open in the same direction as the overnight leg and hold it "
                "through the first hour; invalidate if they track HSCEI instead, which would mean local "
                "flow is overriding the global cycle."
            )

    # The Hong Kong rows are the *prior* cash session and therefore predate the
    # US overnight leg.  They are an inheritance state, not evidence that Hong
    # Kong followed, diverged from, or was explained by the later US move.
    temporal_note = (
        "The Hong Kong rows are the prior cash-session close and predate the US overnight leg. "
        "Use them as the inherited local setup only; validate transmission on today's Hong Kong "
        "open, first hour and close."
    )

    # An individual name far outside the rest of its leg is idiosyncratic, not a
    # cycle read, and must not be presented as evidence for the chain.
    outliers = [
        f"{item['label']} {item['display']}"
        for item in hk_available
        if hk_avg is not None and abs(item["change_pct"] - hk_avg) >= OUTLIER_THRESHOLD_PCT
    ]

    return {
        "status": "ok" if overnight_avg is not None else "unavailable",
        "verdict": verdict,
        "headline": headline,
        "expression": expression,
        "test": test,
        "overnight_leg": overnight,
        "hk_leg": hk,
        "overnight_avg_pct": round(overnight_avg, 2) if overnight_avg is not None else None,
        "hk_avg_pct": round(hk_avg, 2) if hk_avg is not None else None,
        "hk_followed_overnight": None,
        "amplification": None,
        "divergence_note": "",
        "temporal_note": temporal_note,
        "comparison_posture": "pending_next_hk_session",
        "single_name_outliers": outliers,
        "stale_inputs": stale_inputs,
    }
