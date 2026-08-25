"""Visualise the overnight-semis to Hong Kong-tech read-through.

Section 2.3 states this chain in words. For a desk that has to decide within
minutes whether the global cycle explains the local tape, seeing the two legs
side by side answers it faster than reading two paragraphs — particularly the
case where Hong Kong moved the same way but several times harder, which means
something local is doing the work.

All figures come from ``build_ai_tmt_chain`` rather than being recomputed, so
the chart and the prose cannot disagree.
"""

from __future__ import annotations

import os
import textwrap
from typing import Any, Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#13202b"
SLATE = "#5d6973"
MUTED = "#7a858e"
LINE = "#d7dde1"
PAPER = "#fbfaf7"
NAVY = "#123a56"
AMBER = "#b45309"
CHART_LAYOUT_VERSION = "ai-tmt-chain-v1"


def _bar_color(value: float) -> str:
    return NAVY if value >= 0 else AMBER


def _leg_values(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [item for item in (rows or []) if item.get("available") and item.get("change_pct") is not None]


def generate_ai_tmt_chain_chart(chain: Dict[str, Any], output_path: str) -> Optional[str]:
    """Render the chain chart. Returns the path, or None when unavailable."""
    if not chain or chain.get("status") == "unavailable":
        return None

    overnight = _leg_values(chain.get("overnight_leg", []))
    hk = _leg_values(chain.get("hk_leg", []))
    if not overnight or not hk:
        return None

    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(12.6, 5.4), facecolor=PAPER, gridspec_kw={"wspace": 0.42}
    )

    # One shared scale across both panels. Independent axes would rescale each
    # leg to fill its panel and hide the amplification this chart exists to
    # show: a 4x larger Hong Kong move would look the same size as the overnight
    # one.
    span = max(abs(float(item["change_pct"])) for item in overnight + hk) or 1.0
    span *= 1.45

    for ax, rows, title, subtitle in (
        (ax_left, overnight, "Overnight leg", "US semis complex, previous close"),
        (ax_right, hk, "Hong Kong leg", "What it read into locally"),
    ):
        labels = [item["label"] for item in rows]
        values = [float(item["change_pct"]) for item in rows]
        positions = range(len(labels))
        ax.barh(list(positions), values, color=[_bar_color(v) for v in values], height=0.55)
        ax.set_yticks(list(positions))
        ax.set_yticklabels(labels, fontsize=10.5, color=INK)
        ax.invert_yaxis()
        ax.axvline(0, color="#98a2b3", linewidth=1)
        ax.set_facecolor("#ffffff")
        ax.grid(axis="x", color=LINE, linewidth=0.8, alpha=0.7)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"{title}\n", fontsize=13, fontweight="bold", color=INK, loc="left", pad=26)
        ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=9.5, color=MUTED, va="bottom")
        ax.set_xlabel("1D %", fontsize=9.5, color=SLATE)

        for pos, value in zip(positions, values):
            # Label inside the bar when it is long enough, otherwise just past
            # its tip. Either way it never reaches the y-axis tick labels.
            inside = abs(value) > span * 0.22
            if inside:
                x = value - (span * 0.02 if value >= 0 else -span * 0.02)
                ha = "right" if value >= 0 else "left"
                color = "#ffffff"
            else:
                x = value + (span * 0.02 if value >= 0 else -span * 0.02)
                ha = "left" if value >= 0 else "right"
                color = _bar_color(value)
            ax.text(x, pos, f"{value:+.2f}%", va="center", ha=ha, fontsize=10, fontweight="bold", color=color)
        ax.set_xlim(-span, span)

    fig.suptitle(
        "AI / TMT hand-off | overnight semis and prior Hong Kong close",
        x=0.055,
        y=0.98,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )

    # State the relationship in the chart, not only in the prose.
    verdict = chain.get("temporal_note") or "Transmission remains pending the next Hong Kong session."
    # Wrap rather than clip: the verdict is the point of the chart.
    caption = textwrap.fill(" ".join(str(verdict).split()), width=118)
    caption_lines = caption.count("\n") + 1
    fig.text(0.055, 0.135, caption, fontsize=9.5, color=SLATE, ha="left", va="top", linespacing=1.5)

    stale = chain.get("stale_inputs") or []
    if stale:
        note = "Excluded as stale: " + "; ".join(str(item) for item in stale[:2])
        fig.text(0.055, 0.02, textwrap.shorten(note, width=118, placeholder="…"), fontsize=8.5, color=MUTED, ha="left")

    # Reserve height for however many lines the caption actually needed, so the
    # verdict is never cut off at the bottom edge.
    fig.subplots_adjust(left=0.11, right=0.965, top=0.78, bottom=0.16 + 0.045 * caption_lines)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=PAPER)
    plt.close(fig)
    return output_path
