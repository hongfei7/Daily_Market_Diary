from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple


def _make_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def _cell(value: Any) -> str:
        text = str(value)
        return text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")

    lines = [
        "| " + " | ".join(_cell(header) for header in headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _visual_rows(
    dashboard_rel_path: str,
    daily_chart_rel_path: str,
    trend_pack_rel_path: str,
) -> List[Tuple[str, str, str]]:
    rows: List[Tuple[str, str, str]] = []
    if dashboard_rel_path:
        rows.append(
            (
                "Visual Dashboard",
                f"`{dashboard_rel_path}`",
                "Cross-asset regime board and Hong Kong local tape snapshot.",
            )
        )
    if daily_chart_rel_path:
        rows.append(
            (
                "Daily One Chart",
                f"`{daily_chart_rel_path}`",
                "Single highest-conviction visual story for the day.",
            )
        )
    if trend_pack_rel_path:
        rows.append(
            (
                "Hong Kong Trend Pack",
                f"`{trend_pack_rel_path}`",
                "Historical context for flows, liquidity, leadership, and A/H dispersion.",
            )
        )
    return rows


def render_chart_appendix(
    bundle: Dict[str, Any],
    dashboard_rel_path: str = "",
    daily_chart_rel_path: str = "",
    trend_pack_rel_path: str = "",
) -> str:
    overview = bundle.get("overview", {}) or {}
    chart_read = overview.get("chart_read", {}) or {}
    risk = ((bundle.get("attribution", {}) or {}).get("risk_dashboard", {}) or {})

    lines: List[str] = [
        "This appendix is professional-path only. It keeps a traceable index of the report visuals and the deterministic chart cues used to frame the morning note, without injecting the legacy intraday chart pack.",
        "",
    ]

    visual_rows = _visual_rows(dashboard_rel_path, daily_chart_rel_path, trend_pack_rel_path)
    if visual_rows:
        lines.append("### Visual Index")
        lines.append(_make_table(["Visual", "Path", "Role"], visual_rows))
        lines.append("")

    fx_bullets = chart_read.get("fx", []) or []
    asset_bullets = chart_read.get("assets", []) or []
    if fx_bullets or asset_bullets:
        lines.append("### Deterministic Chart Cues")
        for bullet in fx_bullets[:3]:
            lines.append(f"- FX / liquidity: {bullet}")
        for bullet in asset_bullets[:4]:
            lines.append(f"- Cross-asset: {bullet}")
        lines.append("")

    components = risk.get("components", []) or []
    if components:
        lines.append("### Risk Dashboard Components")
        rows = [
            (
                item.get("label", ""),
                f"{item.get('delta', 0):+}",
                str(item.get("evidence", "") or "")[:84],
            )
            for item in components[:6]
        ]
        lines.append(_make_table(["Component", "Score impact", "Evidence"], rows))

    return "\n".join(line for line in lines if line is not None).strip()
