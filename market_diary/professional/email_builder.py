from __future__ import annotations

import html
from typing import Any, Dict, List, Optional

from market_diary.professional.instruments import format_summary_change
from market_diary.professional.report_formatting import _truncate


def _safe(value: Any) -> str:
    return html.escape(str(value or ""), quote=True)


def _hk_lens(bundle: Dict[str, Any]) -> str:
    view = bundle.get("hk_desk_view", {}) or {}
    llm = bundle.get("llm_sections", {}) or {}
    return str(view.get("lens") or llm.get("hk_local_leadership") or view.get("leadership") or "").strip()


def _hk_local_highlights(bundle: Dict[str, Any], limit: int = 4) -> List[Dict[str, str]]:
    rows = []
    for item in (bundle.get("hk_quick_checks", []) or []):
        status = str(item.get("status", ""))
        if status in {"live_local", "stale_local", "live_hybrid", "live_public", "stale_public"}:
            rows.append({"metric": str(item.get("metric", "")), "value": str(item.get("value", ""))})
        if len(rows) >= limit:
            break
    return rows


def _market_tiles(bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    summary = bundle.get("market_summary", {}) or {}
    wanted = [
        ("Equities", "S&P 500", "S&P 500"),
        ("Equities", "Nasdaq 100", "Nasdaq 100"),
        ("Equities", "Hang Seng Index", "Hang Seng"),
        ("Equities", "Hang Seng TECH ETF", "3033.HK ETF"),
        ("Rates", "10Y Treasury", "US 10Y"),
        ("FX", "USD/CNH", "USD/CNH"),
    ]
    rows = []
    for category, name, fallback in wanted:
        item = (summary.get(category, {}) or {}).get(name, {}) or {}
        if not isinstance(item, dict) or item.get("Price") is None:
            continue
        price = item.get("Price")
        if str(item.get("Price Unit", "")) == "yield_pct":
            level = f"{float(price):.3f}%"
        else:
            try:
                level = f"{float(price):,.2f}"
            except (TypeError, ValueError):
                level = str(price)
        rows.append(
            {
                "label": str(item.get("Display Name") or fallback),
                "level": level,
                "change": format_summary_change(item),
            }
        )
    return rows


def build_email_subject(bundle: Dict[str, Any]) -> str:
    meta = bundle.get("meta", {}) or {}
    overview = bundle.get("overview", {}) or {}
    return f"Hong Kong Morning Briefing | {meta.get('briefing_date', meta.get('report_date', ''))} | {overview.get('risk_regime', 'Market')}"


def build_email_text(bundle: Dict[str, Any]) -> str:
    meta = bundle.get("meta", {}) or {}
    llm = bundle.get("llm_sections", {}) or {}
    overview = bundle.get("overview", {}) or {}
    today = bundle.get("today_forward", {}) or {}
    quality = bundle.get("report_quality", {}) or {}
    release = quality.get("release_recommendation", {}) or {}
    pulse = llm.get("one_line_market_pulse") or overview.get("theme", "")
    hk_view = bundle.get("hk_desk_view", {}) or {}
    hk_lens = _hk_lens(bundle)
    risk_check = hk_view.get("invalidation") or llm.get("risk_check") or "Reassess if rates, CNH, or Hong Kong breadth contradict the base case."

    lines = [
        f"HONG KONG MORNING BRIEF | {meta.get('briefing_date', meta.get('report_date', ''))}",
        "Commute edition: 5-minute scan, 25-30 minute deep read, optional 10-15 minute appendix.",
        f"Global through {meta.get('global_market_date', meta.get('effective_date', ''))} | HK/China through {meta.get('hk_data_date', meta.get('data_through', ''))}",
        "",
        f"BASE CASE: {pulse}",
        f"HONG KONG LENS: {hk_lens}",
        f"INVALIDATE / REASSESS: {risk_check}",
        f"QUALITY: {quality.get('score', 'N/A')}/100 ({quality.get('grade', 'N/A')}) | {release.get('label', 'N/A')}",
        "",
        "MARKET TAPE",
    ]
    lines.extend(f"- {row['label']}: {row['level']} | {row['change']}" for row in _market_tiles(bundle))
    lines.extend(["", "READ THIS FIRST"])
    for idx, item in enumerate((bundle.get("must_watch", []) or [])[:3], 1):
        lines.append(f"{idx}. {item.get('title', '')} — {_truncate(item.get('summary', ''), 180)}")
    lines.extend(["", "TODAY'S CONFIRMATION TESTS"])
    lines.extend(f"- {line}" for line in (today.get("focus_lines", []) or [])[:3])
    local = _hk_local_highlights(bundle)
    if local:
        lines.extend(["", "HONG KONG LOCAL CHECKS"])
        lines.extend(f"- {item['metric']}: {item['value']}" for item in local)
    lines.extend(
        [
            "",
            "DEEP-READ SETUP",
            str(llm.get("deep_read_setup") or overview.get("theme", "")),
            str(llm.get("hk_review_setup") or ""),
            "",
            "The full Markdown report and visual dashboard are attached.",
        ]
    )
    return "\n".join(line for line in lines if line is not None)


def build_email_html(bundle: Dict[str, Any], dashboard_cid: Optional[str] = None) -> str:
    meta = bundle.get("meta", {}) or {}
    llm = bundle.get("llm_sections", {}) or {}
    overview = bundle.get("overview", {}) or {}
    today = bundle.get("today_forward", {}) or {}
    report_quality = bundle.get("report_quality", {}) or {}
    release = report_quality.get("release_recommendation", {}) or {}
    market_quality = meta.get("market_quality", {}) or {}

    pulse = llm.get("one_line_market_pulse") or overview.get("theme", "")
    hk_view = bundle.get("hk_desk_view", {}) or {}
    hk_lens = _hk_lens(bundle)
    deep_read = llm.get("deep_read_setup") or overview.get("theme", "")
    hk_read = llm.get("hk_review_setup") or "Use local breadth, turnover, Southbound activity and CNH to confirm the overseas signal."
    risk_check = hk_view.get("invalidation") or llm.get("risk_check") or "Reassess if rates, CNH, or Hong Kong breadth contradict the base case."
    hk_confirmation = hk_view.get("confirmation") or "Watch Hong Kong breadth, CNH and local flow for same-day confirmation."
    interview_answer = llm.get("interview_answer") or pulse

    tape_cells = [
        f"<td class='metric'><span>{_safe(row['label'])}</span><strong>{_safe(row['level'])}</strong><em>{_safe(row['change'])}</em></td>"
        for row in _market_tiles(bundle)
    ]
    tape_html = "".join(f"<tr>{''.join(tape_cells[index:index + 3])}</tr>" for index in range(0, len(tape_cells), 3))
    checklist_html = "".join(
        "<li><strong>" + _safe(item.get("title", "")) + "</strong><p>" + _safe(_truncate(item.get("summary", ""), 220)) + "</p></li>"
        for item in (bundle.get("must_watch", []) or [])[:3]
    ) or "<li><strong>No priority item</strong><p>The deterministic report did not identify a decision-grade watch item.</p></li>"
    focus_html = "".join(f"<li>{_safe(line)}</li>" for line in (today.get("focus_lines", []) or [])[:3])
    local_html = "".join(
        f"<tr><th>{_safe(item['metric'])}</th><td>{_safe(item['value'])}</td></tr>" for item in _hk_local_highlights(bundle)
    )
    guidance_html = "".join(
        f"<li><strong>{_safe(str(item.get('level', 'advisory')).capitalize())}:</strong> {_safe(item.get('message', ''))}</li>"
        for item in (report_quality.get("runtime_guidance", []) or [])[:3]
    )
    dashboard_html = (
        f"<section><div class='eyebrow'>VISUAL SNAPSHOT</div><h2>Cross-asset dashboard</h2><img class='dashboard' src='cid:{_safe(dashboard_cid)}' alt='Cross-asset research dashboard'></section>"
        if dashboard_cid
        else ""
    )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{_safe(build_email_subject(bundle))}</title>
  <style>
    * {{ box-sizing:border-box; }}
    body {{ margin:0; background:#eef1f3; color:#18232c; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif; line-height:1.55; }}
    .shell {{ width:100%; max-width:820px; margin:0 auto; background:#fff; border-top:7px solid #123a56; }}
    header, section {{ padding:28px 38px; }}
    section {{ border-top:1px solid #dfe5e8; }}
    .eyebrow {{ color:#1f5f8b; font-size:11px; font-weight:800; letter-spacing:1.4px; }}
    h1 {{ margin:8px 0 7px; font:700 31px/1.16 Georgia,"Times New Roman",serif; color:#102a43; }}
    h2 {{ margin:5px 0 14px; font:700 21px/1.25 Georgia,"Times New Roman",serif; color:#102a43; }}
    .meta {{ margin:0; color:#60717d; font-size:13px; }}
    .route {{ display:inline-block; margin-top:15px; padding:7px 10px; background:#e9f1f5; color:#123a56; font-size:12px; font-weight:700; }}
    .verdict {{ margin:22px 0 0; padding:18px 20px; border-left:4px solid #d18b2c; background:#f8f6f1; font:700 19px/1.45 Georgia,"Times New Roman",serif; }}
    .decision {{ width:100%; border-collapse:collapse; margin-top:18px; }}
    .decision td {{ width:33.3%; vertical-align:top; padding:12px; border:1px solid #dfe5e8; }}
    .decision span {{ display:block; color:#6a7882; font-size:10px; font-weight:800; letter-spacing:1px; }}
    .decision strong {{ display:block; margin-top:5px; color:#123a56; font-size:15px; }}
    .tape {{ width:100%; border-collapse:separate; border-spacing:7px; margin:0 -7px; }}
    .metric {{ width:33.3%; min-width:150px; padding:12px; background:#f4f6f7; border-top:3px solid #9aabb5; }}
    .metric span, .metric strong, .metric em {{ display:block; }}
    .metric span {{ min-height:34px; color:#566874; font-size:11px; font-weight:700; }}
    .metric strong {{ color:#102a43; font-size:18px; }}
    .metric em {{ color:#b06d16; font-size:13px; font-style:normal; font-weight:700; }}
    ol, ul {{ margin:0; padding-left:21px; }}
    li {{ margin:0 0 12px; }}
    li p {{ margin:3px 0 0; color:#4e5f69; }}
    .frame {{ width:100%; border-collapse:collapse; }}
    .frame th, .frame td {{ padding:10px 0; border-bottom:1px solid #e5eaed; text-align:left; vertical-align:top; }}
    .frame th {{ width:34%; color:#566874; font-size:12px; }}
    .prose {{ margin:0 0 14px; color:#263844; }}
    .dashboard {{ display:block; width:100%; height:auto; border:1px solid #dfe5e8; }}
    footer {{ padding:22px 38px 30px; background:#102a43; color:#d9e3e8; font-size:12px; }}
    @media only screen and (max-width:600px) {{
      header, section {{ padding:22px 18px; }}
      h1 {{ font-size:27px; }}
      .verdict {{ font-size:17px; padding:15px; }}
      .decision, .decision tbody, .decision tr, .decision td {{ display:block; width:100%; }}
      .decision td {{ border-bottom:0; }}
      .decision td:last-child {{ border-bottom:1px solid #dfe5e8; }}
      .tape, .tape tbody, .tape tr {{ display:block; width:100%; }}
      .metric {{ display:inline-block; width:48%; min-width:0; margin:1%; vertical-align:top; }}
      footer {{ padding:20px 18px; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header>
      <div class="eyebrow">HONG KONG INSTITUTIONAL RESEARCH · COMMUTE EDITION</div>
      <h1>Morning Research Workbench</h1>
      <p class="meta">Issue {_safe(meta.get('briefing_date', meta.get('report_date', '')))} · Global through {_safe(meta.get('global_market_date', meta.get('effective_date', '')))} · HK/China through {_safe(meta.get('hk_data_date', meta.get('data_through', '')))}</p>
      <div class="route">5 min scan · 25–30 min deep read · optional 10–15 min appendix</div>
      <div class="verdict">{_safe(pulse)}</div>
      <table class="decision" role="presentation"><tr>
        <td><span>REGIME</span><strong>{_safe(overview.get('risk_regime', 'N/A'))}</strong></td>
        <td><span>REPORT QUALITY</span><strong>{_safe(report_quality.get('score', 'N/A'))}/100 · {_safe(report_quality.get('grade', 'N/A'))}</strong></td>
        <td><span>RELEASE</span><strong>{_safe(release.get('label', 'N/A'))}</strong></td>
      </tr></table>
    </header>
    <section>
      <div class="eyebrow">MARKET TAPE</div><h2>Levels, moves and units</h2>
      <table class="tape" role="presentation">{tape_html}</table>
    </section>
    <section>
      <div class="eyebrow">READ THIS FIRST</div><h2>Three decisions that matter</h2>
      <ol>{checklist_html}</ol>
    </section>
    <section>
      <div class="eyebrow">DECISION FRAME</div><h2>What confirms or breaks the view</h2>
      <table class="frame">
        <tr><th>Base case</th><td>{_safe(pulse)}</td></tr>
        <tr><th>Hong Kong lens</th><td>{_safe(hk_lens)}</td></tr>
        <tr><th>HK confirmation</th><td>{_safe(hk_confirmation)}</td></tr>
        <tr><th>Confirmation tests</th><td><ul>{focus_html or '<li>No same-day confirmation test was available.</li>'}</ul></td></tr>
        <tr><th>Invalidate / reassess</th><td>{_safe(risk_check)}</td></tr>
      </table>
    </section>
    <section>
      <div class="eyebrow">DEEP READ · 10–15 MIN IN EMAIL</div><h2>Overnight transmission into Hong Kong</h2>
      <p class="prose">{_safe(deep_read)}</p><p class="prose">{_safe(hk_read)}</p>
    </section>
    {dashboard_html}
    {"<section><div class='eyebrow'>LOCAL CONFIRMATION</div><h2>Hong Kong local checks</h2><table class='frame'>" + local_html + "</table></section>" if local_html else ""}
    {"<section><div class='eyebrow'>DATA USE</div><h2>Caveats that travel with the report</h2><ul>" + guidance_html + "</ul></section>" if guidance_html else ""}
    <section>
      <div class="eyebrow">60-SECOND ANSWER</div><h2>How to say it</h2><p class="prose">{_safe(interview_answer)}</p>
    </section>
    <footer>Coverage {_safe(market_quality.get('available', 'N/A'))}/{_safe(market_quality.get('total', 'N/A'))}. The attached full report is designed for 35–50 minutes including charts and the optional audit appendix.</footer>
  </main>
</body>
</html>
"""
