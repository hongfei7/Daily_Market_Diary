from __future__ import annotations

from typing import Any, Dict, List, Optional

from professional.report_formatting import _truncate


def _hk_local_highlights(bundle: Dict[str, Any], limit: int = 3) -> List[Dict[str, str]]:
    rows = []
    for item in (bundle.get("hk_quick_checks", []) or []):
        status = str(item.get("status", ""))
        if status in {"live_local", "stale_local", "live_hybrid", "live_public", "stale_public"}:
            rows.append(
                {
                    "metric": str(item.get("metric", "")),
                    "value": str(item.get("value", "")),
                }
            )
        if len(rows) >= limit:
            break
    return rows


def build_email_subject(bundle: Dict[str, Any]) -> str:
    meta = bundle.get("meta", {}) or {}
    overview = bundle.get("overview", {}) or {}
    return f"Hong Kong Morning Briefing | {meta.get('briefing_date', meta.get('report_date', ''))} | {overview.get('risk_regime', 'Market')}"


def build_email_text(bundle: Dict[str, Any]) -> str:
    meta = bundle.get("meta", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    overview = bundle.get("overview", {}) or {}
    today_forward = bundle.get("today_forward", {}) or {}
    hk_local_highlights = _hk_local_highlights(bundle)
    report_quality = bundle.get("report_quality", {}) or {}

    lines = [
        f"Hong Kong Morning Briefing | {meta.get('briefing_date', meta.get('report_date', ''))}",
        f"Global markets through: {meta.get('global_market_date', meta.get('effective_date', ''))}",
        f"HK/China local data through: {meta.get('hk_data_date', meta.get('data_through', ''))}",
        "",
        llm_sections.get("one_line_market_pulse") or overview.get("theme", ""),
    ]
    if report_quality:
        lines.append(f"Report quality: {report_quality.get('score', 'N/A')}/100 | Grade {report_quality.get('grade', 'N/A')}")
    lines.extend(["", "Top checklist:"])

    for idx, item in enumerate((bundle.get("must_watch", []) or [])[:5], 1):
        lines.append(f"{idx}. {item.get('title', '')} | {_truncate(item.get('summary', ''), 100)}")

    if today_forward.get("focus_lines"):
        lines.append("")
        lines.append("Today ahead:")
        for line in today_forward.get("focus_lines", [])[:3]:
            lines.append(f"- {line}")

    if hk_local_highlights:
        lines.append("")
        lines.append("Hong Kong local checks:")
        for item in hk_local_highlights:
            lines.append(f"- {item['metric']}: {item['value']}")

    lines.append("")
    lines.append("The full markdown report and dashboard image are attached.")
    return "\n".join(lines)


def build_email_html(bundle: Dict[str, Any], dashboard_cid: Optional[str] = None) -> str:
    meta = bundle.get("meta", {}) or {}
    llm_sections = bundle.get("llm_sections", {}) or {}
    overview = bundle.get("overview", {}) or {}
    quality = meta.get("market_quality", {}) or {}
    today_forward = bundle.get("today_forward", {}) or {}
    hk_local_highlights = _hk_local_highlights(bundle)
    report_quality = bundle.get("report_quality", {}) or {}

    quality_parts: List[str] = []
    if quality.get("available") and quality.get("total"):
        quality_parts.append(f"Coverage {quality['available']}/{quality['total']}")
    if quality.get("fallback"):
        quality_parts.append(f"Fallbacks {len(quality.get('fallback', []))}")
    if quality.get("missing"):
        quality_parts.append(f"Missing {len(quality.get('missing', []))}")
    if report_quality:
        quality_parts.append(f"Report quality {report_quality.get('score', 'N/A')}/100 ({report_quality.get('grade', 'N/A')})")

    checklist_items = []
    for item in (bundle.get("must_watch", []) or [])[:5]:
        checklist_items.append(
            f"<li><strong>{item.get('title', '')}</strong><br>{_truncate(item.get('summary', ''), 140)}</li>"
        )

    focus_items = []
    for line in today_forward.get("focus_lines", [])[:3]:
        focus_items.append(f"<li>{line}</li>")

    hk_local_items = []
    for item in hk_local_highlights:
        hk_local_items.append(f"<li><strong>{item['metric']}</strong>: {item['value']}</li>")

    dashboard_block = (
        f'<div style="margin:16px 0;"><img src="cid:{dashboard_cid}" alt="Research dashboard" style="max-width:100%; border:1px solid #d0d7de; border-radius:8px;"></div>'
        if dashboard_cid
        else ""
    )

    pulse = llm_sections.get("one_line_market_pulse") or overview.get("theme", "")
    deep_read = llm_sections.get("deep_read_setup") or overview.get("theme", "")
    interview_answer = llm_sections.get("interview_answer", "")

    return f"""\
<html>
  <body style="font-family:Segoe UI, Arial, sans-serif; color:#1f2328; line-height:1.5; margin:0; padding:24px; background:#f6f8fa;">
    <div style="max-width:860px; margin:0 auto; background:#ffffff; border:1px solid #d0d7de; border-radius:12px; padding:24px;">
      <div style="font-size:12px; color:#59636e; margin-bottom:8px;">Hong Kong sell-side morning briefing</div>
      <h1 style="margin:0 0 12px 0; font-size:24px;">{meta.get('briefing_date', meta.get('report_date', ''))}</h1>
      <p style="margin:0 0 12px 0; color:#59636e;">Global markets through: {meta.get('global_market_date', meta.get('effective_date', ''))} | HK/China local data through: {meta.get('hk_data_date', meta.get('data_through', ''))}</p>
      <p style="font-size:18px; margin:0 0 16px 0;"><strong>{pulse}</strong></p>
      <p style="margin:0 0 16px 0; color:#59636e;">{ " | ".join(quality_parts) if quality_parts else "Market-quality diagnostics were not available." }</p>
      {dashboard_block}
      <h2 style="font-size:18px; margin:20px 0 8px 0;">Deep-read setup</h2>
      <p style="margin:0 0 16px 0;">{deep_read}</p>
      <h2 style="font-size:18px; margin:20px 0 8px 0;">Top checklist</h2>
      <ol style="padding-left:20px; margin:0 0 16px 0;">
        {''.join(checklist_items) if checklist_items else '<li>No priority items were available.</li>'}
      </ol>
      <h2 style="font-size:18px; margin:20px 0 8px 0;">Today ahead</h2>
      <ul style="padding-left:20px; margin:0 0 16px 0;">
        {''.join(focus_items) if focus_items else '<li>No same-day focus items were available.</li>'}
      </ul>
      {"<h2 style='font-size:18px; margin:20px 0 8px 0;'>Hong Kong local checks</h2><ul style='padding-left:20px; margin:0 0 16px 0;'>" + ''.join(hk_local_items) + "</ul>" if hk_local_items else ""}
      <h2 style="font-size:18px; margin:20px 0 8px 0;">Suggested market answer</h2>
      <p style="margin:0 0 8px 0;">{interview_answer or 'Use the attached full report for a more detailed view.'}</p>
      <p style="margin:16px 0 0 0; color:#59636e;">The full markdown report and dashboard image are attached for desktop follow-up.</p>
    </div>
  </body>
</html>
"""
