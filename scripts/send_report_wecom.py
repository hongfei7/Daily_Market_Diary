"""
Send morning briefing to WeCom (企业微信) via webhook bot.

Modes:
  summary   Compact markdown summary (4096-byte limit) + GitHub link.       [default]
  file      Full report as a styled self-contained HTML file attachment.     [best for full display]
  full      Summary markdown message + HTML file.                           [recommended]

The HTML file embeds all chart images as base64 data URIs so it renders
completely offline when opened inside WeCom's built-in browser.
"""

from __future__ import annotations

import argparse
import base64
import html as html_lib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import mistune
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "market_diary"))

from professional.report_formatting import _fmt_pct, _fmt_price
from professional.instruments import format_summary_change


WECOM_MARKDOWN_BYTE_LIMIT = 4096
# Keep a safety margin for platform-side normalization and future copy changes.
WECOM_SAFE_MARKDOWN_BYTE_LIMIT = 3800
WECOM_FILE_SIZE_LIMIT = 20 * 1024 * 1024  # 20 MB
WECOM_IMAGE_SIZE_LIMIT = 2 * 1024 * 1024  # 2 MB


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send morning briefing to WeCom.")
    parser.add_argument("--report-date", required=True, help="Report date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default="reports_professional", help="Directory containing generated outputs.")
    parser.add_argument(
        "--mode",
        choices=("summary", "file", "full"),
        default="summary",
        help="Delivery mode: summary=markdown card, file=HTML attachment, full=both (default: summary)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Render/preview without sending.")
    parser.add_argument(
        "--receipt-file",
        default="",
        help="Optional JSON receipt path written only after WeCom confirms successful delivery.",
    )
    return parser.parse_args()


def _load_bundle(output_dir: Path, report_date: str) -> dict:
    bundle_path = output_dir / "raw" / f"{report_date}_bundle.json"
    with bundle_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_report_url(report_date: str) -> str:
    base = (os.getenv("WECOM_REPORT_BASE_URL") or "").strip().rstrip("/")
    if base:
        return f"{base}/{report_date}_morning_briefing.md"

    server = (os.getenv("GITHUB_SERVER_URL") or "https://github.com").strip()
    repo = (os.getenv("GITHUB_REPOSITORY") or "").strip()
    if repo:
        return f"{server}/{repo}/blob/main/reports_professional/archive/{report_date}/morning_briefing.md"

    return ""


# ---------------------------------------------------------------------------
# WeCom HTTP helpers
# ---------------------------------------------------------------------------


def _wecom_post(webhook_url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(webhook_url, json=payload, timeout=30)
    response.raise_for_status()
    result = response.json()
    if result.get("errcode") != 0:
        raise RuntimeError(f"WeCom API error: {result.get('errmsg', 'unknown')} (code {result.get('errcode')})")
    return result


def _wecom_upload(webhook_url: str, file_path: Path, media_type: str = "file") -> str:
    """Upload a file to WeCom and return the media_id."""
    # Extract key and build the upload URL from the webhook send URL
    if "key=" not in webhook_url:
        raise ValueError(f"Invalid WeCom webhook URL: missing 'key' parameter")
    key = webhook_url.split("key=")[-1].split("&")[0]  # strip trailing params

    upload_url = webhook_url.rstrip("/").replace("/send", "/upload_media")
    if "?" in upload_url:
        upload_url = upload_url.split("?")[0]
    upload_url = f"{upload_url}?key={key}&type={media_type}"

    file_size = file_path.stat().st_size
    if media_type == "image" and file_size > WECOM_IMAGE_SIZE_LIMIT:
        raise ValueError(f"Image size {file_size} exceeds WeCom 2MB limit")
    if media_type == "file" and file_size > WECOM_FILE_SIZE_LIMIT:
        raise ValueError(f"File size {file_size} exceeds WeCom 20MB limit")

    with file_path.open("rb") as handle:
        response = requests.post(upload_url, files={"media": (file_path.name, handle)}, timeout=60)
    response.raise_for_status()
    result = response.json()
    if result.get("errcode") != 0:
        raise RuntimeError(f"WeCom upload error: {result.get('errmsg', 'unknown')} (code {result.get('errcode')})")
    return result["media_id"]


def _write_delivery_receipt(
    path: Optional[Path],
    report_date: str,
    kind: str,
    response: Dict[str, Any],
) -> None:
    if path is None:
        return
    payload = {
        "status": "ok",
        "channel": "wecom",
        "kind": kind,
        "report_date": report_date,
        "sent_at_utc": datetime.now(timezone.utc).isoformat(),
        "response": {
            "errcode": response.get("errcode"),
            "errmsg": response.get("errmsg", ""),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Mode: summary (markdown card)
# ---------------------------------------------------------------------------


def _fmt_color_change(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.2f}%"


def _compact_text(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)].rstrip() + "…"


def _price_line(symbol: str, price: Any, change: Any, color_sign: bool = True) -> str:
    if price is None and change is None:
        return f"**{symbol}** N/A"
    p = _fmt_price(price, 2) if price is not None else "N/A"
    if color_sign and change is not None:
        try:
            c = _fmt_color_change(float(change))
        except (TypeError, ValueError):
            c = _fmt_pct(change)
    else:
        c = _fmt_pct(change) if change is not None else "N/A"
    return f"**{symbol}** {p} {c}"


def _local_display(bundle: Dict[str, Any], key: str) -> str:
    item = ((bundle.get("hk_local", {}) or {}).get(key, {}) or {})
    if not isinstance(item, dict):
        return "N/A"
    return str(item.get("display_value") or item.get("value") or "N/A")


def _market_snapshot_lines(bundle: Dict[str, Any]) -> List[str]:
    summary = bundle.get("market_summary", {}) or {}
    lines = []

    def _item(cat: str, name: str):
        return (summary.get(cat, {}) or {}).get(name, {}) or {}

    def _p(cat: str, name: str):
        return _item(cat, name).get("Price")

    def _c(cat: str, name: str):
        v = _item(cat, name).get("Pct Change")
        if isinstance(v, str):
            v = v.replace("%", "").strip()
        try:
            return float(v) if v is not None else None
        except (TypeError, ValueError):
            return None

    # Line 1: HSI + turnover + short sell
    hsi = _price_line("HSI", _p("Equities", "Hang Seng Index"), _c("Equities", "Hang Seng Index"))
    turnover_str = _local_display(bundle, "main_board_turnover")
    short_str = _local_display(bundle, "short_selling_ratio")
    lines.append(f"> {hsi} | Turnover {turnover_str} | SS {short_str}")

    # Line 2: US equities
    spx = _price_line("SPX", _p("Equities", "S&P 500"), _c("Equities", "S&P 500"))
    ndx = _price_line("NDX", _p("Equities", "Nasdaq 100"), _c("Equities", "Nasdaq 100"))
    hstech = _price_line("3033.HK ETF", _p("Equities", "Hang Seng TECH ETF"), _c("Equities", "Hang Seng TECH ETF"))
    lines.append(f"> {spx} | {ndx} | {hstech}")

    # Line 3: Rates / FX / Commodities
    parts = []
    us10y = _p("Rates", "10Y Treasury")
    if us10y is not None:
        rate_item = _item("Rates", "10Y Treasury")
        parts.append(f"**US10Y** {_fmt_price(us10y, 3)}% {format_summary_change(rate_item)}")
    dxy = _p("FX", "DXY")
    if dxy is not None:
        parts.append(f"**DXY** {_fmt_price(dxy, 1)}")
    cnh = _p("FX", "USD/CNH")
    if cnh is not None:
        parts.append(f"**CNH** {_fmt_price(cnh, 4)}")
    vix = _p("Vol", "VIX")
    if vix is not None:
        parts.append(f"**VIX** {_fmt_price(vix, 1)}")
    gold = _p("Commodities", "Gold")
    gold_c = _c("Commodities", "Gold")
    if gold is not None:
        parts.append(_price_line("Gold", gold, gold_c, color_sign=True))
    if parts:
        lines.append("> " + " | ".join(parts))

    return lines


def _hk_checks_line(bundle: Dict[str, Any], limit: int = 3) -> str:
    hk_checks = bundle.get("hk_quick_checks", []) or []
    parts = []
    count = 0
    for item in hk_checks:
        status = str(item.get("status", ""))
        if status in {"live_local", "stale_local", "live_hybrid", "live_public", "stale_public"}:
            metric = str(item.get("metric", ""))
            value = str(item.get("value", ""))
            if metric and value:
                parts.append(f"**{metric}** {value}")
                count += 1
                if count >= limit:
                    break
    return " | ".join(parts) if parts else ""


def _must_watch_lines(bundle: Dict[str, Any], limit: int = 3) -> List[str]:
    items = bundle.get("must_watch", []) or []
    colors = ["warning", "comment", "info"]
    lines = []
    for idx, item in enumerate(items[:limit]):
        color = colors[idx] if idx < len(colors) else "info"
        title = str(item.get("title", ""))
        summary = _compact_text(item.get("summary", ""), 80)
        lines.append(f'> <font color="{color}">[{idx + 1}] {title}</font>')
        if summary:
            lines.append(f"> {summary}")
    return lines


def _today_lines(bundle: Dict[str, Any], limit: int = 3) -> List[str]:
    today = bundle.get("today_forward", {}) or {}
    focus = today.get("focus_lines", []) or []
    if not focus:
        return []
    return [f"> {_compact_text(line, 100)}" for line in focus[:limit]]


def _append_group_with_budget(lines: List[str], group: List[str], footer: List[str]) -> None:
    """Append as much of a lower-priority group as fits without losing the footer."""
    if not group:
        return
    candidate = "\n".join([*lines, *group, *footer])
    if len(candidate.encode("utf-8")) <= WECOM_SAFE_MARKDOWN_BYTE_LIMIT:
        lines.extend(group)
        return

    header, *items = group
    selected: List[str] = []
    for line in items:
        candidate = "\n".join([*lines, header, *selected, line, *footer])
        if len(candidate.encode("utf-8")) <= WECOM_SAFE_MARKDOWN_BYTE_LIMIT:
            selected.append(line)
    if selected:
        lines.extend([header, *selected])


def build_summary_markdown(bundle: Dict[str, Any], report_date: str) -> str:
    """Build a compact WeCom markdown message within the 4096-byte limit."""
    meta = bundle.get("meta", {}) or {}
    overview = bundle.get("overview", {}) or {}
    llm = bundle.get("llm_sections", {}) or {}
    quality = bundle.get("report_quality", {}) or {}
    release = quality.get("release_recommendation", {}) or {}

    briefing_date = meta.get("briefing_date", report_date)
    global_date = meta.get("global_market_date", "")
    hk_date = meta.get("hk_data_date", "")
    risk = overview.get("risk_regime", "N/A")
    score = quality.get("score", "N/A")
    grade = quality.get("grade", "N/A")
    pulse = llm.get("one_line_market_pulse") or overview.get("theme", "")
    hk_view = bundle.get("hk_desk_view", {}) or {}
    hk_lens = hk_view.get("lens") or llm.get("hk_local_leadership") or hk_view.get("leadership") or ""

    report_url = _resolve_report_url(report_date)

    risk_check = hk_view.get("invalidation") or llm.get("risk_check") or "Reassess if rates, CNH, or Hong Kong breadth contradict the base case."
    confirmation = hk_view.get("confirmation") or ""
    if not confirmation:
        confirmation_lines = (bundle.get("today_forward", {}) or {}).get("focus_lines", []) or []
        confirmation = confirmation_lines[0] if confirmation_lines else ""
    report_link = f"[Open full report]({report_url})" if report_url else "Full report attachment follows."
    footer = ["", report_link]

    lines = [
        f"# HK Morning Brief | {briefing_date}",
        "> **5-minute scan** | Full report linked below.",
    ]

    if pulse:
        lines.append(f"> {_compact_text(pulse, 120)}")
    lines.append(f"> **Risk**: {risk} | **Quality**: {score}/100 ({grade})")

    if release:
        label = release.get("label", "")
        reason = release.get("reason", "")
        if label:
            tag = f"**Release**: {label}"
            if reason:
                tag += f" ({_compact_text(reason, 60)})"
            lines.append(f"> {tag}")

    date_parts = []
    if global_date:
        date_parts.append(f"**Global** {global_date}")
    if hk_date:
        date_parts.append(f"**HK** {hk_date}")
    if date_parts:
        lines.append(" | ".join(date_parts))

    lines.append("## Decision frame")
    if hk_lens:
        lines.append(f"> **HK lens:** {_compact_text(hk_lens, 300)}")
    if confirmation:
        lines.append(f"> **Confirm:** {_compact_text(confirmation, 160)}")
    else:
        lines.append("> **Confirm:** Watch Hong Kong breadth, CNH and local flow for same-day confirmation.")
    lines.append(f"> **Invalidate:** {_compact_text(risk_check, 110)}")

    _append_group_with_budget(lines, ["## Markets", *_market_snapshot_lines(bundle)], footer)

    mw_lines = _must_watch_lines(bundle)
    if mw_lines:
        _append_group_with_budget(lines, ["## Priority", *mw_lines], footer)

    today = _today_lines(bundle)
    if today:
        _append_group_with_budget(lines, ["## Today", *today], footer)

    hk_line = _hk_checks_line(bundle)
    if hk_line:
        _append_group_with_budget(lines, ["## Local checks", hk_line], footer)

    body = "\n".join([*lines, *footer])
    if len(body.encode("utf-8")) > WECOM_SAFE_MARKDOWN_BYTE_LIMIT:
        raise ValueError("Required WeCom decision summary exceeds the safe markdown byte budget.")

    return body


def send_summary(
    webhook_url: str,
    bundle: Dict[str, Any],
    report_date: str,
    dry_run: bool = False,
    receipt_path: Optional[Path] = None,
) -> str:
    """Send markdown summary. Returns the markdown text."""
    markdown = build_summary_markdown(bundle, report_date)
    if dry_run:
        print("=== WeCom Markdown Message ===")
        print(markdown)
        print(
            f"=== Byte count: {len(markdown.encode('utf-8'))} / "
            f"{WECOM_SAFE_MARKDOWN_BYTE_LIMIT} safe ({WECOM_MARKDOWN_BYTE_LIMIT} platform) ==="
        )
    else:
        result = _wecom_post(webhook_url, {"msgtype": "markdown", "markdown": {"content": markdown}})
        _write_delivery_receipt(receipt_path, report_date, "summary", result)
        print("WeCom markdown summary sent.")
    return markdown


# ---------------------------------------------------------------------------
# Mode: file (HTML report attachment)
# ---------------------------------------------------------------------------

HTML_CSS = """\
<style>
  :root {
    --ink: #101114;
    --navy: #18364a;
    --blue: #176b92;
    --muted: #5f666b;
    --line: #d9dcdd;
    --paper: #ffffff;
    --warm: #f4f2ed;
    --soft: #f6f7f7;
    --supportive: #176b92;
    --adverse: #9a5a21;
  }
  * { box-sizing: border-box; }
  html { background: #efefec; scroll-behavior: smooth; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Arial,
                 sans-serif;
    color: var(--ink); line-height: 1.62; max-width: 1360px; margin: 0 auto;
    padding: 0 44px 64px; background: var(--paper); font-size: 15px;
    -webkit-font-smoothing: antialiased; overflow-x: hidden;
  }
  h1, h2, h3, h4, h5, h6 { color: var(--ink); letter-spacing: -.025em; }
  h1 { font-size: 32px; line-height: 1.12; margin: 42px 0 18px; }
  h2 { font-size: 30px; line-height: 1.14; margin: 0 0 26px; padding-top: 18px; border-top: 5px solid var(--ink); }
  h3 { font-size: 21px; line-height: 1.26; margin: 38px 0 14px; padding-left: 12px; border-left: 3px solid var(--blue); }
  h4 { font-size: 16px; margin: 28px 0 10px; color: var(--navy); letter-spacing: -.01em; }
  h5, h6 { font-size: 15px; margin: 22px 0 8px; }
  p { margin: 0 0 13px; }
  blockquote {
    border-left: 3px solid var(--blue); padding: 13px 17px; margin: 18px 0 24px;
    background: var(--soft); color: #374047; font-size: 14px;
  }
  blockquote p:last-child { margin-bottom: 0; }
  .report-shell { border-top: 9px solid var(--ink); }
  .report-masthead { padding: 24px 0 34px; border-bottom: 1px solid var(--line); }
  .masthead-line {
    display: flex; justify-content: space-between; gap: 20px; align-items: baseline;
    padding-bottom: 19px; border-bottom: 1px solid var(--ink);
  }
  .report-wordmark, .issue-label {
    margin: 0; font-size: 10px; font-weight: 800; letter-spacing: .18em; text-transform: uppercase;
  }
  .report-wordmark { color: var(--blue); }
  .issue-label { color: var(--muted); letter-spacing: .11em; }
  .masthead-grid { display: grid; grid-template-columns: minmax(0, 1fr) 260px; gap: 54px; padding-top: 34px; }
  .report-eyebrow { margin: 0 0 11px; color: var(--blue); font-size: 11px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
  .report-masthead h1 { margin: 0; max-width: 820px; font-size: 48px; font-weight: 720; line-height: 1.02; }
  .report-deck {
    max-width: 900px; margin: 24px 0 0; font-size: 24px; font-weight: 500;
    line-height: 1.38; letter-spacing: -.02em; color: #2e3438;
  }
  .issue-facts { margin: 2px 0 0; padding: 0 0 0 20px; border-left: 1px solid var(--line); }
  .issue-facts div { padding: 0 0 14px; margin: 0 0 14px; border-bottom: 1px solid var(--line); }
  .issue-facts div:last-child { margin-bottom: 0; border-bottom: 0; }
  .issue-facts dt { color: var(--muted); font-size: 9px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
  .issue-facts dd { margin: 4px 0 0; color: var(--ink); font-size: 13px; font-weight: 700; line-height: 1.35; }
  .reading-path { display: grid; grid-template-columns: repeat(3, 1fr); gap: 0; margin: 31px 0 0; padding: 0; list-style: none; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .reading-path li { display: grid; grid-template-columns: 30px 1fr; gap: 10px; margin: 0; padding: 14px 18px 14px 0; border-right: 1px solid var(--line); }
  .reading-path li:last-child { border-right: 0; padding-left: 18px; }
  .reading-path li:nth-child(2) { padding-left: 18px; }
  .reading-path b { color: var(--blue); font-size: 11px; }
  .reading-path span { display: block; font-size: 12px; font-weight: 750; line-height: 1.2; }
  .reading-path small { display: block; margin-top: 3px; color: var(--muted); font-size: 10px; }
  .report-grid { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 58px; align-items: start; padding-top: 34px; }
  .report-toc { position: sticky; top: 18px; padding-top: 8px; }
  .report-toc-title { padding-bottom: 9px; border-bottom: 2px solid var(--ink); font-size: 9px; font-weight: 800; color: var(--ink); letter-spacing: .14em; text-transform: uppercase; }
  .report-toc a { display: block; margin-top: 11px; color: #4e555a; font-size: 11px; line-height: 1.35; text-decoration: none; }
  .report-toc a:hover { color: var(--blue); }
  .mobile-toc { display: none; }
  .report-content { min-width: 0; max-width: 990px; }
  .report-content > h1 { display: none; }
  .report-section { margin: 0 0 58px; scroll-margin-top: 18px; }
  .section-executive-summary > ul {
    display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 18px;
    padding: 0; margin: 0; list-style: none;
  }
  .section-executive-summary > ul > li {
    margin: 0; padding: 14px 0 0; border-top: 3px solid var(--blue); color: #353c41;
  }
  .section-executive-summary > ul > li strong { display: block; margin-bottom: 7px; color: var(--ink); font-size: 11px; letter-spacing: .05em; text-transform: uppercase; }
  .section-visual-dashboard { padding: 24px 26px 2px; background: var(--warm); }
  .section-visual-dashboard h2 { border-top-color: var(--blue); }
  .section-optional-appendix-traceability-and-performance,
  .section-traceable-appendix,
  .section-supplementary-visual-appendix { padding: 28px 30px; background: var(--soft); }
  .table-shell { width: 100%; margin: 18px 0 26px; overflow-x: auto; border-top: 2px solid var(--ink); -webkit-overflow-scrolling: touch; }
  table { width: 100%; border-collapse: collapse; margin: 0; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  th { background: var(--paper); color: var(--ink); padding: 10px 10px; text-align: left; font-size: 10px; letter-spacing: .055em; text-transform: uppercase; font-weight: 800; border-bottom: 1px solid #9ea4a7; vertical-align: bottom; }
  td { padding: 10px; border-bottom: 1px solid #e3e4e4; vertical-align: top; }
  tbody tr:hover td { background: #f7f7f5; }
  th:first-child, td:first-child { font-weight: 700; }
  th:nth-child(2), td:nth-child(2), th:nth-child(3), td:nth-child(3) { font-variant-numeric: tabular-nums; }
  .company-event-monitor { margin: 20px 0 30px; }
  .event-monitor-summary {
    display: grid; grid-template-columns: minmax(0, 1fr) auto; gap: 32px; align-items: center;
    padding: 20px 22px; background: var(--warm); border-top: 3px solid var(--navy); border-bottom: 1px solid var(--line);
  }
  .event-summary-copy { min-width: 0; }
  .event-kicker { display: block; margin-bottom: 5px; color: var(--blue); font-size: 9px; font-weight: 800; letter-spacing: .14em; text-transform: uppercase; }
  .event-summary-copy h4 { margin: 0; color: var(--ink); font-size: 17px; line-height: 1.34; }
  .event-summary-copy p { margin: 7px 0 0; color: var(--muted); font-size: 11px; }
  .event-stats { display: grid; grid-template-columns: repeat(3, 84px); border-left: 1px solid #cfd1d0; }
  .event-stats div { min-width: 0; padding: 3px 12px; border-right: 1px solid #cfd1d0; }
  .event-stats strong { display: block; font-size: 22px; line-height: 1; font-variant-numeric: tabular-nums; }
  .event-stats span { display: block; margin-top: 6px; color: var(--muted); font-size: 8px; font-weight: 750; letter-spacing: .07em; line-height: 1.25; text-transform: uppercase; }
  .event-card-list { display: grid; gap: 12px; margin-top: 14px; }
  .event-card { position: relative; padding: 16px 18px 15px 20px; border: 1px solid var(--line); border-left: 4px solid #8b9398; background: var(--paper); }
  .event-card.priority-portfolio { border-left-color: #8d2e24; }
  .event-card.priority-high { border-left-color: var(--adverse); }
  .event-card.priority-review { border-left-color: var(--blue); }
  .event-card-meta { display: flex; flex-wrap: wrap; gap: 7px 13px; align-items: center; color: var(--muted); font-size: 9px; font-weight: 700; letter-spacing: .055em; text-transform: uppercase; }
  .event-card-meta time { margin-left: auto; font-variant-numeric: tabular-nums; }
  .event-priority { padding: 2px 7px; border: 1px solid currentColor; color: var(--navy); }
  .priority-portfolio .event-priority { color: #8d2e24; }
  .priority-high .event-priority { color: var(--adverse); }
  .priority-review .event-priority { color: var(--blue); }
  .event-card h5 { margin: 10px 0 7px; font-size: 16px; line-height: 1.25; }
  .event-fact { margin: 0; color: #252b2f; font-size: 13px; font-weight: 600; line-height: 1.48; }
  .event-drivers { margin: 8px 0 0; color: #4f585e; font-size: 11px; line-height: 1.5; }
  .event-drivers span { margin-right: 7px; color: var(--ink); font-size: 8px; font-weight: 800; letter-spacing: .08em; text-transform: uppercase; }
  .event-read-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin-top: 13px; padding-top: 12px; border-top: 1px solid #e2e3e2; }
  .event-read-grid span { display: block; margin-bottom: 4px; color: var(--muted); font-size: 8px; font-weight: 800; letter-spacing: .09em; text-transform: uppercase; }
  .event-read-grid p { margin: 0; color: #3b4348; font-size: 11px; line-height: 1.45; }
  .event-source { display: inline-block; margin-top: 11px; font-size: 10px; font-weight: 750; text-decoration: none; }
  .event-source-muted { color: var(--muted); }
  .event-monitor-empty { padding: 18px 20px; border: 1px solid var(--line); border-left: 4px solid #8b9398; background: var(--soft); }
  .event-monitor-empty strong { font-size: 13px; }
  .event-monitor-empty p { margin: 5px 0 0; color: var(--muted); font-size: 11px; }
  .event-coverage-note { margin: 12px 0 0; padding: 10px 12px; border-left: 2px solid #9da4a7; color: var(--muted); background: #fafafa; font-size: 10.5px; line-height: 1.45; }
  .move-positive { color: var(--supportive); font-weight: 750; white-space: nowrap; }
  .move-negative { color: var(--adverse); font-weight: 750; white-space: nowrap; }
  img { display: block; max-width: 100%; height: auto; margin: 22px 0 30px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .section-visual-dashboard img { margin-top: 16px; border: 0; }
  code { background: #f0f1f1; padding: 2px 5px; font-family: "SF Mono", "Fira Code", "Consolas", monospace; font-size: 12px; }
  pre { background: var(--soft); padding: 14px; overflow-x: auto; font-size: 12px; line-height: 1.5; border-left: 3px solid var(--navy); }
  pre code { background: none; padding: 0; }
  ul, ol { padding-left: 22px; margin: 8px 0 17px; }
  li { margin: 6px 0; }
  strong { color: var(--ink); }
  a { color: var(--blue); text-decoration-thickness: 1px; text-underline-offset: 2px; }
  hr { border: none; border-top: 1px solid var(--line); margin: 34px 0; }
  .report-footer { margin-top: 22px; padding: 23px 0 8px; border-top: 5px solid var(--ink); font-size: 11px; color: var(--muted); display: flex; justify-content: space-between; gap: 18px; }
  @media (max-width: 860px) {
    body { padding: 0 18px 38px; font-size: 14px; }
    .report-shell { border-top-width: 6px; }
    .report-masthead { padding: 17px 0 24px; }
    .masthead-line { padding-bottom: 13px; }
    .report-wordmark, .issue-label { font-size: 8px; }
    .masthead-grid { display: block; padding-top: 24px; }
    .report-masthead h1 { font-size: 36px; }
    .report-deck { margin-top: 18px; font-size: 19px; line-height: 1.42; }
    .issue-facts { display: grid; grid-template-columns: 1fr 1fr; gap: 0 18px; margin-top: 24px; padding: 0; border-left: 0; border-top: 1px solid var(--line); }
    .issue-facts div { padding-top: 11px; margin-bottom: 0; }
    .reading-path { margin-top: 22px; }
    .reading-path li { display: block; padding: 11px 8px 11px 0; }
    .reading-path li:nth-child(2), .reading-path li:last-child { padding-left: 10px; }
    .reading-path b { display: none; }
    .reading-path span { font-size: 10px; }
    .reading-path small { font-size: 9px; }
    .report-grid { display: block; padding-top: 20px; }
    .report-toc { display: none; }
    .mobile-toc { display: block; margin: 0 0 30px; border-top: 1px solid var(--ink); border-bottom: 1px solid var(--line); }
    .mobile-toc summary { padding: 12px 0; cursor: pointer; color: var(--ink); font-size: 10px; font-weight: 800; letter-spacing: .12em; text-transform: uppercase; }
    .mobile-toc a { display: block; padding: 8px 0; color: #4e555a; font-size: 12px; text-decoration: none; }
    .mobile-toc nav { padding: 0 0 12px; }
    .report-section { margin-bottom: 45px; }
    h2 { font-size: 25px; margin-bottom: 21px; padding-top: 14px; border-top-width: 4px; }
    h3 { font-size: 18px; margin-top: 31px; }
    .section-executive-summary > ul { display: block; }
    .section-executive-summary > ul > li { margin-bottom: 20px; }
    .section-visual-dashboard,
    .section-optional-appendix-traceability-and-performance,
    .section-traceable-appendix,
    .section-supplementary-visual-appendix { margin-left: -18px; margin-right: -18px; padding: 24px 18px 2px; }
    table { min-width: 660px; font-size: 12px; }
    th, td { padding: 9px 8px; }
    .event-monitor-summary { display: block; padding: 17px 16px; }
    .event-summary-copy h4 { font-size: 15px; }
    .event-stats { grid-template-columns: repeat(3, 1fr); margin-top: 15px; padding-top: 13px; border-top: 1px solid #cfd1d0; border-left: 0; }
    .event-stats div { padding: 0 9px; }
    .event-stats div:first-child { padding-left: 0; }
    .event-stats div:last-child { padding-right: 0; border-right: 0; }
    .event-stats strong { font-size: 19px; }
    .event-card { padding: 15px 14px 14px 16px; }
    .event-card-meta time { width: 100%; margin-left: 0; }
    .event-read-grid { display: block; }
    .event-read-grid div + div { margin-top: 10px; }
    .report-footer { display: block; }
    .report-footer span { display: block; margin-top: 5px; }
  }
  @media (max-width: 430px) {
    body { padding-left: 14px; padding-right: 14px; }
    .report-masthead h1 { font-size: 32px; }
    .report-deck { font-size: 18px; }
    .section-visual-dashboard,
    .section-optional-appendix-traceability-and-performance,
    .section-traceable-appendix,
    .section-supplementary-visual-appendix { margin-left: -14px; margin-right: -14px; padding-left: 14px; padding-right: 14px; }
  }
  @media print {
    html { background: #fff; }
    body { max-width: none; padding: 0; font-size: 10.5pt; }
    .report-toc, .mobile-toc { display: none; }
    .report-grid { display: block; }
    .report-content { max-width: none; }
    .table-shell { overflow: visible; break-inside: avoid; }
    table { min-width: 0; }
    h2, h3, img { break-after: avoid; }
    .report-section { break-inside: auto; }
  }
</style>"""


def _structure_report_html(body_html: str) -> tuple[str, List[tuple[str, str]]]:
    seen: Dict[str, int] = {}
    toc: List[tuple[str, str]] = []

    def _heading(match: re.Match) -> str:
        level = match.group("level")
        inner = match.group("inner")
        label = re.sub(r"<[^>]+>", "", inner)
        slug = re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "section"
        count = seen.get(slug, 0) + 1
        seen[slug] = count
        if count > 1:
            slug = f"{slug}-{count}"
        if level == "2":
            toc.append((slug, label))
        return f'<h{level} id="{slug}">{inner}</h{level}>'

    body_html = re.sub(
        r'<h(?P<level>[2-3])>(?P<inner>.*?)</h(?P=level)>',
        _heading,
        body_html,
        flags=re.DOTALL,
    )
    body_html = re.sub(r"<table>(.*?)</table>", r'<div class="table-shell"><table>\1</table></div>', body_html, flags=re.DOTALL)

    def _movement_cell(match: re.Match) -> str:
        inner = match.group("inner")
        movement = re.search(
            r"(?<![0-9])(?P<strong><strong>)?(?P<move>[+-][0-9][0-9,.]*(?:\.[0-9]+)?(?:%|bp|bn|mn)?)(?P<close></strong>)?",
            inner,
        )
        if not movement:
            return match.group(0)
        css_class = "move-positive" if movement.group("move").startswith("+") else "move-negative"
        decorated = (
            inner[: movement.start()]
            + f'<span class="{css_class}">'
            + movement.group(0)
            + "</span>"
            + inner[movement.end() :]
        )
        return f"<td>{decorated}</td>"

    body_html = re.sub(r"<td>(?P<inner>.*?)</td>", _movement_cell, body_html, flags=re.DOTALL)

    # The report title and commute metadata are represented by the designed
    # masthead below; remove their duplicated Markdown rendering from the body.
    body_html = re.sub(
        r'(?P<title><h1>.*?</h1>)\s*(?:<blockquote>.*?</blockquote>)?\s*'
        r'(?:<p><em>Data through:.*?</em></p>)?',
        r'\g<title>',
        body_html,
        count=1,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Wrap each H2-led chapter so hierarchy, appendix treatment, and mobile
    # spacing can be controlled without changing the underlying Markdown.
    matches = list(re.finditer(r'<h2 id="(?P<slug>[^"]+)">.*?</h2>', body_html, flags=re.DOTALL))
    if matches:
        pieces = [body_html[: matches[0].start()]]
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(body_html)
            slug = match.group("slug")
            pieces.append(f'<section class="report-section section-{slug}">{body_html[match.start():end]}</section>')
        body_html = "".join(pieces)
    return body_html, toc


def _extract_header_context(md_text: str, report_date: str) -> Dict[str, str]:
    def _match(pattern: str, default: str = "") -> str:
        found = re.search(pattern, md_text, flags=re.MULTILINE | re.IGNORECASE)
        return " ".join(found.group(1).split()).strip() if found else default

    pulse = _match(
        r"^- \*\*Market pulse:\*\*\s*(.+)$",
        "Evidence-led Hong Kong market briefing and decision checklist.",
    )
    mode = _match(r"Mode:\s*`([^`]+)`", "Daily briefing")
    global_date = _match(
        r"Data through:\s*global\s*`([^`]+)`",
        _match(r"Global request:\s*`([^`]+)`", "See report"),
    )
    hk_date = _match(
        r"HK/China\s*`([^`]+)`",
        _match(r"HK/China request:\s*`([^`]+)`", "See report"),
    )
    quality = _match(r"Report quality:\s*`([^`]+)`", "See validation")
    return {
        "pulse": pulse,
        "mode": mode,
        "global_date": global_date,
        "hk_date": hk_date,
        "quality": quality,
        "report_date": report_date,
    }


def _md_to_html(md_text: str, output_dir: Path, report_date: str, md_source_dir: Optional[Path] = None) -> str:
    """Convert markdown report to a self-contained HTML document with embedded images."""
    # Convert MD to HTML body
    body_html = mistune.html(md_text)
    body_html, toc = _structure_report_html(body_html)

    # Collect search dirs for resolving relative image paths
    search_dirs = [output_dir]
    if md_source_dir is not None:
        search_dirs.insert(0, md_source_dir)

    def _resolve_img_path(src: str) -> Optional[Path]:
        src_path = Path(src)
        if src_path.is_absolute():
            return src_path if src_path.exists() else None
        for base in search_dirs:
            candidate = base / src
            if candidate.exists():
                return candidate
        return None

    # Replace local image paths with base64 data URIs
    def _replace_img(match: re.Match) -> str:
        src = match.group("src")
        alt = match.group("alt") or ""
        img_path = _resolve_img_path(src)
        if img_path is None:
            print(f"  [WARN] Image not found: {src} (searched: {[str(d) for d in search_dirs]})")
            return match.group(0)

        try:
            suffix = img_path.suffix.lower()
            mime = {
                ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
                ".gif": "image/gif", ".webp": "image/webp", ".svg": "image/svg+xml",
            }.get(suffix, "image/png")

            encoded = base64.b64encode(img_path.read_bytes()).decode()
            return f'<img src="data:{mime};base64,{encoded}" alt="{alt}" />'
        except Exception:
            return match.group(0)

    # Match img tags — mistune outputs <img src="..." alt="..." /> with src before alt
    body_html = re.sub(
        r'<img\s+src="(?P<src>[^"]+)"\s+alt="(?P<alt>[^"]*)"[^>]*/?>',
        _replace_img,
        body_html,
    )

    title = f"HK Morning Market Brief | {report_date}"
    header = _extract_header_context(md_text, report_date)
    pulse = html_lib.escape(header["pulse"])
    toc_html = "".join(f'<a href="#{slug}">{html_lib.escape(label)}</a>' for slug, label in toc)

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{HTML_CSS}
</head>
<body>
<div class="report-shell">
  <header class="report-masthead">
    <div class="masthead-line">
      <p class="report-wordmark">Hong Kong Market Intelligence</p>
      <p class="issue-label">Daily research note · {report_date}</p>
    </div>
    <div class="masthead-grid">
      <div>
        <p class="report-eyebrow">Decision brief · Source-audited</p>
        <h1>Morning Market Brief</h1>
        <p class="report-deck">{pulse}</p>
      </div>
      <dl class="issue-facts">
        <div><dt>Edition</dt><dd>{html_lib.escape(header['mode'])}</dd></div>
        <div><dt>Global through</dt><dd>{html_lib.escape(header['global_date'])}</dd></div>
        <div><dt>HK / China through</dt><dd>{html_lib.escape(header['hk_date'])}</dd></div>
        <div><dt>Report quality</dt><dd>{html_lib.escape(header['quality'])}</dd></div>
      </dl>
    </div>
    <ol class="reading-path" aria-label="Commute reading path">
      <li><b>01</b><div><span>Decision scan</span><small>5 minutes</small></div></li>
      <li><b>02</b><div><span>Causal deep read</span><small>25–30 minutes</small></div></li>
      <li><b>03</b><div><span>Evidence appendix</span><small>Optional 10–15 minutes</small></div></li>
    </ol>
  </header>
  <div class="report-grid">
    <nav class="report-toc" aria-label="Report sections">
      <div class="report-toc-title">In this issue</div>
      {toc_html}
    </nav>
    <main class="report-content">
      <details class="mobile-toc">
        <summary>In this issue</summary>
        <nav aria-label="Mobile report sections">{toc_html}</nav>
      </details>
      {body_html}
    </main>
  </div>
  <div class="report-footer">
    <strong>Morning Market Brief</strong>
    <span>Generated {report_date} · For research review, not investment advice</span>
  </div>
</div>
</body>
</html>"""


def send_file(
    webhook_url: str,
    output_dir: Path,
    report_date: str,
    dry_run: bool = False,
    receipt_path: Optional[Path] = None,
) -> str:
    """Convert markdown report to self-contained HTML, upload to WeCom, and send as file."""
    # Try date-prefixed filename first, then archive layout, then plain morning_briefing.md
    md_path = output_dir / f"{report_date}_morning_briefing.md"
    if not md_path.exists():
        md_path = output_dir / "archive" / report_date / "morning_briefing.md"
    if not md_path.exists():
        md_path = output_dir / "morning_briefing.md"
    if not md_path.exists():
        raise FileNotFoundError(f"Report file not found in {output_dir}")

    md_text = md_path.read_text(encoding="utf-8")
    html = _md_to_html(md_text, output_dir, report_date, md_source_dir=md_path.parent)

    # Write HTML to a temp file for upload
    html_filename = f"{report_date}_morning_briefing.html"
    tmp_path = output_dir / html_filename
    tmp_path.write_text(html, encoding="utf-8")

    if dry_run:
        print(f"=== HTML file written to: {tmp_path} ===")
        print(f"=== File size: {tmp_path.stat().st_size:,} bytes ===")
        return html

    try:
        print(f"Uploading {html_filename} ({tmp_path.stat().st_size:,} bytes)...")
        media_id = _wecom_upload(webhook_url, tmp_path, media_type="file")
        print(f"Uploaded. media_id={media_id}")

        result = _wecom_post(webhook_url, {"msgtype": "file", "file": {"media_id": media_id}})
        _write_delivery_receipt(receipt_path, report_date, "file", result)
        print("WeCom file message sent.")
    finally:
        # Clean up temp file (keep it on dry_run for inspection)
        if tmp_path.exists():
            tmp_path.unlink()

    return html


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    args = _parse_args()
    output_dir = (ROOT / args.output_dir).resolve()

    webhook_url = (os.getenv("WECOM_WEBHOOK_URL") or "").strip()
    receipt_path = Path(args.receipt_file).resolve() if args.receipt_file else None
    if not webhook_url and not args.dry_run:
        print("WECOM_WEBHOOK_URL is not set; primary WeCom delivery cannot proceed.", file=sys.stderr)
        return 1

    # Only load bundle for modes that need it (not file-only mode)
    bundle = None
    if args.mode in ("summary", "full"):
        bundle = _load_bundle(output_dir, args.report_date)

    if args.mode == "summary":
        markdown = send_summary(
            webhook_url,
            bundle,
            args.report_date,
            dry_run=args.dry_run,
            receipt_path=receipt_path,
        )
        if args.dry_run:
            (output_dir / f"{args.report_date}_wecom_preview.md").write_text(markdown, encoding="utf-8")
    elif args.mode == "file":
        send_file(
            webhook_url,
            output_dir,
            args.report_date,
            dry_run=args.dry_run,
            receipt_path=receipt_path,
        )
    elif args.mode == "full":
        markdown = send_summary(webhook_url, bundle, args.report_date, dry_run=args.dry_run)
        if args.dry_run:
            (output_dir / f"{args.report_date}_wecom_preview.md").write_text(markdown, encoding="utf-8")
        send_file(webhook_url, output_dir, args.report_date, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
