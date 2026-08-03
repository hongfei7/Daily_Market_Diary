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
from pathlib import Path
from typing import Any, Dict, List, Optional

import mistune
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "market_diary"))

from professional.report_formatting import _truncate, _fmt_pct, _fmt_price
from professional.instruments import format_summary_change


WECOM_MARKDOWN_BYTE_LIMIT = 4096
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
        return f"{server}/{repo}/blob/main/reports_professional/{report_date}_morning_briefing.md"

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


# ---------------------------------------------------------------------------
# Mode: summary (markdown card)
# ---------------------------------------------------------------------------


def _fmt_color_change(value: Optional[float]) -> str:
    if value is None:
        return "N/A"
    text = f"{value:+.2f}%"
    if value >= 1.0:
        return f'<font color="warning">{text}</font>'
    if value <= -1.0:
        return f'<font color="info">{text}</font>'
    return text


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
    turnover_val = _item("Equities", "Hang Seng Index").get("Volume")
    turnover_str = f"HK${float(turnover_val)/1e9:.1f}bn" if turnover_val is not None else "N/A"
    short_sell_val = _item("Equities", "Hang Seng Index").get("Short Sell Ratio")
    if short_sell_val is not None:
        try:
            ss = float(short_sell_val)
            short_str = f"{ss:.1f}%"
        except (TypeError, ValueError):
            short_str = str(short_sell_val)
    else:
        short_str = "N/A"
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


def _hk_checks_line(bundle: Dict[str, Any]) -> str:
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
                if count >= 4:
                    break
    return " | ".join(parts) if parts else ""


def _must_watch_lines(bundle: Dict[str, Any], limit: int = 3) -> List[str]:
    items = bundle.get("must_watch", []) or []
    colors = ["warning", "comment", "info"]
    lines = []
    for idx, item in enumerate(items[:limit]):
        color = colors[idx] if idx < len(colors) else "info"
        title = str(item.get("title", ""))
        summary = _truncate(str(item.get("summary", "")), 80)
        lines.append(f'> <font color="{color}">[{idx + 1}] {title}</font>')
        if summary:
            lines.append(f"> {summary}")
    return lines


def _today_lines(bundle: Dict[str, Any], limit: int = 3) -> List[str]:
    today = bundle.get("today_forward", {}) or {}
    focus = today.get("focus_lines", []) or []
    if not focus:
        return []
    return [f"> {_truncate(str(line), 100)}" for line in focus[:limit]]


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

    report_url = _resolve_report_url(report_date)

    lines = [f"# HK Morning Brief | {briefing_date}"]

    if pulse:
        lines.append(f"> {_truncate(str(pulse), 120)}")
    lines.append(f"> **Risk**: {risk} | **Quality**: {score}/{grade}")

    if release:
        label = release.get("label", "")
        reason = release.get("reason", "")
        if label:
            tag = f"**Release**: {label}"
            if reason:
                tag += f" ({_truncate(str(reason), 60)})"
            lines.append(f"> {tag}")

    date_parts = []
    if global_date:
        date_parts.append(f"**Global** {global_date}")
    if hk_date:
        date_parts.append(f"**HK** {hk_date}")
    if date_parts:
        lines.append(" | ".join(date_parts))

    lines.append("## Markets")
    lines.extend(_market_snapshot_lines(bundle))

    mw_lines = _must_watch_lines(bundle)
    if mw_lines:
        lines.append("## Priority")
        lines.extend(mw_lines)

    today = _today_lines(bundle)
    if today:
        lines.append("## Today")
        lines.extend(today)

    hk_line = _hk_checks_line(bundle)
    if hk_line:
        lines.append(hk_line)

    if report_url:
        lines.append("")
        lines.append(f"[Full Report]({report_url})")

    body = "\n".join(lines)

    # Trim to fit WeCom's 4096-byte limit
    while len(body.encode("utf-8")) > WECOM_MARKDOWN_BYTE_LIMIT:
        body = "\n".join(body.split("\n")[:-1])
        if not body:
            body = "Report too large for WeCom message."
            break

    return body


def send_summary(webhook_url: str, bundle: Dict[str, Any], report_date: str, dry_run: bool = False) -> str:
    """Send markdown summary. Returns the markdown text."""
    markdown = build_summary_markdown(bundle, report_date)
    if dry_run:
        print("=== WeCom Markdown Message ===")
        print(markdown)
        print(f"=== Byte count: {len(markdown.encode('utf-8'))} / {WECOM_MARKDOWN_BYTE_LIMIT} ===")
    else:
        _wecom_post(webhook_url, {"msgtype": "markdown", "markdown": {"content": markdown}})
        print("WeCom markdown summary sent.")
    return markdown


# ---------------------------------------------------------------------------
# Mode: file (HTML report attachment)
# ---------------------------------------------------------------------------

HTML_CSS = """\
<style>
  :root {
    --ink: #111820;
    --navy: #123a56;
    --blue: #1f5f8b;
    --muted: #58656f;
    --line: #d8dde1;
    --soft: #f4f6f7;
    --positive: #1f5f8b;
    --negative: #b54708;
  }
  * { box-sizing: border-box; }
  html { background: #eef1f2; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Arial,
                 sans-serif;
    color: var(--ink); line-height: 1.58; max-width: 1240px; margin: 0 auto;
    padding: 0 30px 56px; background: #fff; font-size: 15px;
  }
  h1, h2, h3, h4, h5, h6 { color: var(--ink); letter-spacing: -0.015em; }
  h1 { font-size: 30px; line-height: 1.15; margin: 44px 0 20px; }
  h2 { font-size: 24px; line-height: 1.2; margin: 54px 0 20px; padding-top: 14px; border-top: 3px solid var(--navy); }
  h3 { font-size: 19px; line-height: 1.3; margin: 34px 0 13px; }
  h4 { font-size: 16px; margin: 26px 0 10px; color: var(--navy); }
  h5, h6 { font-size: 15px; margin: 22px 0 8px; }
  p { margin: 0 0 13px; }
  blockquote {
    border-left: 4px solid var(--navy); padding: 11px 16px; margin: 16px 0 22px;
    background: var(--soft); color: #3d4952; font-size: 14px;
  }
  blockquote p:last-child { margin-bottom: 0; }
  .report-shell { border-top: 7px solid var(--navy); }
  .report-header {
    padding: 44px 0 34px; border-bottom: 1px solid var(--line); margin-bottom: 28px;
  }
  .report-eyebrow {
    margin: 0 0 12px; color: var(--blue); font-size: 12px; font-weight: 700;
    letter-spacing: .13em; text-transform: uppercase;
  }
  .report-header h1 { margin: 0; max-width: 820px; font-size: 40px; font-weight: 650; }
  .report-date { margin: 14px 0 0; color: var(--muted); font-size: 14px; }
  .reading-route {
    display: inline-block; margin: 14px 0 0; padding: 7px 10px; background: #e9f1f5;
    color: var(--navy); font-size: 12px; font-weight: 700;
  }
  .report-deck {
    max-width: 900px; margin: 24px 0 0; padding-left: 18px; border-left: 4px solid var(--blue);
    font-family: Georgia, "Times New Roman", serif; font-size: 21px; line-height: 1.45; color: #26343e;
  }
  .report-grid { display: grid; grid-template-columns: 190px minmax(0, 1fr); gap: 48px; align-items: start; }
  .report-toc { position: sticky; top: 18px; padding-top: 10px; }
  .report-toc-title { font-size: 11px; font-weight: 700; color: var(--muted); letter-spacing: .1em; text-transform: uppercase; }
  .report-toc a { display: block; margin-top: 10px; color: #43515b; font-size: 12px; line-height: 1.35; text-decoration: none; }
  .report-toc a:hover { color: var(--blue); }
  .report-content { min-width: 0; max-width: 940px; }
  .report-content > h1:first-child { display: none; }
  .table-shell { width: 100%; margin: 18px 0 24px; overflow-x: auto; border-top: 2px solid var(--navy); }
  table {
    width: 100%; border-collapse: collapse; margin: 0; font-size: 13px;
  }
  th {
    background: #fff; color: var(--navy); padding: 10px 11px; text-align: left;
    font-weight: 700; border-bottom: 1px solid #aeb8bf; vertical-align: bottom;
  }
  td {
    padding: 10px 11px; border-bottom: 1px solid #e2e6e8; vertical-align: top;
  }
  tr:nth-child(even) td { background: #f8f9f9; }
  tbody tr:hover td { background: #f1f4f5; }
  th:first-child, td:first-child { font-weight: 650; }
  .move-positive { color: var(--positive); font-weight: 700; white-space: nowrap; }
  .move-negative { color: var(--negative); font-weight: 700; white-space: nowrap; }
  img { display: block; max-width: 100%; height: auto; margin: 22px 0 30px; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  code {
    background: #f1f3f4; padding: 2px 5px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace; font-size: 13px;
  }
  pre {
    background: #f4f6f7; padding: 14px; overflow-x: auto;
    font-size: 12px; line-height: 1.5; border-left: 3px solid var(--navy);
  }
  pre code { background: none; padding: 0; }
  ul, ol { padding-left: 23px; margin: 8px 0 16px; }
  li { margin: 6px 0; }
  strong { color: #10171d; }
  a { color: var(--blue); text-decoration: none; }
  a:hover { text-decoration: underline; }
  hr { border: none; border-top: 1px solid var(--line); margin: 34px 0; }
  .report-footer {
    margin-top: 54px; padding: 20px 0; border-top: 1px solid var(--line);
    font-size: 12px; color: #75818a; text-align: left;
  }
  @media (max-width: 860px) {
    body { padding: 0 18px 40px; }
    .report-header { padding: 32px 0 26px; }
    .report-header h1 { font-size: 32px; }
    .report-deck { font-size: 18px; }
    .report-grid { display: block; }
    .report-toc { position: static; display: flex; flex-wrap: wrap; gap: 8px 16px; padding: 0 0 20px; border-bottom: 1px solid var(--line); }
    .report-toc-title { width: 100%; }
    .report-toc a { margin: 0; }
    h2 { font-size: 21px; margin-top: 42px; }
    h3 { font-size: 18px; }
    table { min-width: 620px; }
  }
  @media print {
    html { background: #fff; }
    body { max-width: none; padding: 0; font-size: 10.5pt; }
    .report-toc { display: none; }
    .report-grid { display: block; }
    .report-content { max-width: none; }
    .table-shell { overflow: visible; break-inside: avoid; }
    table { min-width: 0; }
    h2, h3, img { break-after: avoid; }
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
    return body_html, toc


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

    title = f"HK Morning Brief | {report_date}"
    pulse_match = re.search(r"^- \*\*Market pulse:\*\*\s*(.+)$", md_text, flags=re.MULTILINE)
    pulse = html_lib.escape(pulse_match.group(1).strip() if pulse_match else "Evidence-led Hong Kong market briefing and decision checklist.")
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
  <header class="report-header">
    <p class="report-eyebrow">Hong Kong institutional research</p>
    <h1>Morning Research Workbench</h1>
    <p class="report-date">Issue date {report_date} · Decision brief · Source-audited</p>
    <p class="reading-route">5 min scan · 25–30 min deep read · optional 10–15 min appendix</p>
    <p class="report-deck">{pulse}</p>
  </header>
  <div class="report-grid">
    <nav class="report-toc" aria-label="Report sections">
      <div class="report-toc-title">In this issue</div>
      {toc_html}
    </nav>
    <main class="report-content">
      {body_html}
    </main>
  </div>
  <div class="report-footer">
    Morning Research Workbench · Generated {report_date} · For research review, not investment advice
  </div>
</div>
</body>
</html>"""


def send_file(
    webhook_url: str,
    output_dir: Path,
    report_date: str,
    dry_run: bool = False,
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

        _wecom_post(webhook_url, {"msgtype": "file", "file": {"media_id": media_id}})
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
    if not webhook_url and not args.dry_run:
        print("WECOM_WEBHOOK_URL is not set. Skipping WeCom delivery.")
        return 0

    # Only load bundle for modes that need it (not file-only mode)
    bundle = None
    if args.mode in ("summary", "full"):
        bundle = _load_bundle(output_dir, args.report_date)

    if args.mode == "summary":
        send_summary(webhook_url, bundle, args.report_date, dry_run=args.dry_run)
    elif args.mode == "file":
        send_file(webhook_url, output_dir, args.report_date, dry_run=args.dry_run)
    elif args.mode == "full":
        send_summary(webhook_url, bundle, args.report_date, dry_run=args.dry_run)
        send_file(webhook_url, output_dir, args.report_date, dry_run=args.dry_run)
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
