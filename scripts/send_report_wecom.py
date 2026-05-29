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
    hstech = _price_line("HSTECH", _p("Equities", "Hang Seng TECH ETF"), _c("Equities", "Hang Seng TECH ETF"))
    lines.append(f"> {spx} | {ndx} | {hstech}")

    # Line 3: Rates / FX / Commodities
    parts = []
    us10y = _p("Rates", "10Y Treasury")
    if us10y is not None:
        parts.append(f"**US10Y** {_fmt_price(us10y, 2)}%")
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
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "PingFang SC",
                 "Hiragino Sans GB", "Microsoft YaHei", "Helvetica Neue", Arial,
                 sans-serif;
    color: #1f2328; line-height: 1.65; max-width: 720px; margin: 0 auto;
    padding: 16px; background: #fff; font-size: 15px;
  }
  h1 { font-size: 22px; border-bottom: 2px solid #1677ff; padding-bottom: 8px; margin: 24px 0 16px; }
  h2 { font-size: 18px; color: #1677ff; margin: 20px 0 10px; padding: 6px 0; border-bottom: 1px solid #e8e8e8; }
  h3 { font-size: 16px; margin: 16px 0 8px; color: #333; }
  h4, h5, h6 { font-size: 15px; margin: 14px 0 6px; }
  p { margin: 0 0 10px; }
  blockquote {
    border-left: 3px solid #1677ff; padding: 6px 12px; margin: 10px 0;
    background: #f0f5ff; color: #555; font-size: 14px;
  }
  table {
    width: 100%; border-collapse: collapse; margin: 12px 0; font-size: 13px;
  }
  th {
    background: #1677ff; color: #fff; padding: 8px 10px; text-align: left;
    font-weight: 600;
  }
  td {
    padding: 7px 10px; border-bottom: 1px solid #f0f0f0;
  }
  tr:nth-child(even) td { background: #fafafa; }
  img { max-width: 100%; height: auto; border-radius: 6px; margin: 8px 0; }
  code {
    background: #f5f5f5; padding: 2px 5px; border-radius: 3px;
    font-family: "SF Mono", "Fira Code", "Consolas", monospace; font-size: 13px;
  }
  pre {
    background: #f6f8fa; padding: 12px; border-radius: 6px; overflow-x: auto;
    font-size: 12px; line-height: 1.5; border: 1px solid #e1e4e8;
  }
  pre code { background: none; padding: 0; }
  ul, ol { padding-left: 22px; margin: 6px 0 10px; }
  li { margin: 3px 0; }
  strong { color: #1a1a1a; }
  a { color: #1677ff; text-decoration: none; }
  a:hover { text-decoration: underline; }
  hr { border: none; border-top: 1px solid #e8e8e8; margin: 20px 0; }
  .report-header {
    background: linear-gradient(135deg, #1677ff 0%, #0958d9 100%);
    color: #fff; padding: 20px 24px; border-radius: 10px; margin-bottom: 20px;
  }
  .report-header h1 { color: #fff; border: none; margin: 0 0 6px; font-size: 22px; }
  .report-header p { color: rgba(255,255,255,0.85); margin: 2px 0; font-size: 13px; }
  .report-footer {
    margin-top: 30px; padding: 16px; background: #f6f8fa; border-radius: 8px;
    font-size: 12px; color: #8c8c8c; text-align: center;
  }
</style>"""


def _md_to_html(md_text: str, output_dir: Path, report_date: str, md_source_dir: Optional[Path] = None) -> str:
    """Convert markdown report to a self-contained HTML document with embedded images."""
    # Convert MD to HTML body
    body_html = mistune.html(md_text)

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

    return f"""\
<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{HTML_CSS}
</head>
<body>
<div class="report-header">
  <h1>{title}</h1>
  <p>Generated by Morning Research Workbench</p>
</div>
{body_html}
<div class="report-footer">
  Morning Research Workbench | Generated {report_date}
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
