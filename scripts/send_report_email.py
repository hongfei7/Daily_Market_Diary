from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "market_diary"))

from professional.email_builder import build_email_html, build_email_subject, build_email_text


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send the generated morning briefing by email.")
    parser.add_argument("--report-date", required=True, help="Report date in YYYY-MM-DD format.")
    parser.add_argument("--output-dir", default="reports_professional", help="Directory containing generated outputs.")
    parser.add_argument("--to", required=True, help="Recipient email address.")
    parser.add_argument("--dry-run", action="store_true", help="Render an email preview without sending.")
    return parser.parse_args()


def _load_bundle(output_dir: Path, report_date: str) -> dict:
    bundle_path = output_dir / "raw" / f"{report_date}_bundle.json"
    with bundle_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _attach_markdown(message: MIMEMultipart, path: Path) -> None:
    with path.open("rb") as handle:
        part = MIMEApplication(handle.read(), _subtype="markdown")
    part.add_header("Content-Disposition", "attachment", filename=path.name)
    message.attach(part)


def _attach_dashboard(message: MIMEMultipart, path: Path, content_id: str) -> None:
    with path.open("rb") as handle:
        part = MIMEImage(handle.read(), _subtype="png")
    part.add_header("Content-ID", f"<{content_id}>")
    part.add_header("Content-Disposition", "inline", filename=path.name)
    message.attach(part)


def main() -> int:
    args = _parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    report_path = output_dir / f"{args.report_date}_morning_briefing.md"
    dashboard_path = output_dir / "charts" / f"dashboard_{args.report_date}.png"
    preview_path = output_dir / f"{args.report_date}_email_preview.html"

    if not report_path.exists():
        raise FileNotFoundError(f"Report file not found: {report_path}")

    bundle = _load_bundle(output_dir, args.report_date)
    dashboard_cid = "research_dashboard"
    html_body = build_email_html(bundle, dashboard_cid=dashboard_cid if dashboard_path.exists() else None)
    text_body = build_email_text(bundle)
    subject = build_email_subject(bundle)

    if args.dry_run:
        preview_path.write_text(html_body, encoding="utf-8")
        print(f"Email preview written to: {preview_path}")
        print(f"Subject: {subject}")
        return 0

    smtp_host = (os.getenv("SMTP_HOST") or "").strip()
    smtp_port = int((os.getenv("SMTP_PORT") or "587").strip())
    smtp_username = (os.getenv("SMTP_USERNAME") or "").strip()
    smtp_password = (os.getenv("SMTP_PASSWORD") or "").strip()
    smtp_from = (os.getenv("SMTP_FROM") or smtp_username).strip()
    smtp_use_tls = (os.getenv("SMTP_USE_TLS") or "true").strip().lower() not in {"0", "false", "no"}

    if not smtp_host or not smtp_username or not smtp_password or not smtp_from:
        raise RuntimeError("SMTP credentials are incomplete. Set SMTP_HOST, SMTP_USERNAME, SMTP_PASSWORD, and optionally SMTP_FROM.")

    message = MIMEMultipart("related")
    message["Subject"] = subject
    message["From"] = smtp_from
    message["To"] = args.to

    alternative = MIMEMultipart("alternative")
    alternative.attach(MIMEText(text_body, "plain", "utf-8"))
    alternative.attach(MIMEText(html_body, "html", "utf-8"))
    message.attach(alternative)

    _attach_markdown(message, report_path)
    if dashboard_path.exists():
        _attach_dashboard(message, dashboard_path, dashboard_cid)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.ehlo()
        if smtp_use_tls:
            server.starttls()
            server.ehlo()
        server.login(smtp_username, smtp_password)
        server.sendmail(smtp_from, [args.to], message.as_string())

    print(f"Email sent to {args.to}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
