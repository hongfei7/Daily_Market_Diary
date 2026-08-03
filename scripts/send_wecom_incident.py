"""Send a dependency-free WeCom incident notice for a failed briefing run."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Notify WeCom that the morning briefing is unavailable.")
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--reason", required=True)
    return parser.parse_args()


def build_incident_markdown(report_date: str, reason: str, now_hkt: str, run_url: str = "") -> str:
    lines = [
        f"# Morning Brief delivery warning | {report_date}",
        f"> The {now_hkt} HKT run could not release a verified full report.",
        f"> **Reason:** {reason}",
        "> The 06:47 recovery run will retry automatically when this is the primary scheduled run.",
    ]
    if run_url:
        lines.extend(["", f"[Open GitHub Actions diagnostics]({run_url})"])
    return "\n".join(lines)


def main() -> int:
    args = _parse_args()
    webhook = (os.getenv("WECOM_WEBHOOK_URL") or "").strip()
    if not webhook:
        print("WECOM_WEBHOOK_URL is not configured; incident notification cannot be sent.", file=sys.stderr)
        return 1

    run_url = ""
    server = (os.getenv("GITHUB_SERVER_URL") or "").rstrip("/")
    repo = (os.getenv("GITHUB_REPOSITORY") or "").strip()
    run_id = (os.getenv("GITHUB_RUN_ID") or "").strip()
    if server and repo and run_id:
        run_url = f"{server}/{repo}/actions/runs/{run_id}"

    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%H:%M")
    markdown = build_incident_markdown(args.report_date, args.reason, now, run_url)
    payload = json.dumps(
        {"msgtype": "markdown", "markdown": {"content": markdown}},
        ensure_ascii=False,
    ).encode("utf-8")
    request = urllib.request.Request(webhook, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=30) as response:
        result = json.loads(response.read().decode("utf-8"))
    if result.get("errcode") != 0:
        print(f"WeCom incident notification failed: {result}", file=sys.stderr)
        return 1
    print("WeCom incident notification sent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
