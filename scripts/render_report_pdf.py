"""Render the audited self-contained HTML report as an A4 PDF.

HTML remains the canonical reader surface.  This script uses the installed
Chrome/Chromium engine so the print companion shares the same content and CSS;
``pypdf`` then validates the artifact before it is eligible for delivery.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
import re
from pathlib import Path
from typing import Any, Dict, Iterable

from pypdf import PdfReader
from websockets.sync.client import connect


ROOT = Path(__file__).resolve().parents[1]
A4_POINTS = (595.28, 841.89)
DEFAULT_MAX_BYTES = 18 * 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render and validate the morning briefing PDF.")
    parser.add_argument("--report-date", required=True)
    parser.add_argument("--output-dir", default="reports_professional")
    parser.add_argument("--html-path", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--scope", choices=("core", "full"), default="core")
    parser.add_argument("--min-pages", type=int, default=2)
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Optional safety ceiling; 0 disables the page-count ceiling.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=260,
        help="Reject a text-only page below this content threshold; image-led pages are allowed.",
    )
    return parser.parse_args()


def _browser_candidates() -> Iterable[Path]:
    configured = (os.getenv("DMD_CHROME_PATH") or "").strip()
    if configured:
        yield Path(configured)
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        resolved = shutil.which(name)
        if resolved:
            yield Path(resolved)
    for path in (
        Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
        Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge"),
    ):
        yield path


def find_browser() -> Path:
    for candidate in _browser_candidates():
        if candidate.exists() and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError("Chrome/Chromium was not found. Set DMD_CHROME_PATH or install Google Chrome.")


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _browser_version(browser: Path) -> str:
    try:
        result = subprocess.run([str(browser), "--version"], check=True, capture_output=True, text=True, timeout=10)
        return result.stdout.strip() or result.stderr.strip()
    except Exception:
        return browser.name


class _CdpClient:
    def __init__(self, websocket_url: str) -> None:
        # Base64-encoded reports routinely exceed the library's 1 MiB default
        # frame limit even when the final PDF is small.
        self._socket = connect(websocket_url, open_timeout=10, close_timeout=5, max_size=None)
        self._next_id = 0

    def close(self) -> None:
        self._socket.close()

    def call(self, method: str, params: Dict[str, Any] | None = None) -> Dict[str, Any]:
        self._next_id += 1
        message_id = self._next_id
        self._socket.send(json.dumps({"id": message_id, "method": method, "params": params or {}}))
        while True:
            payload = json.loads(self._socket.recv())
            if payload.get("id") != message_id:
                continue
            if payload.get("error"):
                raise RuntimeError(f"Chrome DevTools error for {method}: {payload['error']}")
            return payload.get("result", {}) or {}


def _wait_for_page(port: int, expected_uri: str, timeout_seconds: float = 20.0) -> Dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=2) as response:
                targets = json.load(response)
            pages = [item for item in targets if item.get("type") == "page" and item.get("webSocketDebuggerUrl")]
            for page in pages:
                if page.get("url") == expected_uri:
                    return page
            if pages:
                return pages[0]
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.2)
    raise RuntimeError(f"Chrome page target was not ready: {last_error}")


def _scoped_html(source: Path, destination: Path, scope: str) -> None:
    html = source.read_text(encoding="utf-8")
    body_class = "print-core" if scope == "core" else "print-full"
    if 'class="report-screen"' in html:
        html = html.replace('class="report-screen"', f'class="report-screen {body_class}"', 1)
    else:
        html = html.replace("<body", f'<body class="{body_class}"', 1)
    destination.write_text(html, encoding="utf-8")


def _render_with_browser(html_path: Path, pdf_path: Path, report_date: str, scope: str) -> str:
    browser = find_browser()
    port = _free_port()
    # Chrome can briefly keep a profile file open after the browser process has
    # exited.  A successful PDF must not be reclassified as a rendering failure
    # solely because that disposable profile loses a cleanup race on CI.
    with tempfile.TemporaryDirectory(prefix="dmd-pdf-", ignore_cleanup_errors=True) as temp_dir:
        temp_root = Path(temp_dir)
        scoped_html = temp_root / f"report-{scope}.html"
        _scoped_html(html_path, scoped_html, scope)
        uri = scoped_html.resolve().as_uri()
        process = subprocess.Popen(
            [
                str(browser),
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                f"--remote-debugging-port={port}",
                f"--user-data-dir={temp_root / 'chrome-profile'}",
                "--allow-file-access-from-files",
                uri,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        client = None
        try:
            target = _wait_for_page(port, uri)
            client = _CdpClient(str(target["webSocketDebuggerUrl"]))
            client.call("Page.enable")
            ready = client.call(
                "Runtime.evaluate",
                {
                    "expression": """
                    (async () => {
                      if (document.fonts && document.fonts.ready) await document.fonts.ready;
                      const images = Array.from(document.images);
                      await Promise.all(images.map(img => img.complete ? Promise.resolve() :
                        new Promise((resolve, reject) => {
                          img.addEventListener('load', resolve, {once:true});
                          img.addEventListener('error', reject, {once:true});
                        })));
                      return {images: images.length, broken: images.filter(img => !img.naturalWidth).length};
                    })()
                    """,
                    "awaitPromise": True,
                    "returnByValue": True,
                },
            )
            value = (((ready.get("result") or {}).get("value")) or {})
            if int(value.get("broken", 0) or 0):
                raise RuntimeError(f"PDF render blocked: {value['broken']} image(s) failed to load")

            footer = (
                '<div style="width:100%;font-size:8px;color:#5f666b;padding:0 15mm;'
                'display:flex;justify-content:space-between;font-family:Arial,sans-serif">'
                f'<span>Morning Market Brief · {report_date} · Research only</span>'
                '<span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>'
            )
            result = client.call(
                "Page.printToPDF",
                {
                    "landscape": False,
                    "displayHeaderFooter": True,
                    "printBackground": True,
                    "preferCSSPageSize": True,
                    "headerTemplate": "<span></span>",
                    "footerTemplate": footer,
                    "marginTop": 0.0,
                    "marginBottom": 0.0,
                    "marginLeft": 0.0,
                    "marginRight": 0.0,
                    "transferMode": "ReturnAsBase64",
                },
            )
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(base64.b64decode(result["data"]))
        finally:
            if client is not None:
                try:
                    # Ask the browser (and its profile-writing child processes)
                    # to shut down cleanly before closing the DevTools socket.
                    client.call("Browser.close")
                except Exception:
                    pass
                try:
                    client.close()
                except Exception:
                    pass
            if process.poll() is None:
                process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return _browser_version(browser)


def validate_pdf(
    pdf_path: Path,
    *,
    report_date: str,
    min_pages: int,
    max_pages: int,
    min_text_chars: int = 260,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Dict[str, Any]:
    if not pdf_path.exists() or pdf_path.stat().st_size < 1024:
        raise RuntimeError("PDF output is missing or implausibly small")
    if pdf_path.stat().st_size > max_bytes:
        raise RuntimeError(f"PDF exceeds the {max_bytes:,}-byte delivery limit")
    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    if page_count < min_pages:
        raise RuntimeError(f"PDF page count {page_count} is below the minimum {min_pages}")
    if max_pages > 0 and page_count > max_pages:
        raise RuntimeError(f"PDF page count {page_count} exceeds the safety ceiling {max_pages}")

    expected_width, expected_height = A4_POINTS
    text_parts = []
    page_density = []
    low_information_pages = []
    for index, page in enumerate(reader.pages, start=1):
        width = float(page.mediabox.width)
        height = float(page.mediabox.height)
        if abs(width - expected_width) > 3 or abs(height - expected_height) > 3:
            raise RuntimeError(f"Page {index} is not A4: {width:.2f} x {height:.2f}pt")
        page_text = page.extract_text() or ""
        text_parts.append(page_text)
        # Ignore the repeated print footer when assessing whether a page has a
        # real information job.  An image-led chart page is valid even with
        # little extractable text; a stranded heading or caption is not.
        content_text = re.sub(
            rf"Morning Market Brief\s*·\s*{re.escape(report_date)}\s*·\s*Research only\s*\d+\s*/\s*\d+",
            "",
            page_text,
            flags=re.IGNORECASE,
        )
        content_chars = len(re.sub(r"\s+", "", content_text))
        try:
            image_count = len(page.images)
        except Exception:
            image_count = 0
        page_density.append({"page": index, "text_chars": content_chars, "images": image_count})
        if content_chars < min_text_chars and image_count == 0:
            low_information_pages.append(index)
    text = "\n".join(text_parts)
    for required in ("Morning Market Brief", report_date):
        if required not in text:
            raise RuntimeError(f"PDF is missing required text: {required}")
    if low_information_pages:
        pages = ", ".join(str(page) for page in low_information_pages)
        raise RuntimeError(
            f"PDF contains low-information text-only page(s): {pages}. "
            "Fix pagination instead of compressing the report."
        )
    return {
        "status": "ok",
        "report_date": report_date,
        "page_count": page_count,
        "size_bytes": pdf_path.stat().st_size,
        "page_size": "A4",
        "low_information_pages": [],
        "page_density": page_density,
    }


def render_report_pdf(
    html_path: Path,
    pdf_path: Path,
    *,
    report_date: str,
    scope: str,
    min_pages: int = 2,
    max_pages: int = 0,
    min_text_chars: int = 260,
) -> Dict[str, Any]:
    renderer_version = _render_with_browser(html_path, pdf_path, report_date, scope)
    receipt = validate_pdf(
        pdf_path,
        report_date=report_date,
        min_pages=min_pages,
        max_pages=max_pages,
        min_text_chars=min_text_chars,
    )
    receipt.update({"scope": scope, "renderer": renderer_version, "filename": pdf_path.name})
    receipt_path = pdf_path.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    args = _parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    html_path = Path(args.html_path).resolve() if args.html_path else output_dir / f"{args.report_date}_morning_briefing.html"
    if not html_path.exists():
        raise FileNotFoundError(f"HTML report not found: {html_path}")
    default_name = f"{args.report_date}_morning_briefing_{args.scope}.pdf"
    pdf_path = Path(args.output).resolve() if args.output else output_dir / default_name
    receipt = render_report_pdf(
        html_path,
        pdf_path,
        report_date=args.report_date,
        scope=args.scope,
        min_pages=args.min_pages,
        max_pages=args.max_pages,
        min_text_chars=args.min_text_chars,
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
