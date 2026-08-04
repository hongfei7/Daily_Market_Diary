"""HKEXnews announcement adapters for Hong Kong company-event tracking."""

from __future__ import annotations

import os
import re
from io import BytesIO
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - dependency is pinned in production.
    PdfReader = None


HKEXNEWS_HOST = "https://www1.hkexnews.hk"
HKEXNEWS_PREDEFINED_TEMPLATE = "https://www1.hkexnews.hk/search/predefineddoc.xhtml?predefineddocuments={category_id}&lang=EN"
HKEX_PROFIT_WARNING_URL = "http://www3.hkexnews.hk/reports/profitwarning/ncms/profitwarning_anntdate_des.htm"
HKEXNEWS_SOURCE = "HKEXnews"
USER_AGENT = "Daily-Market-Diary/3.1"
REQUEST_TIMEOUT = float(os.environ.get("DMD_PUBLIC_REQUEST_TIMEOUT_SECONDS", "12"))
REQUEST_RETRY_TOTAL = int(os.environ.get("DMD_PUBLIC_RETRY_TOTAL", "1"))
FILING_TIMEOUT = float(os.environ.get("DMD_HKEX_FILING_TIMEOUT_SECONDS", "8"))
FILING_MAX_BYTES = int(os.environ.get("DMD_HKEX_FILING_MAX_BYTES", str(5 * 1024 * 1024)))
FILING_MAX_PAGES = int(os.environ.get("DMD_HKEX_FILING_MAX_PAGES", "3"))
FILING_ENRICHMENT_LIMIT = int(os.environ.get("DMD_HKEX_FILING_ENRICHMENT_LIMIT", "2"))

PREDEFINED_CATEGORIES = {
    "results_announcements": {
        "id": "7",
        "label": "Results Announcements",
        "event_type": "Results announcement",
    },
    "trading_halts": {
        "id": "9",
        "label": "Resumption / Suspension / Trading Halt",
        "event_type": "Trading-status announcement",
    },
}


def _session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=REQUEST_RETRY_TOTAL,
        connect=REQUEST_RETRY_TOTAL,
        read=REQUEST_RETRY_TOTAL,
        backoff_factor=1.0,
        allowed_methods=frozenset(["GET"]),
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


def _parse_target(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _normalize_code(code: Any) -> str:
    digits = re.sub(r"\D", "", str(code or ""))
    return digits.zfill(5) if digits else ""


def _watchlist_terms(watchlists: Optional[Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Set[str]]:
    codes: Set[str] = set()
    names: Set[str] = set()
    for items in (watchlists or {}).values():
        for item in items or []:
            code = _normalize_code(str(item.get("ticker", "")).split(".")[0])
            if code:
                codes.add(code)
            name = str(item.get("name", "") or "").lower().strip()
            if name:
                names.add(name)
    return {"codes": codes, "names": names}


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _parse_release_time(value: str) -> Optional[datetime]:
    value = _clean_text(value)
    for fmt in ("%d/%m/%Y %H:%M", "%d/%m/%Y"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_named_date(value: str) -> Optional[date]:
    value = _clean_text(value)
    for fmt in ("%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue
    return None


def _is_recent(item_date: date, target: date, max_age_days: int) -> bool:
    return target - timedelta(days=max_age_days) <= item_date <= target


def _importance_score(item: Dict[str, Any], watch_terms: Dict[str, Set[str]]) -> float:
    text = " ".join(
        [
            str(item.get("title", "")),
            str(item.get("document", "")),
            str(item.get("company", "")),
            str(item.get("event_type", "")),
        ]
    ).lower()
    score = 1.0
    watchlist_match = item.get("code") in watch_terms.get("codes", set()) or any(
        name and name in text for name in watch_terms.get("names", set())
    )
    if watchlist_match:
        score += 3.0
    for token in ("inside information", "suspension", "resumption", "trading halt"):
        if token in text:
            score += 2.5
    if "profit warning" in text:
        score += 1.5
    for token in ("final results", "interim results", "quarterly results", "delay in results", "revision"):
        if token in text:
            score += 1.0
    return score


def _grade(score: float) -> str:
    if score >= 5.0:
        return "A"
    if score >= 3.5:
        return "B"
    return "C"


def _watchlist_match(item: Dict[str, Any], watch_terms: Dict[str, Set[str]]) -> bool:
    if item.get("code") in watch_terms.get("codes", set()):
        return True
    text = " ".join([str(item.get("company", "")), str(item.get("title", ""))]).lower()
    return any(name and name in text for name in watch_terms.get("names", set()))


def _clip_sentence(value: str, limit: int) -> str:
    text = _clean_text(value)
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rsplit(" ", 1)[0].rstrip(" ,;:") + "…"


def _filing_sentences(text: str) -> List[str]:
    normalized = _clean_text(text.replace("\x00", " "))
    if not normalized:
        return []
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z(])", normalized)
        if len(sentence.strip()) >= 24
    ]


def _extract_filing_details(text: str) -> Dict[str, str]:
    """Extract source-near facts without inferring an EPS or valuation impact."""

    sentences = _filing_sentences(text)
    result_sentence = next(
        (
            sentence
            for sentence in sentences
            if any(token in sentence.lower() for token in ("expected to record", "expects to record", "anticipated to record"))
            and any(token in sentence.lower() for token in ("profit", "loss", "revenue", "earnings"))
        ),
        "",
    )
    driver_sentence = next(
        (
            sentence
            for sentence in sentences
            if any(token in sentence.lower() for token in ("mainly attributable to", "primarily attributable to", "main reasons"))
        ),
        "",
    )
    next_event_sentence = next(
        (
            sentence
            for sentence in sentences
            if "expected to be published" in sentence.lower()
            and any(token in sentence.lower() for token in ("results", "announcement", "report"))
        ),
        "",
    )
    if result_sentence:
        compact_result = re.search(
            r"((?:the\s+)?(?:group|company)\s+(?:is|are)\s+expected\s+to\s+record.+)",
            result_sentence,
            flags=re.IGNORECASE,
        )
        if compact_result:
            result_sentence = compact_result.group(1)
    next_disclosure = ""
    if next_event_sentence:
        disclosure_date = re.search(
            r"expected\s+to\s+be\s+published(?:\s+on)?\s+(\d{1,2}\s+[A-Za-z]+\s+\d{4})",
            next_event_sentence,
            flags=re.IGNORECASE,
        )
        next_disclosure = (
            f"Next results disclosure expected on {disclosure_date.group(1)} (official filing)."
            if disclosure_date
            else _clip_sentence(next_event_sentence, 190)
        )
    return {
        "filing_extract": _clip_sentence(result_sentence, 380),
        "filing_drivers": _clip_sentence(driver_sentence, 280),
        "next_disclosure": next_disclosure,
        "filing_detail_status": "parsed" if result_sentence else "headline_only",
    }


def _enrich_official_filing(item: Dict[str, Any], session: requests.Session) -> Optional[str]:
    """Attach bounded primary-filing facts for portfolio-relevant announcements."""

    url = str(item.get("url", "") or "").strip()
    if not url or PdfReader is None:
        item["filing_detail_status"] = "parser_unavailable" if PdfReader is None else "headline_only"
        return "pypdf is unavailable" if PdfReader is None else None
    try:
        response = session.get(url.replace("http://", "https://", 1), timeout=FILING_TIMEOUT)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            raise ValueError("official filing response was not a PDF")
        if len(content) > FILING_MAX_BYTES:
            raise ValueError(f"official filing exceeded {FILING_MAX_BYTES} bytes")
        reader = PdfReader(BytesIO(content))
        text = " ".join((page.extract_text() or "") for page in reader.pages[:FILING_MAX_PAGES])
        item.update(_extract_filing_details(text))
        return None
    except Exception as exc:
        item["filing_detail_status"] = "headline_only"
        return f"{item.get('ticker', '')}: {type(exc).__name__}: {exc}"


def _parse_predefined(
    category_key: str,
    target: date,
    max_age_days: int,
    watch_terms: Dict[str, Set[str]],
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    category = PREDEFINED_CATEGORIES[category_key]
    url = HKEXNEWS_PREDEFINED_TEMPLATE.format(category_id=category["id"])
    http = session or _session()
    response = http.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    rows: List[Dict[str, Any]] = []

    for tr in soup.select("table tbody tr"):
        release_text = _clean_text(tr.select_one(".release-time").get_text(" ", strip=True) if tr.select_one(".release-time") else "")
        code = _normalize_code(tr.select_one(".stock-short-code").get_text(" ", strip=True) if tr.select_one(".stock-short-code") else "")
        company = _clean_text(tr.select_one(".stock-short-name").get_text(" ", strip=True) if tr.select_one(".stock-short-name") else "")
        headline = _clean_text(tr.select_one(".headline").get_text(" ", strip=True) if tr.select_one(".headline") else "")
        link = tr.select_one(".doc-link a")
        title = _clean_text(link.get_text(" ", strip=True) if link else "")
        href = urljoin(HKEXNEWS_HOST, link.get("href", "")) if link else ""
        released_at = _parse_release_time(release_text)
        if released_at is None or not _is_recent(released_at.date(), target, max_age_days):
            continue

        item = {
            "release_time": released_at.strftime("%Y-%m-%d %H:%M"),
            "date": released_at.date().isoformat(),
            "code": code,
            "ticker": f"{code}.HK" if code else "",
            "company": company,
            "event_type": category["event_type"],
            "document": headline,
            "title": title,
            "url": href,
            "source": HKEXNEWS_SOURCE,
        }
        score = _importance_score(item, watch_terms)
        item["score"] = round(score, 2)
        item["grade"] = _grade(score)
        item["watchlist_match"] = _watchlist_match(item, watch_terms)
        item["date_confidence"] = "confirmed"
        rows.append(item)

    rows.sort(key=lambda item: (item.get("score", 0), item.get("release_time", "")), reverse=True)
    return rows


def _parse_profit_warnings(
    target: date,
    max_age_days: int,
    watch_terms: Dict[str, Set[str]],
    session: Optional[requests.Session] = None,
) -> List[Dict[str, Any]]:
    http = session or _session()
    response = http.get(HKEX_PROFIT_WARNING_URL, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    rows: List[Dict[str, Any]] = []

    for tr in soup.select("table tr"):
        cells = tr.find_all("td")
        if len(cells) < 4:
            continue
        code = _normalize_code(cells[0].get_text(" ", strip=True))
        company = _clean_text(cells[1].get_text(" ", strip=True))
        ann_date = _parse_named_date(cells[2].get_text(" ", strip=True))
        link = cells[3].find("a")
        href = urljoin("https://www3.hkexnews.hk", link.get("href", "")) if link else ""
        if ann_date is None or not _is_recent(ann_date, target, max_age_days):
            continue

        item = {
            "release_time": ann_date.isoformat(),
            "date": ann_date.isoformat(),
            "code": code,
            "ticker": f"{code}.HK" if code else "",
            "company": company,
            "event_type": "Profit warning",
            "document": "Announcements Concerning Profit Warning",
            "title": "Profit warning / alert announcement",
            "url": href,
            "source": HKEXNEWS_SOURCE,
        }
        score = _importance_score(item, watch_terms)
        item["score"] = round(score, 2)
        item["grade"] = _grade(score)
        item["watchlist_match"] = _watchlist_match(item, watch_terms)
        item["date_confidence"] = "confirmed_date"
        item["filing_detail_status"] = "headline_only"
        rows.append(item)

    rows.sort(key=lambda item: (item.get("score", 0), item.get("date", "")), reverse=True)
    return rows


def fetch_hkex_announcements(
    report_date: str,
    watchlists: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    max_age_days: int = 7,
    limit: int = 20,
) -> Dict[str, Any]:
    """Fetch recent HKEXnews company announcements from public official pages."""

    target = _parse_target(report_date)
    watch_terms = _watchlist_terms(watchlists)
    data: Dict[str, List[Dict[str, Any]]] = {
        "results_announcements": [],
        "profit_warnings": [],
        "trading_halts": [],
        "watchlist_hits": [],
        "top_announcements": [],
    }
    errors: List[str] = []
    session = _session()

    for category_key in PREDEFINED_CATEGORIES:
        try:
            data[category_key] = _parse_predefined(category_key, target, max_age_days, watch_terms, session=session)[:limit]
        except Exception as exc:
            errors.append(f"{category_key}: {type(exc).__name__}: {exc}")

    try:
        data["profit_warnings"] = _parse_profit_warnings(target, max_age_days, watch_terms, session=session)[:limit]
    except Exception as exc:
        errors.append(f"profit_warnings: {type(exc).__name__}: {exc}")

    combined = data["profit_warnings"] + data["results_announcements"] + data["trading_halts"]
    combined.sort(key=lambda item: (item.get("score", 0), item.get("release_time", "")), reverse=True)
    data["top_announcements"] = combined[:limit]
    data["watchlist_hits"] = [item for item in combined if item.get("watchlist_match")][:limit]

    filing_errors: List[str] = []
    for item in data["watchlist_hits"][:FILING_ENRICHMENT_LIMIT]:
        if str(item.get("url", "")).lower().endswith(".pdf"):
            error = _enrich_official_filing(item, session)
            if error:
                filing_errors.append(error)

    status = "ok" if combined and not errors else "partial" if combined else "error"
    return {
        "status": status,
        "data": data,
        "meta": {
            "report_date": report_date,
            "source": HKEXNEWS_SOURCE,
            "source_urls": [
                HKEXNEWS_PREDEFINED_TEMPLATE.format(category_id=category["id"])
                for category in PREDEFINED_CATEGORIES.values()
            ]
            + [HKEX_PROFIT_WARNING_URL],
            "available_count": len(combined),
            "errors": errors,
            "filing_enriched_count": sum(
                1 for item in data["watchlist_hits"] if item.get("filing_detail_status") == "parsed"
            ),
            "filing_enrichment_errors": filing_errors,
        },
    }
