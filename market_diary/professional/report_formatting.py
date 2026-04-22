from __future__ import annotations

from typing import Any, Dict, Iterable, Sequence


def _report_setting(bundle: Dict[str, Any], key: str, default: int) -> int:
    report_config = (bundle.get("report_config", {}) or {})
    value = report_config.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:+.2f}%"


def _fmt_price(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) < 10:
        digits = max(digits, 4)
    return f"{number:,.{digits}f}"


def _fmt_hkd_bn(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"HK${number / 1_000_000_000:.1f}bn"


def _fmt_millions(value: Any, currency: str = "HK$") -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{currency}{number:,.1f}mn"


def _status_label(status: str) -> str:
    mapping = {
        "live_local": "Live local",
        "stale_local": "Stale local",
        "live_public": "Live public",
        "stale_public": "Stale public",
        "live_quote": "Live quote",
        "live_hybrid": "Live quote + local",
        "proxy": "Proxy fallback",
        "unavailable": "Unavailable",
    }
    return mapping.get(str(status or ""), str(status or "Unavailable").replace("_", " ").title())


def _source_as_of(item: Dict[str, Any]) -> str:
    source = str(item.get("source", "") or "").strip()
    as_of = str(item.get("as_of", "") or "").strip()
    if source and as_of:
        return f"{source} | {as_of}"
    if source:
        return source
    if as_of:
        return as_of
    return "N/A"


def _compact_source_name(source: str, limit: int = 34) -> str:
    text = str(source or "").strip()
    if not text:
        return ""
    aliases = (
        ("Hong Kong Monetary Authority", "HKMA"),
        ("HKMA Daily Figures", "HKMA"),
        ("HKEX Daily Quotations", "HKEX"),
        ("HKEX Stock Connect", "HKEX Connect"),
        ("Public Yahoo Finance", "Yahoo"),
        ("Yahoo Finance", "Yahoo"),
        ("Eastmoney Treasury Yield History", "Eastmoney"),
    )
    for needle, alias in aliases:
        if needle.lower() in text.lower():
            return alias
    if len(text) <= limit:
        return text
    return text[:limit].rstrip(" -|/")


def _compact_source_as_of(item: Dict[str, Any]) -> str:
    source = _compact_source_name(str(item.get("source", "") or ""))
    as_of = str(item.get("as_of", "") or "").strip()
    if source and as_of:
        return f"{source} | {as_of}"
    return source or as_of or "N/A"


def _bundle_metric(bundle: Dict[str, Any], section: str, key: str) -> Dict[str, Any]:
    section_data = bundle.get(section, {}) or {}
    item = section_data.get(key, {}) if isinstance(section_data, dict) else {}
    return item if isinstance(item, dict) else {}


def _truncate(text: str, limit: int = 110, suffix: str = " [trimmed]") -> str:
    """Shorten report text without making it look like an unfinished sentence."""
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized

    if limit <= len(suffix) + 8:
        return normalized[:limit].rstrip()

    budget = limit - len(suffix)
    candidate = normalized[:budget].rstrip()
    min_boundary = max(18, int(budget * 0.55))

    sentence_boundaries = [candidate.rfind(marker) for marker in (". ", "! ", "? ", "; ", ": ")]
    sentence_boundary = max(sentence_boundaries)
    if sentence_boundary >= min_boundary:
        candidate = candidate[: sentence_boundary + 1]
    else:
        word_boundary = candidate.rfind(" ")
        if word_boundary >= min_boundary:
            candidate = candidate[:word_boundary]

    candidate = candidate.rstrip(" ,;:-|")
    return f"{candidate}{suffix}"


def _make_table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    def _cell(value: Any) -> str:
        text = str(value)
        text = text.replace("|", "\\|").replace("\r\n", "<br>").replace("\n", "<br>")
        return text

    lines = ["| " + " | ".join(_cell(header) for header in headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(_cell(cell) for cell in row) + " |")
    return "\n".join(lines)


def _summary_item(bundle: Dict[str, Any], category: str, name: str) -> Dict[str, Any]:
    summary = (bundle.get("market_summary", {}) or {})
    item = (summary.get(category, {}) or {}).get(name, {})
    return item if isinstance(item, dict) else {}


def _summary_price(bundle: Dict[str, Any], category: str, name: str) -> Any:
    return _summary_item(bundle, category, name).get("Price")


def _summary_pct(bundle: Dict[str, Any], category: str, name: str) -> Any:
    item = _summary_item(bundle, category, name)
    value = item.get("Pct Change")
    if isinstance(value, str):
        value = value.replace("%", "").strip()
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_alert_pct(value: Any, threshold: float = 1.5) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    text = f"{number:+.2f}%"
    return f"**{text}**" if abs(number) >= threshold else text
