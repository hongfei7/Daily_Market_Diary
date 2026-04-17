"""Lightweight text cleanup helpers for public-market news feeds."""

from __future__ import annotations

import html
import re
from typing import Any


HTML_TAG_RE = re.compile(r"<[^>]+>")
WHITESPACE_RE = re.compile(r"\s+")

COMMON_MOJIBAKE_REPLACEMENTS = {
    "\u00e2\u20ac\u2122": "'",
    "\u00e2\u20ac\u02dc": "'",
    "\u00e2\u20ac\u0153": '"',
    "\u00e2\u20ac\u009d": '"',
    "\u00e2\u20ac\u201c": "-",
    "\u00e2\u20ac\u201d": "-",
    "\u00e2\u20ac\u00a6": "...",
    "\u00e2\u201a\u00ac": "EUR ",
    "\u00c2\u00a0": " ",
    "\u00c2": "",
    "聽": " ",
    "\u00e8\u0081\u00bd": " ",
    "鑱": " ",
    "鈥檚": "'s",
    "鈥檛": "n't",
    "鈥檒l": "'ll",
    "鈥檇": "'d",
    "鈥檝e": "'ve",
}


def _repair_common_mojibake(text: str) -> str:
    repaired = text
    for source, target in COMMON_MOJIBAKE_REPLACEMENTS.items():
        repaired = repaired.replace(source, target)
    return repaired


def normalize_news_text(
    value: Any,
    *,
    strip_html_tags: bool = True,
    collapse_whitespace: bool = True,
    max_length: int = 0,
) -> str:
    if value is None:
        return ""

    text = str(value)
    if strip_html_tags:
        text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ").replace("\u200b", "").replace("\ufeff", "")
    text = _repair_common_mojibake(text)
    text = re.sub(r"鈧\?(\d)", r"EUR \1", text)
    text = re.sub(r"鈧(\d)", r"EUR \1", text)
    text = re.sub(r"鈥([A-Za-z])", r"-\1", text)
    if collapse_whitespace:
        text = WHITESPACE_RE.sub(" ", text).strip()
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)

    if max_length > 0 and len(text) > max_length:
        return text[: max_length - 3].rstrip() + "..."
    return text
