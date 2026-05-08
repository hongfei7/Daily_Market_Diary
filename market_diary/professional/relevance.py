from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Set


_STOPWORDS = {
    "a",
    "an",
    "and",
    "bank",
    "co",
    "company",
    "corp",
    "corporation",
    "fund",
    "group",
    "holdings",
    "hk",
    "holdings",
    "inc",
    "index",
    "international",
    "limited",
    "ltd",
    "of",
    "plc",
    "the",
    "trust",
    "etf",
    "class",
    "ordinary",
    "shares",
}

_SECTOR_HINTS = {
    "Semiconductors and AI": {"ai", "chip", "chips", "semiconductor", "gpu", "foundry", "memory", "server", "model"},
    "Technology": {"technology", "software", "cloud", "server", "data", "ai", "chip"},
    "China Internet": {"internet", "platform", "e-commerce", "ecommerce", "ads", "cloud", "gaming", "delivery"},
    "Financials": {"bank", "banking", "broker", "insurance", "asset", "wealth", "loan", "credit", "exchange"},
    "Autos and EV": {"auto", "autos", "ev", "electric", "vehicle", "vehicles", "battery", "car", "cars"},
    "Industrials": {"industrial", "factory", "manufacturing", "machinery", "logistics", "shipping", "freight"},
    "Energy": {"oil", "gas", "energy", "lng", "refining", "crude", "upstream"},
    "Healthcare": {"healthcare", "drug", "biotech", "medical", "hospital", "trial", "approval"},
    "Consumer": {"consumer", "retail", "shopping", "travel", "food", "beverage"},
}

_HK_MARKET_TERMS = {
    "hong kong",
    "hang seng",
    "hsi",
    "hscei",
    "hstech",
    "hkex",
    "stock connect",
    "southbound",
    "northbound",
    "usd/hkd",
    "usd-cnh",
    "usd/cnh",
    "cnh",
    "china",
    "offshore china",
}


def _normalize_text(*parts: Any) -> str:
    return " ".join(str(part or "") for part in parts).strip().lower()


def _compact_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9.+/-]{1,}", text.lower())


def _significant_tokens(text: str) -> List[str]:
    tokens: List[str] = []
    for token in _tokenize(text):
        cleaned = token.strip(".+-/")
        if len(cleaned) < 2:
            continue
        if cleaned in _STOPWORDS:
            continue
        if cleaned.isdigit():
            continue
        tokens.append(cleaned)
    return tokens


def build_coverage_profiles(watchlists: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, str]]:
    profiles: List[Dict[str, str]] = []
    for bucket_items in (watchlists or {}).values():
        for item in bucket_items:
            ticker = str(item.get("ticker", "")).strip()
            name = str(item.get("name", "")).strip()
            sector = str(item.get("sector", "")).strip()
            if not ticker and not name:
                continue
            profiles.append({"ticker": ticker, "name": name, "sector": sector})
    return profiles


def canonical_hk_leadership(base_label: Any, llm_label: Any = "") -> str:
    preferred = str(llm_label or "").strip()
    fallback = str(base_label or "").strip()
    text = (preferred or fallback).lower()
    tokens = set(_significant_tokens(text))
    if not text:
        return "Leadership could not be determined cleanly"
    if any(phrase in text for phrase in ("growth / internet led", "growth-led", "internet-led")) or (
        {"growth", "internet"} & tokens
    ) or ("platform" in tokens):
        return "Hong Kong growth / internet led"
    if any(phrase in text for phrase in ("state-owned", "old-economy", "soe-led", "value-led", "defensive-led")) or (
        {"soe", "value", "defensive", "financial"} & tokens
    ):
        return "State-owned / old-economy H-shares led"
    if any(token in text for token in ("broad and balanced", "broad beta", "mixed", "balanced")):
        return "Leadership was broad and balanced"
    return preferred or fallback or "Leadership could not be determined cleanly"


def watchlist_story_relevance(name: str, ticker: str, sector: str, title: str, summary: str = "") -> float:
    text = _normalize_text(title, summary)
    compact_text = _compact_text(text)
    tokens = set(_significant_tokens(text))
    score = 0.0

    ticker_root = str(ticker or "").split(".")[0].lower()
    if ticker_root and ticker_root in text:
        score += 3.0

    compact_name = _compact_text(name)
    if compact_name and len(compact_name) >= 5 and compact_name in compact_text:
        score += 3.0

    for token in list(dict.fromkeys(_significant_tokens(name)))[:4]:
        if token in tokens:
            score += 1.0

    for token in list(dict.fromkeys(_significant_tokens(sector)))[:3]:
        if token in tokens:
            score += 0.35

    return score


def sector_story_relevance(
    title: str,
    summary: str,
    sector_label: str,
    coverage_profiles: Iterable[Dict[str, str]],
) -> float:
    text = _normalize_text(title, summary)
    tokens = set(_significant_tokens(text))
    score = 0.0

    sector_tokens: Set[str] = set()
    sector_tokens.update(_SECTOR_HINTS.get(sector_label, set()))
    sector_tokens.update(token for token in _significant_tokens(sector_label) if token not in {"other"})
    sector_matches = sum(1 for token in sector_tokens if token in tokens)
    score += min(sector_matches, 3) * 0.9

    company_match = 0.0
    for profile in coverage_profiles:
        company_match = max(
            company_match,
            watchlist_story_relevance(profile.get("name", ""), profile.get("ticker", ""), profile.get("sector", ""), title, summary),
        )
    score += min(company_match, 4.0)

    if any(term in text for term in _HK_MARKET_TERMS):
        score += 0.5

    return round(score, 2)


def is_relevant_sector_story(
    title: str,
    summary: str,
    sector_label: str,
    coverage_profiles: Iterable[Dict[str, str]],
) -> bool:
    text = _normalize_text(title, summary)
    score = sector_story_relevance(title, summary, sector_label, coverage_profiles)
    if score >= 2.2:
        return True
    if score >= 1.8 and str(sector_label or "").lower() not in {"other"}:
        return True
    if score >= 1.4 and any(term in text for term in _HK_MARKET_TERMS):
        return True
    return False


def is_relevant_llm_story(headline: str, hk_market_impact: str) -> bool:
    impact = str(hk_market_impact or "").strip().lower()
    if any(phrase in impact for phrase in ("lower direct relevance", "low direct relevance", "indirect impact")):
        return False
    text = _normalize_text(headline, hk_market_impact)
    if any(term in text for term in _HK_MARKET_TERMS):
        return True
    if any(term in text for term in ("internet", "tech", "semiconductor", "chip", "ai", "ev", "bank", "insurance", "property", "consumer")):
        return True
    return False
