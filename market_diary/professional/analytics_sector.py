from __future__ import annotations

from typing import Any, Dict, List, Tuple

from market_diary.modules.text_normalizer import normalize_news_text
from market_diary.professional.relevance import build_coverage_profiles, is_relevant_sector_story


SECTOR_LABELS = {
    "Technology": "Technology",
    "Financials": "Financials",
    "Healthcare": "Healthcare",
    "Energy": "Energy",
    "Consumer": "Consumer",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "Real Estate": "Real Estate",
    "Other": "Other",
}


def _contains_coverage(text: str, coverage_terms: List[str]) -> bool:
    lowered = text.lower()
    return any(term and term.lower() in lowered for term in coverage_terms)


def _news_grade(score: float) -> str:
    if score >= 3.5:
        return "A"
    if score >= 1.8:
        return "B"
    return "C"


def _news_importance(text: str, sector_label: str) -> Tuple[str, str]:
    lowered = text.lower()
    if any(word in lowered for word in ("earnings", "guidance", "profit warning", "results", "outlook")):
        return "This can directly reshape earnings forecasts and the valuation framework", "Short-term catalyst"
    if any(word in lowered for word in ("upgrade", "downgrade", "price target", "rating")):
        return "This signals a marginal shift in sell-side consensus", "Short-term catalyst"
    if any(word in lowered for word in ("merger", "acquisition", "deal", "placement", "buyback")):
        return "Capital allocation or shareholder-return events can trigger a valuation reset", "Medium-term trend"
    if any(word in lowered for word in ("regulation", "approval", "policy", "rulemaking")):
        return f"Policy and regulation can reshape the {sector_label} industry logic", "Medium-term trend"
    if any(word in lowered for word in ("launch", "product", "order", "contract", "capacity")):
        return "This is more about order visibility and product-cycle validation", "Short-term catalyst"
    return f"Assess whether the story can propagate through the {sector_label} value chain", "Monitor"


def build_sector_news_digest(sector_data: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    watchlists = config.get("watchlists", {}) or {}
    coverage_terms: List[str] = []
    for bucket_items in watchlists.values():
        for item in bucket_items:
            coverage_terms.append(item.get("name", ""))
            coverage_terms.append(item.get("ticker", "").split(".")[0])
    coverage_profiles = build_coverage_profiles(watchlists)

    graded_news: List[Dict[str, Any]] = []
    sector_news = (sector_data or {}).get("sector_news", {}) or {}
    for sector, news_items in sector_news.items():
        sector_label = SECTOR_LABELS.get(sector, sector)
        for news in news_items:
            title = normalize_news_text(news.get("title", ""), strip_html_tags=True)
            summary = normalize_news_text(news.get("summary", ""), strip_html_tags=True)
            text = f"{title} {summary}"
            if not is_relevant_sector_story(title, summary, sector_label, coverage_profiles):
                continue
            score = float(news.get("importance_score", 0.0))
            if _contains_coverage(text, coverage_terms):
                score += 1.5
            if any(token in text.lower() for token in ("earnings", "guidance", "merger", "deal", "regulation", "approval", "upgrade", "downgrade", "buyback", "placement", "results")):
                score += 1.2
            why, horizon = _news_importance(text, sector_label)
            graded_news.append(
                {
                    "sector": sector_label,
                    "title": title,
                    "summary": summary,
                    "grade": _news_grade(score),
                    "why": why,
                    "impact_target": f"Map first into {sector_label} leaders and close peers",
                    "horizon": horizon,
                    "score": round(score, 2),
                    "source": normalize_news_text(news.get("source", ""), strip_html_tags=False),
                    "url": normalize_news_text(news.get("link", ""), strip_html_tags=False),
                }
            )

    graded_news.sort(key=lambda item: item.get("score", 0), reverse=True)

    sell_side: List[Dict[str, Any]] = []
    for change in (sector_data or {}).get("analyst_changes", []) or []:
        sell_side.append(
            {
                "ticker": change.get("ticker", ""),
                "firm": change.get("firm", ""),
                "action": change.get("action", ""),
                "summary": f"{change.get('from_rating', '')} -> {change.get('to_rating', '')}",
                "target_change": f"{change.get('previous_target', '')} -> {change.get('price_target', '')}",
            }
        )

    return {
        "graded_news": graded_news,
        "sell_side": sell_side,
        "earnings_calendar": (sector_data or {}).get("earnings_calendar", []) or [],
    }
