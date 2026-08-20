"""Risk radar and event watch adapters."""

import math
from datetime import date, datetime
from typing import Dict, List, Optional

from market_diary.modules.macro_schedule import scheduled_events
from market_diary.modules.provenance import provenance_record, unavailable_record

RISK_LEVELS_SOURCE = "Derived reference levels (round-number grid) and rule-based release schedule"


# Round-number levels act as reference points for each instrument. Deriving them
# from the price itself avoids a hard-coded table that silently goes stale, which
# is why ``KEY_LEVELS`` was left empty and the whole feed reported as
# unavailable every day.
_ROUND_STEPS: Dict[str, float] = {
    "SPX": 100.0,
    "NDX": 500.0,
    "HSI": 500.0,
    "HSCEI": 200.0,
    "DXY": 1.0,
    "US10Y": 0.25,
}
# USD/HKD is governed by the Convertibility Undertakings, not round numbers.
_PEG_LEVELS = {"USD/HKD": {"support": [7.7500], "resistance": [7.8500]}}


def _derive_levels(symbol: str, price: float) -> Dict[str, List[float]]:
    """Build support/resistance around a price from its own round-number grid."""
    if symbol in _PEG_LEVELS:
        return _PEG_LEVELS[symbol]
    step = _ROUND_STEPS.get(symbol)
    if not step or price <= 0:
        return {"support": [], "resistance": []}
    base = math.floor(price / step) * step
    supports = [round(base - step * i, 4) for i in range(0, 2) if base - step * i > 0]
    resistances = [round(base + step * i, 4) for i in range(1, 3)]
    return {"support": supports, "resistance": resistances}


class RiskRadar:
    """Build a compact risk watchlist for the morning briefing."""

    def fetch_geopolitical_risks(self) -> List[Dict]:
        """Return no geopolitical claims without a dated news source."""
        return []

    def fetch_upcoming_events(self, days_ahead: int = 7, reference: Optional[date] = None) -> List[Dict]:
        """Scheduled macro releases that can reprice Hong Kong in the next week.

        These come from the same rule-based schedule the macro section uses, so
        the risk feed and the calendar cannot disagree.
        """
        anchor = reference or datetime.now().date()
        events = scheduled_events(anchor, days_back=0, days_forward=days_ahead)
        return [
            {
                "date": item["date"],
                "type": item["channel"],
                "description": f"{item['country']} {item['indicator']}",
                "importance": item["impact"],
                "channel_note": item["channel_note"],
                "timing_confidence": item["timing_confidence"],
                # A dated event must stay traceable to its publisher.
                "source": "Rule-based CN/HK/US release schedule",
                "source_url": item.get("source_url", ""),
            }
            for item in events
            if item["status"] == "upcoming"
        ]

    def fetch_technical_levels(self, current_prices: Dict) -> Dict:
        """Locate each price against its nearest derived support and resistance."""
        analysis: Dict = {}

        for symbol, current in (current_prices or {}).items():
            try:
                current = float(current)
            except (TypeError, ValueError):
                continue
            if current <= 0:
                continue
            levels = _derive_levels(symbol, current)
            if not levels["support"] and not levels["resistance"]:
                continue

            resistance_candidates = [level for level in levels["resistance"] if level > current]
            support_candidates = [level for level in levels["support"] if level < current]
            nearest_resistance = min(resistance_candidates, default=None)
            nearest_support = max(support_candidates, default=None)

            analysis[symbol] = {
                "current": current,
                "nearest_resistance": nearest_resistance,
                "nearest_support": nearest_support,
                "resistance_distance": round((nearest_resistance / current - 1) * 100, 2)
                if nearest_resistance
                else None,
                "support_distance": round((current / nearest_support - 1) * 100, 2)
                if nearest_support
                else None,
            }

        return analysis

    def fetch_sentiment_indicators(self) -> Dict:
        """Return no sentiment readings without a verified market-data source."""
        return {}

    def format_for_report(
        self,
        geo_risks: List[Dict],
        events: List[Dict],
        tech_levels: Dict,
        sentiment: Dict,
    ) -> str:
        """Format the risk payload into a readable fallback block."""
        lines: List[str] = ["### Risk Radar", ""]

        if geo_risks:
            lines.append("#### Geopolitical Risks")
            lines.append("")
            for risk in geo_risks:
                lines.append(
                    f"- **{risk['region']}**: {risk['event']} | Impact: {risk['impact']} | "
                    f"Severity: {risk['risk_level']}"
                )
            lines.append("")

        if events:
            lines.append("#### Upcoming Events")
            lines.append("")
            lines.append("| Date | Type | Description | Importance |")
            lines.append("|------|------|-------------|------------|")
            for event in events:
                lines.append(
                    f"| {event['date']} | {event['type']} | {event['description']} | "
                    f"{event['importance']} |"
                )
            lines.append("")

        if tech_levels:
            lines.append("#### Key Levels")
            lines.append("")
            lines.append("| Asset | Current | Nearest Resistance | Nearest Support |")
            lines.append("|-------|---------|--------------------|-----------------|")
            for symbol, data in tech_levels.items():
                resistance = (
                    f"{data['nearest_resistance']:.2f} ({data['resistance_distance']:+.2f}%)"
                    if data["nearest_resistance"] is not None
                    else "N/A"
                )
                support = (
                    f"{data['nearest_support']:.2f} ({-data['support_distance']:+.2f}%)"
                    if data["nearest_support"] is not None
                    else "N/A"
                )
                lines.append(
                    f"| {symbol} | {data['current']:.2f} | {resistance} | {support} |"
                )
            lines.append("")

        if sentiment:
            lines.append("#### Sentiment")
            lines.append("")
            aaii = sentiment.get("aaii_bull_bear")
            if aaii:
                lines.append(
                    f"- AAII: bullish {aaii['bullish']:.1f}% | neutral {aaii['neutral']:.1f}% | "
                    f"bearish {aaii['bearish']:.1f}% | {aaii['interpretation']}"
                )

            fear_greed = sentiment.get("fear_greed_index")
            if fear_greed:
                lines.append(
                    f"- Fear and Greed: {fear_greed['value']} ({fear_greed['level']})"
                )

            put_call = sentiment.get("put_call_ratio")
            if put_call:
                lines.append(
                    f"- Put/Call: equity {put_call['equity']:.2f} | index {put_call['index']:.2f} | "
                    f"{put_call['interpretation']}"
                )

            vix_curve = sentiment.get("vix_term_structure")
            if vix_curve:
                lines.append(
                    f"- VIX curve: front {vix_curve['front_month']:.1f} | "
                    f"second {vix_curve['second_month']:.1f} | {vix_curve['slope']} | "
                    f"{vix_curve['interpretation']}"
                )
            lines.append("")

        return "\n".join(lines)


def fetch_risk_data(current_prices: Dict = None) -> Dict:
    """Public entry point for risk monitoring data."""
    radar = RiskRadar()
    current_prices = current_prices or {}

    geo_risks = radar.fetch_geopolitical_risks()
    events = radar.fetch_upcoming_events(days_ahead=7)
    tech_levels = radar.fetch_technical_levels(current_prices)
    sentiment = radar.fetch_sentiment_indicators()

    as_of = datetime.now().date().isoformat()

    if not events and not tech_levels:
        return {
            "status": "unavailable",
            "geopolitical_risks": geo_risks,
            "upcoming_events": events,
            "technical_levels": tech_levels,
            "sentiment_indicators": sentiment,
            "formatted_text": radar.format_for_report(geo_risks, events, tech_levels, sentiment),
            "provenance": [
                unavailable_record(
                    "Risk and sentiment event feed",
                    as_of,
                    "No prices were supplied and no release fell inside the forward window.",
                )
            ],
        }

    return {
        # "partial": levels and the event schedule are derived, but geopolitical
        # and sentiment feeds still have no verified provider.
        "status": "partial",
        "geopolitical_risks": geo_risks,
        "upcoming_events": events,
        "technical_levels": tech_levels,
        "sentiment_indicators": sentiment,
        "formatted_text": radar.format_for_report(geo_risks, events, tech_levels, sentiment),
        "meta": {
            "source": RISK_LEVELS_SOURCE,
            "as_of": as_of,
            "level_basis": "round_number_grid",
            "event_basis": "rule_based_schedule",
            "coverage_note": (
                "Reference levels are derived from a round-number grid, not from technical analysis of "
                "price history. Geopolitical risk and market sentiment indicators remain unsourced."
            ),
        },
        "provenance": [
            provenance_record(
                source_name="Risk and sentiment event feed",
                source_url="",
                as_of=as_of,
                source_type="derived",
                status="partial_public",
                # Levels and the event schedule are derived; geopolitical risk
                # and sentiment remain unsourced, hence the low confidence.
                confidence=0.5,
                note=RISK_LEVELS_SOURCE,
            )
        ],
    }
