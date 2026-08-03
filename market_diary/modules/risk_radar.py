"""Risk radar and event watch adapters."""

from datetime import datetime
from typing import Dict, List

from market_diary.modules.provenance import unavailable_record


class RiskRadar:
    """Build a compact risk watchlist for the morning briefing."""

    KEY_LEVELS: Dict[str, Dict[str, List[float]]] = {}

    def fetch_geopolitical_risks(self) -> List[Dict]:
        """Return no geopolitical claims without a dated news source."""
        return []

    def fetch_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """Return no events without a verified event-calendar source."""
        return []

    def fetch_technical_levels(self, current_prices: Dict) -> Dict:
        """Compare current prices with a small set of risk-monitoring thresholds."""
        analysis: Dict = {}

        for symbol, levels in self.KEY_LEVELS.items():
            current = current_prices.get(symbol)
            if current is None:
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
                datetime.now().date().isoformat(),
                "No verified geopolitical, event-calendar, sentiment, or technical-level provider is configured.",
            )
        ],
    }
