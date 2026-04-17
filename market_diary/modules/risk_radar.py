"""Risk radar and event watch adapters."""

from datetime import datetime, timedelta
from typing import Dict, List


class RiskRadar:
    """Build a compact risk watchlist for the morning briefing."""

    KEY_LEVELS = {
        "HSI": {
            "resistance": [17500, 18000, 18500],
            "support": [16800, 16500, 16000],
        },
        "HSCEI": {
            "resistance": [6400, 6600, 6800],
            "support": [6100, 6000, 5850],
        },
        "DXY": {
            "resistance": [104.5, 105.0, 105.5],
            "support": [103.0, 102.5, 102.0],
        },
        "US10Y": {
            "resistance": [4.40, 4.50, 4.60],
            "support": [4.20, 4.10, 4.00],
        },
        "USD/HKD": {
            "resistance": [7.85],
            "support": [7.80, 7.78, 7.75],
        },
    }

    def fetch_geopolitical_risks(self) -> List[Dict]:
        """Return placeholder geopolitical risks."""
        return [
            {
                "region": "Middle East",
                "event": "Regional tension remains elevated",
                "risk_level": "high",
                "impact": "Oil, freight costs, and safe-haven demand",
                "last_update": datetime.now().isoformat(),
            },
            {
                "region": "US-China",
                "event": "Technology export-control rhetoric remains active",
                "risk_level": "medium",
                "impact": "China internet, semiconductors, and supply chains",
                "last_update": datetime.now().isoformat(),
            },
        ]

    def fetch_upcoming_events(self, days_ahead: int = 7) -> List[Dict]:
        """Return a rolling list of near-term market events."""
        today = datetime.now().date()
        events = [
            {
                "date": (today + timedelta(days=1)).strftime("%Y-%m-%d"),
                "type": "China Data",
                "description": "China activity and credit-related macro releases",
                "importance": "high",
            },
            {
                "date": (today + timedelta(days=2)).strftime("%Y-%m-%d"),
                "type": "Fed Speakers",
                "description": "Federal Reserve speakers on rates and growth",
                "importance": "medium",
            },
            {
                "date": (today + timedelta(days=4)).strftime("%Y-%m-%d"),
                "type": "Options Expiry",
                "description": "Monthly options expiration",
                "importance": "high",
            },
            {
                "date": (today + timedelta(days=6)).strftime("%Y-%m-%d"),
                "type": "Policy Watch",
                "description": "Mainland policy window and liquidity signals",
                "importance": "medium",
            },
        ]

        cutoff = today + timedelta(days=days_ahead)
        filtered = []
        for event in events:
            event_date = datetime.strptime(event["date"], "%Y-%m-%d").date()
            if today <= event_date <= cutoff:
                filtered.append(event)

        return filtered

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
        """Return placeholder sentiment readings."""
        return {
            "aaii_bull_bear": {
                "bullish": 42.5,
                "neutral": 28.3,
                "bearish": 29.2,
                "interpretation": "Neutral to slightly bullish",
            },
            "fear_greed_index": {
                "value": 58,
                "level": "Greed",
            },
            "put_call_ratio": {
                "equity": 0.68,
                "index": 1.15,
                "interpretation": "Single-stock optimism with index hedging",
            },
            "vix_term_structure": {
                "front_month": 19.2,
                "second_month": 20.5,
                "slope": "contango",
                "interpretation": "Volatility curve still implies contained stress",
            },
        }

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
        "geopolitical_risks": geo_risks,
        "upcoming_events": events,
        "technical_levels": tech_levels,
        "sentiment_indicators": sentiment,
        "formatted_text": radar.format_for_report(geo_risks, events, tech_levels, sentiment),
    }
