"""Macro calendar and central-bank event adapters."""

from datetime import datetime
from typing import Dict, List


class MacroCalendar:
    """Provide a lightweight macro calendar payload for the morning briefing."""

    def fetch_economic_calendar(self, date: str) -> Dict:
        """Return released and upcoming macro events for the requested date."""
        released_data = self._fetch_released_data(date)
        upcoming_data = self._fetch_upcoming_data(date)

        return {
            "released": released_data,
            "upcoming": upcoming_data,
            "meta": {
                "date": date,
                "fetch_time": datetime.now().isoformat(),
            },
        }

    def _fetch_released_data(self, date: str) -> List[Dict]:
        """Return placeholder released data until a premium calendar is wired in."""
        return [
            {
                "time": "09:30",
                "country": "China",
                "indicator": "Trade Balance",
                "actual": "USD 85.2bn",
                "forecast": "USD 79.0bn",
                "previous": "USD 80.6bn",
                "impact": "medium",
                "surprise": "beat",
            },
            {
                "time": "20:30",
                "country": "US",
                "indicator": "CPI MoM",
                "actual": "0.3%",
                "forecast": "0.2%",
                "previous": "0.4%",
                "impact": "high",
                "surprise": "beat",
            },
        ]

    def _fetch_upcoming_data(self, date: str) -> List[Dict]:
        """Return placeholder upcoming data for the requested date."""
        return [
            {
                "time": "09:20",
                "country": "China",
                "indicator": "Loan Prime Rate",
                "forecast": "3.45%",
                "previous": "3.45%",
                "impact": "medium",
            },
            {
                "time": "20:30",
                "country": "US",
                "indicator": "Retail Sales MoM",
                "forecast": "0.3%",
                "previous": "0.6%",
                "impact": "high",
            },
        ]

    def fetch_central_bank_events(self, date: str) -> List[Dict]:
        """Return placeholder central-bank events and speeches."""
        return [
            {
                "time": "10:00",
                "bank": "PBOC",
                "event_type": "liquidity",
                "speaker": "Open Market Operations Desk",
                "title": "Daily liquidity operation window",
                "importance": "medium",
            },
            {
                "time": "22:00",
                "bank": "Federal Reserve",
                "event_type": "speech",
                "speaker": "Jerome Powell",
                "title": "Economic outlook remarks",
                "importance": "high",
            },
        ]

    def format_for_report(self, calendar_data: Dict, cb_events: List[Dict]) -> str:
        """Format the macro payload into a readable fallback text block."""
        lines: List[str] = []

        if calendar_data.get("released"):
            lines.append("### Released Data")
            lines.append("")
            lines.append("| Time | Country | Indicator | Actual | Forecast | Prior | Surprise |")
            lines.append("|------|---------|-----------|--------|----------|-------|----------|")
            for item in calendar_data["released"]:
                surprise = item.get("surprise", "inline").upper()
                lines.append(
                    f"| {item['time']} | {item['country']} | {item['indicator']} | "
                    f"{item['actual']} | {item['forecast']} | {item['previous']} | {surprise} |"
                )
            lines.append("")

        if calendar_data.get("upcoming"):
            lines.append("### Upcoming Data")
            lines.append("")
            lines.append("| Time | Country | Indicator | Forecast | Prior | Importance |")
            lines.append("|------|---------|-----------|----------|-------|------------|")
            for item in calendar_data["upcoming"]:
                lines.append(
                    f"| {item['time']} | {item['country']} | {item['indicator']} | "
                    f"{item['forecast']} | {item['previous']} | {item['impact'].upper()} |"
                )
            lines.append("")

        if cb_events:
            lines.append("### Central Bank Events")
            lines.append("")
            for event in cb_events:
                lines.append(
                    f"- **{event['time']}** | {event['bank']} | "
                    f"{event['speaker']}: {event['title']}"
                )
            lines.append("")

        return "\n".join(lines)


def fetch_macro_data(date: str) -> Dict:
    """Public entry point for macro calendar data."""
    calendar = MacroCalendar()
    calendar_data = calendar.fetch_economic_calendar(date)
    cb_events = calendar.fetch_central_bank_events(date)

    return {
        "calendar": calendar_data,
        "central_bank_events": cb_events,
        "formatted_text": calendar.format_for_report(calendar_data, cb_events),
    }
