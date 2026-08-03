"""Macro calendar and central-bank event adapters."""

from datetime import datetime
from typing import Dict, List

from market_diary.modules.provenance import unavailable_record


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
        """Return no release claims until a verified calendar source is configured."""
        return []

    def _fetch_upcoming_data(self, date: str) -> List[Dict]:
        """Return no upcoming claims until a verified calendar source is configured."""
        return []

    def fetch_central_bank_events(self, date: str) -> List[Dict]:
        """Return no central-bank claims until a verified calendar source is configured."""
        return []

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
        "status": "unavailable",
        "calendar": calendar_data,
        "central_bank_events": cb_events,
        "formatted_text": calendar.format_for_report(calendar_data, cb_events),
        "provenance": [
            unavailable_record(
                "Macro calendar",
                date,
                "No verified macro-calendar provider is configured; fabricated fallback events are disabled.",
            )
        ],
    }
