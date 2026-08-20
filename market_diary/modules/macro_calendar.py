"""Macro calendar and central-bank event adapters."""

from datetime import datetime
from typing import Dict, List

from market_diary.modules.macro_schedule import scheduled_events, summarize_channels
from market_diary.modules.provenance import provenance_record, unavailable_record

MACRO_SCHEDULE_SOURCE = "Rule-based CN/HK/US release schedule (scheduled dates, no forecast values)"


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
        """Releases scheduled just before the report date.

        Actual, forecast and prior are left empty: no free source for them is
        configured, and inventing them would be worse than omitting them. The
        schedule alone still tells the desk what has just printed.
        """
        return [
            self._to_calendar_row(item)
            for item in self._scheduled(date)
            if item["status"] == "released"
        ]

    def _fetch_upcoming_data(self, date: str) -> List[Dict]:
        """Releases scheduled on or after the report date."""
        return [
            self._to_calendar_row(item)
            for item in self._scheduled(date)
            if item["status"] == "upcoming"
        ]

    def _scheduled(self, date_value: str) -> List[Dict]:
        try:
            reference = datetime.strptime(date_value, "%Y-%m-%d").date()
        except (TypeError, ValueError):
            return []
        return scheduled_events(reference)

    @staticmethod
    def _to_calendar_row(item: Dict) -> Dict:
        return {
            # The schedule carries a date, not an intraday time. Putting the
            # date in "time" made the radar render "2026-08-20 2026-08-20" and
            # "2026-08-20 2026-08-21", because it concatenates date and time.
            "time": "",
            "date": item["date"],
            "country": item["country"],
            "indicator": item["indicator"],
            # No free forecast/actual source is configured; report the gap
            # rather than filling it.
            "actual": "",
            "forecast": "",
            "previous": "",
            "surprise": "inline",
            "impact": item["impact"],
            "channel": item["channel"],
            "channel_note": item["channel_note"],
            "timing_confidence": item["timing_confidence"],
            "note": item["note"],
            "as_of": item["date"],
            "source": MACRO_SCHEDULE_SOURCE,
            "source_url": item.get("source_url", ""),
        }

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

    events = calendar_data.get("released", []) + calendar_data.get("upcoming", [])
    if not events:
        return {
            "status": "unavailable",
            "calendar": calendar_data,
            "central_bank_events": cb_events,
            "formatted_text": calendar.format_for_report(calendar_data, cb_events),
            "provenance": [
                unavailable_record(
                    "Macro calendar",
                    date,
                    "No scheduled release fell inside the report window.",
                )
            ],
        }

    return {
        # "partial" rather than "ok": scheduled dates are available, actual and
        # forecast values are not.
        "status": "partial",
        "calendar": calendar_data,
        "central_bank_events": cb_events,
        "formatted_text": calendar.format_for_report(calendar_data, cb_events),
        "meta": {
            "source": MACRO_SCHEDULE_SOURCE,
            "report_date": date,
            "event_count": len(events),
            "channels": summarize_channels(
                [
                    {"channel": item.get("channel", "fed_path")}
                    for item in events
                    if item.get("channel")
                ]
            ),
            "coverage_note": (
                "Scheduled dates are rule-derived and reliable; actual, forecast and prior values are "
                "not sourced, so surprise cannot be computed. Central-bank speaker events remain "
                "unavailable."
            ),
        },
        "provenance": [
            provenance_record(
                source_name="Macro calendar",
                source_url="",
                as_of=date,
                source_type="derived",
                status="partial_public",
                # Scheduled dates are rule-derived and reliable; the absent
                # forecast and actual values are what hold the confidence down.
                confidence=0.6,
                note=MACRO_SCHEDULE_SOURCE,
            )
        ],
    }
