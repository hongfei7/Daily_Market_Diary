"""The archive parser must keep up with the report format.

The report header was reworded on 2026-08-01 from ``Market effective date: `X` ``
to ``Market effective `X` ``. ``parse_archived_report`` only matched the old
spelling, so from that date every published report yielded no observation and no
signal. The performance ledger silently froze on pre-August data: 20 days of
published calls became invisible, and the hit rates in the appendix stopped
reflecting recent work.

Nothing failed loudly, because a parser that returns ``None`` looks the same as a
report that had nothing to contribute. This test closes that gap by requiring
every archived report to parse.
"""

from __future__ import annotations

from pathlib import Path

import _bootstrap  # noqa: F401
import pytest

from market_diary.professional.performance import parse_archived_report

ARCHIVE = Path(__file__).resolve().parents[1] / "reports_professional" / "archive"
REPORTS = sorted(ARCHIVE.glob("*/morning_briefing.md"))


@pytest.mark.skipif(not REPORTS, reason="no archived reports available")
def test_every_archived_report_yields_an_observation_and_signal():
    failures = []
    for path in REPORTS:
        observation, signal = parse_archived_report(path)
        if observation is None or signal is None:
            failures.append(
                f"{path.parent.name}: observation={'ok' if observation else 'MISSING'}, "
                f"signal={'ok' if signal else 'MISSING'}"
            )
    assert not failures, "archive parser drifted from the report format:\n" + "\n".join(failures)


@pytest.mark.skipif(not REPORTS, reason="no archived reports available")
def test_parsed_observations_carry_usable_prices():
    for path in REPORTS[-10:]:
        observation, _ = parse_archived_report(path)
        assert observation is not None
        prices = observation["prices"]
        assert prices, f"{path.parent.name} produced no benchmark prices"
        assert all(value > 0 for value in prices.values())


@pytest.mark.parametrize(
    "header",
    [
        "_Data through: global `2026-08-18` | Market effective `2026-08-18` | Generated `x`_",
        "> Briefing date: `2026-07-31` | Market effective date: `2026-07-30` | HK: `x`",
    ],
    ids=["current_format", "pre_august_format"],
)
def test_both_header_spellings_are_accepted(tmp_path, header):
    """Both the current and the pre-2026-08-01 header must parse."""
    report = tmp_path / "morning_briefing.md"
    report.write_text(
        "\n".join(
            [
                "# Morning Research Workbench | 2026-08-19",
                header,
                "**Composite risk score:** `27.4/100` | **Regime:** `Risk-off`",
                "### 1.2 Global Asset Price Dashboard",
                "| Signal | Last / move |",
                "| --- | --- |",
                "| Hang Seng Index | 25,453.23 / +1.34% |",
                "### 1.3 Hong Kong Key Data Quick Check",
            ]
        ),
        encoding="utf-8",
    )
    observation, signal = parse_archived_report(report)
    assert observation is not None
    assert signal is not None
    assert observation["prices"]["Hang Seng Index"] == pytest.approx(25453.23)
