"""Percentile context for Hong Kong local metrics.

The report called a 17.0% short-selling ratio "elevated" purely because it
cleared a hard-coded 16% threshold. Hong Kong market short-selling routinely
runs in the mid-to-high teens, so the label asserted more than the data showed.
"""

from __future__ import annotations

import json

import _bootstrap  # noqa: F401

from market_diary.professional.metric_history import (
    MIN_SAMPLE_FOR_PERCENTILE,
    describe,
    load_history,
    percentile_context,
    record_observations,
    save_history,
)


def _history_with(values):
    """Build a history whose dates all precede 2026-08-19."""
    observations = {}
    for idx, value in enumerate(values):
        observations[f"2026-06-{idx + 1:02d}"] = value
    return {"observations": {"short_selling_ratio": observations}}


def test_insufficient_history_reports_itself_instead_of_guessing():
    history = _history_with([13.0, 14.0, 15.0])
    context = percentile_context(history, "short_selling_ratio", 17.0, "2026-08-19")
    assert context["available"] is False
    assert context["reason"] == "insufficient_history"
    assert context["sample"] == 3
    assert "3/" in describe(context)


def test_percentile_uses_the_trailing_distribution():
    """17.0% against a distribution centred on 17 is typical, not 'elevated'."""
    sample = [15.0, 16.0, 16.5, 17.0, 17.5, 18.0, 19.0] * 4  # 28 readings, median 17
    history = _history_with(sample)
    context = percentile_context(history, "short_selling_ratio", 17.0, "2026-08-19")
    assert context["available"] is True
    assert context["band"] == "typical"
    assert "typical" in describe(context)


def test_genuinely_extreme_reading_is_flagged():
    history = _history_with([13.0 + (i % 4) * 0.5 for i in range(MIN_SAMPLE_FOR_PERCENTILE + 5)])
    context = percentile_context(history, "short_selling_ratio", 24.0, "2026-08-19")
    assert context["available"] is True
    assert context["band"] == "very high"
    assert context["percentile"] > 90


def test_future_dates_are_excluded_from_the_sample():
    """Today's value must not contribute to its own percentile."""
    history = {"observations": {"short_selling_ratio": {"2026-08-19": 99.0, "2026-08-20": 99.0}}}
    for idx in range(MIN_SAMPLE_FOR_PERCENTILE):
        history["observations"]["short_selling_ratio"][f"2026-07-{idx + 1:02d}"] = 15.0
    context = percentile_context(history, "short_selling_ratio", 99.0, "2026-08-19")
    assert context["sample"] == MIN_SAMPLE_FOR_PERCENTILE
    assert context["percentile"] > 90


def test_record_observations_is_append_only():
    history = {"observations": {}}
    record_observations(history, "2026-08-19", {"short_selling_ratio": {"value": 17.0}})
    record_observations(history, "2026-08-19", {"short_selling_ratio": {"value": 99.0}})
    assert history["observations"]["short_selling_ratio"]["2026-08-19"] == 17.0


def test_record_observations_skips_missing_and_non_numeric():
    history = {"observations": {}}
    record_observations(
        history,
        "2026-08-19",
        {
            "short_selling_ratio": {"value": None},
            "turnover_vs_20d": {"value": "N/A"},
            "hibor_1m": {"value": 2.58},
        },
    )
    assert "short_selling_ratio" not in history["observations"]
    assert "turnover_vs_20d" not in history["observations"]
    assert history["observations"]["hibor_1m"]["2026-08-19"] == 2.58


def test_round_trip_through_disk(tmp_path):
    path = tmp_path / "performance" / "metric_history.json"
    history = {"observations": {}}
    record_observations(history, "2026-08-19", {"short_selling_ratio": {"value": 17.0}})
    save_history(history, str(path))

    reloaded = load_history(str(path))
    assert reloaded["observations"]["short_selling_ratio"]["2026-08-19"] == 17.0


def test_missing_or_corrupt_file_degrades_quietly(tmp_path):
    assert load_history(str(tmp_path / "absent.json"))["observations"] == {}

    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert load_history(str(corrupt))["observations"] == {}


def test_saved_history_is_valid_json(tmp_path):
    path = tmp_path / "performance" / "metric_history.json"
    history = {"observations": {}}
    record_observations(history, "2026-08-19", {"short_selling_ratio": {"value": 17.0}})
    save_history(history, str(path))
    assert json.loads(path.read_text(encoding="utf-8"))["schema_version"]
