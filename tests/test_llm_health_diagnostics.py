"""Failure causes must be countable, not just counted.

The narrative overlay ran at 0-2 of 7 successful tasks for fifteen consecutive
days. The run summary reported only "4 error(s)", which cannot separate a
provider outage from a response that never parses, so there was no way to know
which fix would help.
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import pytest

from market_diary.professional.llm_enhancer import build_llm_health, classify_error


@pytest.mark.parametrize(
    "message,expected",
    [
        ("APITimeoutError: Request timed out after 60s", "timeout"),
        ("RateLimitError: 429 Too Many Requests", "rate_limit"),
        ("APIStatusError: 529 overloaded_error", "overloaded"),
        ("JSONDecodeError: Expecting value: line 1 column 1", "json_parse"),
        ("ValueError: LLM response appears truncated (max_tokens)", "truncated"),
        ("AuthenticationError: 401 invalid api key", "auth"),
        ("APIStatusError: 500 internal server error", "http_5xx"),
        ("ConnectionError: connection reset by peer", "connection"),
        ("KeyError: missing field paragraph", "schema_mismatch"),
        ("", "unknown"),
        ("SomethingNobodyAnticipated: weird", "other"),
    ],
)
def test_error_messages_are_classified(message, expected):
    assert classify_error(message) == expected


def test_health_counts_outcomes_and_names_the_dominant_cause():
    task_meta = {
        "tasks": {
            "news_selection": {"status": "ok"},
            "macro_interpretation": {"status": "cached"},
            "company_commentary": {"status": "error", "error_class": "timeout"},
            "theme_deep_dive": {"status": "error", "error_class": "timeout"},
            "overnight_review": {"status": "error", "error_class": "json_parse"},
            "hk_review": {"status": "skipped"},
        }
    }
    health = build_llm_health(task_meta)

    assert health["succeeded"] == 2
    assert health["failed"] == 3
    assert health["skipped"] == 1
    # Skipped tasks are not counted as attempts.
    assert health["success_rate_pct"] == 40.0
    assert health["dominant_failure_class"] == "timeout"
    assert health["failures_by_class"] == {"timeout": 2, "json_parse": 1}
    assert health["failure_class_by_task"]["overnight_review"] == "json_parse"


def test_health_derives_the_class_when_only_a_message_exists():
    task_meta = {"tasks": {"a": {"status": "error", "error": "APITimeoutError: timed out"}}}
    assert build_llm_health(task_meta)["failures_by_class"] == {"timeout": 1}


def test_healthy_run_reports_no_dominant_cause():
    task_meta = {"tasks": {"a": {"status": "ok"}, "b": {"status": "ok"}}}
    health = build_llm_health(task_meta)
    assert health["success_rate_pct"] == 100.0
    assert health["dominant_failure_class"] == ""
    assert health["failures_by_class"] == {}


def test_empty_meta_does_not_crash():
    health = build_llm_health({})
    assert health["tasks_total"] == 0
    assert health["success_rate_pct"] == 0.0
