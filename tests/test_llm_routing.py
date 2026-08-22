"""The narrative tasks must run on the model they were designed for.

Seven tasks, four of which named no provider. ``get_default_provider`` preferred
MiniMax whenever its key existed, and the route fallback defaulted to
MiniMax-M3 — a reasoning model whose reasoning tokens count against
``max_tokens``. Those four tasks spent their budget before emitting JSON and
failed as truncated every day, while the three that named DeepSeek worked. CI
reported exactly that split: 1 succeeded, 4 truncated, 2 skipped.
"""

from __future__ import annotations

import os

import _bootstrap  # noqa: F401
import pytest

from market_diary.modules.llm_client import (
    get_available_providers,
    get_default_base_url,
    get_default_model,
    get_default_provider,
)
from market_diary.professional.config import DEFAULT_LLM_ROUTE_FALLBACK, load_professional_config

ENV_KEYS = [
    "DEEPSEEK_API_KEY",
    "MINIMAX_API_KEY",
    "OPENAI_API_KEY",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "OPENAI_BASE_URL",
    "LLM_PRIMARY_PROVIDER",
]


@pytest.fixture
def clean_env(monkeypatch):
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    return monkeypatch


def test_both_keys_present_resolves_to_deepseek(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "d")
    clean_env.setenv("MINIMAX_API_KEY", "m")
    assert get_default_provider() == "deepseek"
    assert get_default_model() == "deepseek-v4-pro"
    assert get_available_providers() == ["deepseek", "minimax"]


def test_minimax_remains_the_fallback(clean_env):
    """MiniMax is not removed, only demoted."""
    clean_env.setenv("MINIMAX_API_KEY", "m")
    assert get_default_provider() == "minimax"
    assert get_default_model() == "MiniMax-M3"


def test_explicit_primary_provider_is_still_honoured(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "d")
    clean_env.setenv("MINIMAX_API_KEY", "m")
    clean_env.setenv("LLM_PRIMARY_PROVIDER", "minimax")
    assert get_default_provider() == "minimax"


def test_route_fallback_defaults_to_deepseek():
    assert DEFAULT_LLM_ROUTE_FALLBACK["default"] == "deepseek-v4-pro"


def test_every_task_names_its_provider():
    """An implicit default is what routed four tasks to the wrong model."""
    tasks = load_professional_config()["llm"]["tasks"]
    missing = [name for name, task in tasks.items() if not task.get("provider")]
    assert not missing, f"these tasks would fall back to whichever key exists: {missing}"
    assert {task["provider"] for task in tasks.values()} == {"deepseek"}


def test_stale_base_url_for_another_provider_is_ignored(clean_env):
    """A MiniMax URL must not be used for a DeepSeek call."""
    clean_env.setenv("DEEPSEEK_API_KEY", "d")
    clean_env.setenv("MINIMAX_API_KEY", "m")
    clean_env.setenv("LLM_BASE_URL", "https://api.minimaxi.com/v1")
    assert get_default_base_url("deepseek") == "https://api.deepseek.com"


def test_neutral_custom_base_url_is_respected(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "d")
    clean_env.setenv("LLM_BASE_URL", "https://proxy.internal/v1")
    assert get_default_base_url("deepseek") == "https://proxy.internal/v1"


class TestTokenUsageIsRecorded:
    """Budget questions must be measurable rather than inferred."""

    def test_usage_is_captured_with_reasoning_tokens(self):
        from market_diary.professional.llm_enhancer import _token_usage

        class Details:
            reasoning_tokens = 380

        class Usage:
            prompt_tokens = 900
            completion_tokens = 1400
            total_tokens = 2300
            completion_tokens_details = Details()

        class Response:
            usage = Usage()

        usage = _token_usage(Response(), 1400)
        assert usage["completion_tokens"] == 1400
        assert usage["reasoning_tokens"] == 380
        assert usage["budget_used_pct"] == 100.0

    def test_missing_usage_degrades_quietly(self):
        from market_diary.professional.llm_enhancer import _token_usage

        assert _token_usage(object(), 1400) == {"max_tokens": 1400}

    def test_health_surfaces_budget_pressure(self):
        from market_diary.professional.llm_enhancer import build_llm_health

        health = build_llm_health(
            {
                "tasks": {
                    "overnight_review": {
                        "status": "error",
                        "error_class": "truncated",
                        "usage": {"budget_used_pct": 100.0, "completion_tokens": 1400, "max_tokens": 1400},
                    }
                }
            }
        )
        assert health["token_budget_by_task"]["overnight_review"]["budget_used_pct"] == 100.0


class TestTruncationBudget:
    """Measured, not assumed.

    After routing was fixed, company_commentary — which had always named
    DeepSeek — still failed with reasoning_tokens 1399 of max_tokens 1400. The
    usage instrumentation showed reasoning consuming the whole allowance before
    any JSON was emitted, so the budgets were genuinely too small and the earlier
    decision to leave them alone was wrong.
    """

    def test_budgets_leave_room_for_reasoning_plus_output(self):
        tasks = load_professional_config()["llm"]["tasks"]
        # Reasoning alone was measured at ~1400 tokens; the JSON needs several
        # hundred more on top of it.
        too_small = {name: t["max_tokens"] for name, t in tasks.items() if t["max_tokens"] < 2500}
        assert not too_small, f"these budgets cannot fit reasoning plus output: {too_small}"

    def test_retry_escalates_rather_than_repeating_the_same_budget(self):
        from market_diary.professional.llm_enhancer import (
            MAX_TOKENS_CEILING,
            TRUNCATION_RETRY_MULTIPLIER,
        )

        assert TRUNCATION_RETRY_MULTIPLIER > 1.0
        # A retry at the same budget would truncate identically.
        assert int(1400 * TRUNCATION_RETRY_MULTIPLIER) > 1400
        assert MAX_TOKENS_CEILING > max(
            t["max_tokens"] for t in load_professional_config()["llm"]["tasks"].values()
        )

    def test_ceiling_bounds_the_escalation(self):
        from market_diary.professional.llm_enhancer import (
            MAX_TOKENS_CEILING,
            TRUNCATION_RETRY_MULTIPLIER,
        )

        budget = 4000
        for _ in range(10):
            budget = min(int(budget * TRUNCATION_RETRY_MULTIPLIER), MAX_TOKENS_CEILING)
        assert budget == MAX_TOKENS_CEILING
