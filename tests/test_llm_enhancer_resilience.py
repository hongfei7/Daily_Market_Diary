import os
import sys
import tempfile

from _bootstrap import ROOT  # noqa: F401

from market_diary.modules.llm_client import get_available_providers, get_default_base_url, get_default_model
from professional.config import load_professional_config
from professional.llm_enhancer import (
    _cache_path,
    _effective_max_workers,
    _extract_json_object,
    _llm_response_looks_truncated,
    _model_candidates,
)


def test_extract_json_object_accepts_code_fence_and_python_dict() -> None:
    fenced = """```json
{"paragraph":"Desk tone","drivers":["Rates eased"],"hk_open_implication":"Constructive."}
```"""
    pythonish = "{'paragraph': 'Desk tone', 'drivers': ['Rates eased'], 'hk_open_implication': 'Constructive.'}"

    fenced_payload = _extract_json_object(fenced)
    pythonish_payload = _extract_json_object(pythonish)

    assert fenced_payload.get("paragraph") == "Desk tone"
    assert pythonish_payload.get("hk_open_implication") == "Constructive."


def _preserve_env(names):
    return {name: os.environ.get(name) for name in names}


def _restore_env(snapshot) -> None:
    for name, value in snapshot.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value


def test_effective_max_workers_caps_minimax_parallelism() -> None:
    env_names = ["DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_BASE_URL"]
    prior_env = _preserve_env(env_names)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ.pop("MINIMAX_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ["LLM_MODEL"] = "MiniMax-M2.7"
    os.environ.pop("LLM_BASE_URL", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    try:
        workers = _effective_max_workers({"max_workers": 4, "provider_parallelism": {"minimax": 1}}, ["news_selection"])
    finally:
        _restore_env(prior_env)

    assert workers == 1


def test_effective_max_workers_caps_minimax_fallback_route() -> None:
    env_names = ["DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_BASE_URL"]
    prior_env = _preserve_env(env_names)
    os.environ.pop("DEEPSEEK_API_KEY", None)
    os.environ["MINIMAX_API_KEY"] = "test-minimax-key"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("LLM_MODEL", None)
    os.environ.pop("LLM_BASE_URL", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    try:
        workers = _effective_max_workers(load_professional_config()["llm"], ["news_selection"])
    finally:
        _restore_env(prior_env)

    assert workers == 1


def test_deepseek_defaults_when_secret_present() -> None:
    env_names = ["DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_BASE_URL"]
    prior_env = _preserve_env(env_names)
    os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
    os.environ.pop("MINIMAX_API_KEY", None)
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("LLM_MODEL", None)
    os.environ.pop("LLM_BASE_URL", None)
    os.environ.pop("OPENAI_BASE_URL", None)
    try:
        model = get_default_model()
        base_url = get_default_base_url()
        workers = _effective_max_workers(load_professional_config()["llm"], ["news_selection"])
    finally:
        _restore_env(prior_env)

    assert model == "deepseek-v4-pro"
    assert base_url == "http://api.deepseek.com"
    assert workers == 4


def test_deepseek_adds_minimax_fallback_candidate() -> None:
    env_names = ["DEEPSEEK_API_KEY", "MINIMAX_API_KEY", "OPENAI_API_KEY", "LLM_MODEL", "LLM_BASE_URL", "OPENAI_BASE_URL"]
    prior_env = _preserve_env(env_names)
    os.environ["DEEPSEEK_API_KEY"] = "test-deepseek-key"
    os.environ["MINIMAX_API_KEY"] = "test-minimax-key"
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("LLM_MODEL", None)
    os.environ["LLM_BASE_URL"] = "http://api.deepseek.com"
    os.environ.pop("OPENAI_BASE_URL", None)
    try:
        candidates = _model_candidates(load_professional_config()["llm"], "news_selection")
        providers = get_available_providers()
        fallback_base_url = get_default_base_url("minimax")
    finally:
        _restore_env(prior_env)

    assert providers == ["deepseek", "minimax"]
    assert candidates == [
        ("deepseek", "deepseek-v4-pro", "default_model"),
        ("minimax", "MiniMax-M2.7", "default_model:fallback"),
    ]
    assert fallback_base_url == "https://api.minimaxi.com/v1"


def test_cache_path_changes_when_prompt_changes() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        base_context = {"overview": {"theme": "Rates"}}
        path_one = _cache_path(tmpdir, "overnight_review", base_context, "MiniMax-M2.7", "Prompt A")
        path_two = _cache_path(tmpdir, "overnight_review", base_context, "MiniMax-M2.7", "Prompt B")

    assert path_one != path_two


def test_llm_response_truncation_guard() -> None:
    assert _llm_response_looks_truncated('{"paragraph":"Desk tone"}', "length")
    assert _llm_response_looks_truncated('{"paragraph":"Desk tone...', "")
    assert _llm_response_looks_truncated('{"paragraph":"Desk tone"', "")
    assert not _llm_response_looks_truncated('{"paragraph":"Desk tone"}', "stop")


def main() -> None:
    test_extract_json_object_accepts_code_fence_and_python_dict()
    test_effective_max_workers_caps_minimax_parallelism()
    test_effective_max_workers_caps_minimax_fallback_route()
    test_deepseek_defaults_when_secret_present()
    test_deepseek_adds_minimax_fallback_candidate()
    test_cache_path_changes_when_prompt_changes()
    test_llm_response_truncation_guard()
    print("LLM enhancer resilience test passed")


if __name__ == "__main__":
    main()
