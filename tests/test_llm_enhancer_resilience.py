import os
import sys

from _bootstrap import ROOT  # noqa: F401

from professional.llm_enhancer import _effective_max_workers, _extract_json_object, _llm_response_looks_truncated


def test_extract_json_object_accepts_code_fence_and_python_dict() -> None:
    fenced = """```json
{"paragraph":"Desk tone","drivers":["Rates eased"],"hk_open_implication":"Constructive."}
```"""
    pythonish = "{'paragraph': 'Desk tone', 'drivers': ['Rates eased'], 'hk_open_implication': 'Constructive.'}"

    fenced_payload = _extract_json_object(fenced)
    pythonish_payload = _extract_json_object(pythonish)

    assert fenced_payload.get("paragraph") == "Desk tone"
    assert pythonish_payload.get("hk_open_implication") == "Constructive."


def test_effective_max_workers_caps_minimax_parallelism() -> None:
    prior_model = os.environ.get("LLM_MODEL")
    os.environ["LLM_MODEL"] = "MiniMax-M2.7"
    try:
        workers = _effective_max_workers({"max_workers": 4, "provider_parallelism": {"minimax": 1}})
    finally:
        if prior_model is None:
            os.environ.pop("LLM_MODEL", None)
        else:
            os.environ["LLM_MODEL"] = prior_model

    assert workers == 1


def test_llm_response_truncation_guard() -> None:
    assert _llm_response_looks_truncated('{"paragraph":"Desk tone"}', "length")
    assert _llm_response_looks_truncated('{"paragraph":"Desk tone...', "")
    assert _llm_response_looks_truncated('{"paragraph":"Desk tone"', "")
    assert not _llm_response_looks_truncated('{"paragraph":"Desk tone"}', "stop")


def main() -> None:
    test_extract_json_object_accepts_code_fence_and_python_dict()
    test_effective_max_workers_caps_minimax_parallelism()
    test_llm_response_truncation_guard()
    print("LLM enhancer resilience test passed")


if __name__ == "__main__":
    main()
