from __future__ import annotations

from _bootstrap import ROOT  # noqa: F401

from main_professional import _run_external_step


def test_run_external_step_timeout_returns_error_type() -> None:
    import time

    payload = _run_external_step(
        "slow-step",
        lambda: time.sleep(1.2),
        fallback={"data": {}},
        timeout_seconds=0.05,
    )

    assert payload["status"] == "timeout"
    assert payload["error_type"] == "TimeoutError"
    assert payload["step"] == "slow-step"


def test_run_external_step_error_returns_error_type() -> None:
    payload = _run_external_step(
        "broken-step",
        lambda: (_ for _ in ()).throw(ValueError("bad payload")),
        fallback={"data": {}},
        timeout_seconds=1,
    )

    assert payload["status"] == "error"
    assert payload["error_type"] == "ValueError"
    assert payload["step"] == "broken-step"
