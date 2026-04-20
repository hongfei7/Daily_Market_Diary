from __future__ import annotations

import importlib.util
import subprocess
import sys

from _bootstrap import ROOT  # noqa: F401


SCRIPT_PATH = ROOT / "scripts" / "audit_repo_hygiene.py"


def _load_audit_module():
    spec = importlib.util.spec_from_file_location("audit_repo_hygiene", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Unable to load audit_repo_hygiene.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_hygiene_classifier() -> None:
    audit = _load_audit_module()

    secret = audit.classify_path(".apikey")
    assert secret is not None
    assert secret.severity == "error"

    generated = audit.classify_path("reports/2026-04-13.md")
    assert generated is not None
    assert generated.severity == "warning"

    planned_cleanup = audit.classify_path("test_legacy.py", "D")
    assert planned_cleanup is not None
    assert planned_cleanup.severity == "info"


def test_no_tracked_secrets_or_virtualenvs() -> None:
    audit = _load_audit_module()
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=str(ROOT),
        check=True,
        capture_output=True,
        text=True,
    )
    tracked = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    violations = [
        finding.path
        for path in tracked
        for finding in [audit.classify_path(path)]
        if finding and finding.severity == "error"
    ]
    assert not violations, f"Tracked high-risk files: {violations}"


def main() -> None:
    test_hygiene_classifier()
    test_no_tracked_secrets_or_virtualenvs()
    print("Repository hygiene tests passed")


if __name__ == "__main__":
    main()
