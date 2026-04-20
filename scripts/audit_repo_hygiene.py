from __future__ import annotations

import argparse
import fnmatch
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[1]

SECRET_PATTERNS = (
    ".apikey",
    ".env",
    "*.env",
    "secrets.py",
)

VENV_PATTERNS = (
    "venv/*",
    ".venv/*",
    "env/*",
)

GENERATED_PATTERNS = (
    "reports/*",
    "reports_professional/*",
    "market_diary/reports_professional/*",
    "runtime_audit_*/*",
)

ROOT_TEST_PATTERN = "test_*.py"


@dataclass(frozen=True)
class Finding:
    path: str
    category: str
    severity: str
    status: str
    message: str


def _run_git(args: Iterable[str], cwd: Path = ROOT) -> List[str]:
    result = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def tracked_files(root: Path = ROOT) -> List[str]:
    return _run_git(["ls-files"], root)


def porcelain_status(root: Path = ROOT) -> Dict[str, str]:
    statuses: Dict[str, str] = {}
    result = subprocess.run(
        ["git", "status", "--short", "--porcelain"],
        cwd=str(root),
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if not line.strip() or len(line) < 4:
            continue
        if len(line) < 4:
            continue
        status = line[:2].strip()
        path = line[3:].strip()
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        statuses[path.replace("\\", "/")] = status
    return statuses


def _matches(path: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)


def classify_path(path: str, status: str = "") -> Finding | None:
    normalized = path.replace("\\", "/")
    if normalized.startswith("./"):
        normalized = normalized[2:]
    planned_deletion = "D" in status

    if _matches(normalized, SECRET_PATTERNS):
        return Finding(
            normalized,
            "secret",
            "error",
            status,
            "Secrets must never be tracked.",
        )

    if _matches(normalized, VENV_PATTERNS):
        return Finding(
            normalized,
            "environment",
            "error",
            status,
            "Virtual environments must stay outside version control.",
        )

    if _matches(normalized, GENERATED_PATTERNS):
        return Finding(
            normalized,
            "generated-output",
            "info" if planned_deletion else "warning",
            status,
            "Generated reports should be emailed/uploaded as artifacts, not kept as source files.",
        )

    if "/" not in normalized and fnmatch.fnmatch(normalized, ROOT_TEST_PATTERN):
        return Finding(
            normalized,
            "root-test",
            "info" if planned_deletion else "warning",
            status,
            "Tests should live under tests/ so the repository root stays readable.",
        )

    return None


def audit(root: Path = ROOT) -> List[Finding]:
    statuses = porcelain_status(root)
    findings: List[Finding] = []
    for path in tracked_files(root):
        finding = classify_path(path, statuses.get(path, ""))
        if finding:
            findings.append(finding)
    return findings


def print_report(findings: List[Finding]) -> None:
    counts = {"error": 0, "warning": 0, "info": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1

    print("Repository hygiene audit")
    print("=" * 32)
    print(f"errors:   {counts.get('error', 0)}")
    print(f"warnings: {counts.get('warning', 0)}")
    print(f"info:     {counts.get('info', 0)}")

    for severity in ("error", "warning", "info"):
        selected = [item for item in findings if item.severity == severity]
        if not selected:
            continue
        print(f"\n{severity.upper()}")
        for item in selected[:30]:
            status = f" [{item.status}]" if item.status else ""
            print(f"- {item.path}{status}: {item.message}")
        if len(selected) > 30:
            print(f"- ... {len(selected) - 30} more")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit tracked files for repository hygiene issues.")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    args = parser.parse_args(argv)

    findings = audit()
    print_report(findings)

    has_errors = any(item.severity == "error" for item in findings)
    has_warnings = any(item.severity == "warning" for item in findings)
    if has_errors or (args.strict and has_warnings):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
