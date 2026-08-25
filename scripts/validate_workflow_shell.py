from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
from typing import Iterable, List, Tuple


RUN_BLOCK = re.compile(r"^(?P<indent>\s*)run:\s*\|\s*$")
GITHUB_EXPRESSION = re.compile(r"\$\{\{.*?\}\}")


def extract_run_blocks(path: Path) -> Iterable[Tuple[int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    index = 0
    while index < len(lines):
        match = RUN_BLOCK.match(lines[index])
        if not match:
            index += 1
            continue
        run_indent = len(match.group("indent"))
        block_start = index + 1
        block_lines: List[str] = []
        index += 1
        while index < len(lines):
            line = lines[index]
            if line.strip() and len(line) - len(line.lstrip()) <= run_indent:
                break
            block_lines.append(line)
            index += 1
        nonblank = [line for line in block_lines if line.strip()]
        if not nonblank:
            continue
        content_indent = min(len(line) - len(line.lstrip()) for line in nonblank)
        script = "\n".join(line[content_indent:] if line.strip() else "" for line in block_lines) + "\n"
        yield block_start + 1, GITHUB_EXPRESSION.sub("gha_expr", script)


def validate_workflow_shell(path: Path) -> List[str]:
    errors: List[str] = []
    for line_number, script in extract_run_blocks(path):
        result = subprocess.run(
            ["bash", "-n"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            detail = (result.stderr or result.stdout).strip()
            errors.append(f"{path}:{line_number}: {detail}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Run bash syntax validation on GitHub Actions run blocks.")
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    failures: List[str] = []
    for path in args.paths:
        failures.extend(validate_workflow_shell(path))
    if failures:
        print("\n".join(failures))
        return 1
    print(f"Validated shell blocks in {len(args.paths)} workflow(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
