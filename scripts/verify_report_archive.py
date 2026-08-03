from __future__ import annotations

import argparse
from pathlib import Path

from stage_report_archive import verify_archive_integrity_index, write_archive_integrity_index


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build or verify the deterministic report-archive integrity index.")
    parser.add_argument("--archive-root", default=str(ROOT / "reports_professional" / "archive"))
    parser.add_argument("--rebuild", action="store_true", help="Rebuild the index before verification.")
    args = parser.parse_args()
    archive_root = Path(args.archive_root)
    if args.rebuild:
        path = write_archive_integrity_index(archive_root)
        print(f"Wrote {path}")
    result = verify_archive_integrity_index(archive_root)
    print(f"Archive integrity: {result.get('status', 'unknown')}")
    print(f"Dates: {result.get('dates', 0)}")
    print(f"Archive ID: {result.get('archive_id', '')}")
    for error in result.get("errors", []) or []:
        print(f"ERROR: {error}")
    return 0 if result.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
