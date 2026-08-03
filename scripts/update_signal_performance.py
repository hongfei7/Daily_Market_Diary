from __future__ import annotations

import argparse
from pathlib import Path

from market_diary.professional.performance import refresh_performance_tracking


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Rebuild the look-ahead-safe signal performance ledger from published archives.")
    parser.add_argument("--report-root", default=str(ROOT / "reports_professional"))
    parser.add_argument("--cost-bps", type=float, default=10.0)
    args = parser.parse_args()

    report_root = Path(args.report_root)
    performance = refresh_performance_tracking(
        {},
        output_dir=report_root,
        archive_root=report_root / "archive",
        chart_path=report_root / "performance" / "signal_performance.png",
        cost_bps=args.cost_bps,
    )
    quality = performance.get("data_quality", {}) or {}
    print(f"Performance status: {performance.get('status', 'unknown')}")
    print(f"Observations: {quality.get('observations', 0)}")
    print(f"Signals: {quality.get('signals', 0)}")
    print(f"Conflicts: {len(quality.get('conflicts', []) or [])}")
    print(f"Excluded weekend observations: {len(quality.get('excluded_non_session_observations', []) or [])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
