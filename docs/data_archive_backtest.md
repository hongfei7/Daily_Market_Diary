# Archive, Source Health, and Backtest

## Archive contract

Each newly published date is a self-contained, immutable package under `reports_professional/archive/YYYY-MM-DD/`.

- `morning_briefing.md`: the published research view.
- `charts/`: only visuals referenced by the published report.
- `audit/source_health.json`: compact source reliability snapshot.
- `audit/performance_summary.json`: the backtest state visible when the report was published.
- `raw/`: optional full bundle, controlled by `include_raw_bundle`.
- `manifest.json`: deterministic SHA-256 hashes for every payload file except generated navigation README files.
- `archive/integrity_manifest.json`: one deterministic index covering every legacy and current dated payload without rewriting old report pages.

If a date already exists and a rerun produces different payload hashes, publishing fails. Corrections must use a new report date or an explicitly designed revision workflow; history is not silently rewritten.

Verify all 86 existing date packages locally with:

```bash
python scripts/verify_report_archive.py
```

GitHub Actions artifacts retain runtime output for 30 days. Git-tracked date packages and the performance ledger remain in repository history.

## Source-health contract

Every adapter keeps its original provenance records. The source-health layer scores five separate dimensions:

1. Availability: whether the source actually returned usable data.
2. Completeness: the share of records that are active rather than unavailable.
3. Authority: official/licensed sources rank above public, derived, or cached evidence.
4. Confidence: the adapter's declared evidence confidence.
5. Freshness: the share of all active records inside a source-specific age policy.

`market_data` and `hk_local` are critical by default. Market quotes are represented asset by asset, so one fresh quote cannot hide stale instruments. A critical source with no decision-grade record, a future-dated record, or freshness coverage below its minimum ratio forces the strict research policy to manual review. The unattended commute policy keeps that restriction visible in the report and WeCom summary instead of silently dropping the entire morning delivery; invalid provenance or structurally broken artifacts still block publication. Optional sources degrade the report with a visible caveat.

## Backtest contract

The ledger evaluates only views that were actually published.

- Signal: deterministic report regime mapped to long (`+1`), neutral (`0`), or short (`-1`).
- Blocked report: recorded with a zero position so a failed-quality run cannot become a trade after the fact.
- Execution: first available benchmark close on or after the publication date and strictly after the report's market `as_of` date.
- Portfolio return: close-to-close, latest published view per market date, with turnover cost.
- Event windows: 1, 5, and 20 sessions after the entry close.
- Benchmarks: Hang Seng Index and Hang Seng TECH ETF (3033.HK), explicitly identified as a listed proxy rather than the index.
- Guardrails: weekend pseudo-sessions are excluded; current-day returns are never assigned before publication; contradictory prices are excluded rather than selected.
- Sample gate: results are labeled `exploratory` until there are at least 252 sessions and 100 active-signal sessions for a benchmark.

The diagnostic excludes dividends, financing, borrow, taxes, market impact, intraday execution, and benchmark reconstitution. It is designed to answer whether the report's directional regime adds information, not to claim an executable fund track record.

Rebuild it locally with:

```bash
python scripts/update_signal_performance.py
```

The durable outputs are:

- `reports_professional/performance/signal_ledger.json`
- `reports_professional/performance/performance_summary.json`
- `reports_professional/performance/signal_performance.png`
- `reports_professional/performance/README.md`
