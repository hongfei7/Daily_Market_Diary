# Professional Morning Research Workbench

This is the upgraded morning-briefing pipeline for the project. It is designed for a Hong Kong sell-side research-desk workflow rather than a student-style single-script dump.

## What Changed

- The professional path is now English-only.
- The report is layered into a 3-minute edition, a 15-minute edition, and a traceable appendix.
- The pipeline is split into data collection, deterministic analytics, dashboard generation, and report rendering.
- The report no longer depends on the LLM for the core structure. LLM output is optional and additive.
- A dashboard image is generated automatically to make the briefing easier to scan before market open.
- Personal watchlists are first-class inputs instead of ad-hoc notes.
- The default lens now reflects a Hong Kong offshore-China desk: Hang Seng, HSCEI, Hang Seng TECH, FXI, USD/CNH, and USD/HKD all feed the opening read.

## Architecture

```text
market_diary/
|-- main_professional.py
|-- professional/
|   |-- analytics.py                  # bundle orchestrator
|   |-- analytics_briefing.py         # catalysts, source links, must-watch list
|   |-- analytics_flows.py            # movers, ETF/short-sell/options, flow tracker
|   |-- analytics_hk_checks.py        # Hong Kong quick-check table
|   |-- analytics_macro.py            # macro calendar scoring
|   |-- analytics_market.py           # market snapshot and overview helpers
|   |-- analytics_narrative.py        # theme, today-forward, non-trading, weekly review
|   |-- analytics_public_flow.py      # public Stock Connect / A-H fallback enrichment
|   |-- analytics_sector.py           # sector and company news digest
|   |-- analytics_trackers.py         # high-frequency trackers
|   |-- analytics_watchlist.py        # watchlist price/news snapshots
|   |-- attribution.py
|   |-- config.py
|   |-- daily_one_chart.py
|   |-- dashboard.py
|   |-- date_policy.py
|   |-- fact_checker.py
|   |-- llm_enhancer.py
|   |-- models.py
|   |-- report_blocks.py
|   |-- report_builder.py
|   |-- report_formatting.py
|   |-- report_layout.py
|   `-- report_sections.py
`-- modules/
    |-- adapter_ah_premium.py
    |-- adapter_hkex_announce.py
    |-- adapter_shortsell.py
    |-- adapter_stockconnect.py
    |-- china_rates.py
    |-- data_fetcher.py
    |-- hk_local_data.py
    |-- macro_calendar.py
    |-- market_movers.py
    |-- risk_radar.py
    `-- sector_news.py
```

## Output Structure

Each run writes to `reports_professional/`:

- `YYYY-MM-DD_morning_briefing.md`: final markdown briefing
- `charts/dashboard_YYYY-MM-DD.png`: visual dashboard
- `charts/*.png`: supporting charts
- `charts/features_YYYY-MM-DD.json`: extracted chart features
- `raw/YYYY-MM-DD_bundle.json`: structured research bundle

Final markdown reports and their production chart assets can be archived in `reports_professional/` so they can be read directly on GitHub. Raw bundles, email previews, runtime caches, and test-generated chart probes remain ignored unless raw bundle archiving is explicitly requested.
The GitHub-facing archive exposes a stable `reports_professional/latest/README.md` entry for the newest report, and each `archive/YYYY-MM-DD/` folder also carries a `README.md` so the report renders as soon as you open that folder on GitHub.
The directory is now split by purpose: `latest/` is the stable reader entry, `archive/` holds dated published reports, and root-level `charts/` plus `raw/` remain runtime workspace material rather than the primary GitHub browsing path.

## Run

```bash
python market_diary/main_professional.py --review-date 2026-04-13 --no-llm
```

After an editable install, the same entrypoint is available as:

```bash
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
market-diary-professional --review-date 2026-04-13 --no-llm
```

Useful flags:

- `--config PATH`: load a custom JSON config
- `--output-dir PATH`: override the output directory
- `--skip-charts`: skip dashboard, Daily One Chart, Hong Kong Trend Pack, and chart appendix generation
- `--skip-dashboard`: skip dashboard generation
- `--skip-daily-chart`: skip Daily One Chart generation
- `--skip-trend-pack`: skip Hong Kong Trend Pack generation
- `--no-llm`: disable the optional LLM overlay
- `--debug`: save raw input payloads

## Config

The upgraded workbench uses built-in English defaults. It only auto-loads `config.json` if you create one yourself.

Key config areas:

- `macro_indicator_map`: impact mapping for macro releases
- `watchlists.core_coverage`
- `watchlists.focus_pool`
- `watchlists.learning_pool`

See [docs/README.md](docs/README.md) for the supporting documentation index.

## Test

```bash
python scripts/run_tests.py
python scripts/audit_repo_hygiene.py
```

The shared regression runner mirrors the GitHub Actions smoke suite. Add
`--pytest` to also run pytest collection locally.

The analytics layer is intentionally split into narrow modules so changes to
macro scoring, Hong Kong checks, public-flow enrichment, narrative framing, or
watchlist fetching can be tested without touching the bundle orchestrator.

## Publishing

Scheduled GitHub Actions runs generate, audit, email, and upload the briefing
as an artifact. They do not commit generated files back to `main` by default.

Manual workflow runs can publish the GitHub-readable archive by enabling
`publish_archive`. Enable `include_raw_bundle` only when the raw JSON should be
kept as repository audit evidence; otherwise it stays in the workflow artifact.
Publishing the archive also refreshes the stable `latest/` landing page and the
per-date `README.md` entry pages.
