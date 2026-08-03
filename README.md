# Daily Market Diary

Daily Market Diary is a professional morning research workbench for Hong Kong and offshore China market monitoring. It builds a structured pre-market briefing from market data, macro calendars, public flow indicators, watchlist news, chart features, and an optional LLM narrative layer.

The project is designed for a repeatable research-desk workflow: collect the overnight setup, score the local Hong Kong tape, generate charts, render a Markdown briefing, archive the result, and optionally email it before the market opens.

> This repository is for research workflow automation. It is not investment advice.

## What It Produces

Each run writes a self-contained report package under `reports_professional/`.

- Markdown morning briefing
- Dashboard image
- Daily One Chart
- Hong Kong trend pack when enabled
- Chart feature JSON
- Structured raw bundle for audit and debugging
- Source-health snapshot with freshness and authority checks
- Append-only published-signal ledger and look-ahead-safe performance diagnostic
- GitHub-readable archive pages

The stable reader entry is:

- [Latest professional report](reports_professional/latest/README.md)
- [Report archive gallery](reports_professional/README.md)
- [Signal performance ledger](reports_professional/performance/README.md)

## Core Workflow

The professional pipeline follows five stages:

1. Resolve the report date and market calendar context.
2. Collect market, macro, flow, watchlist, and news inputs.
3. Build deterministic analytics and quality checks.
4. Add optional LLM sections for narrative framing.
5. Render Markdown, charts, archive pages, artifacts, and email output.

The core report structure does not depend on an LLM. If the LLM layer is disabled or unavailable, the deterministic report still runs.

## LLM Provider Setup

The primary LLM provider is MiniMax. DeepSeek is the fallback provider and runs the three financial skills in non-publishing shadow mode when its key is available. Claude is not called.

For GitHub Actions, configure these repository secrets:

- `MINIMAX_API_KEY` for the primary provider
- `DEEPSEEK_API_KEY` for fallback and skill shadow runs
- SMTP secrets if email delivery is enabled

The scheduled workflow maps them to:

```text
MiniMax:  MINIMAX_API_KEY, https://api.minimaxi.com/v1, MiniMax-M3
DeepSeek: DEEPSEEK_API_KEY, https://api.deepseek.com, deepseek-v4-pro
```

For local development:

```bash
export MINIMAX_API_KEY="your_minimax_key"
export LLM_BASE_URL="https://api.minimaxi.com/v1"
export LLM_MODEL="MiniMax-M3"
export LLM_PRIMARY_PROVIDER="minimax"
```

To enable DeepSeek fallback and the provider-agnostic skill shadow run:

```bash
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Use `--no-llm` when you want a deterministic run with no model calls.

### Financial skills and research plugins

Three project-local, provider-agnostic skills run in shadow mode when `DEEPSEEK_API_KEY` is available:

- `skills/morning-note`
- `skills/catalyst-calendar`
- `skills/thesis-tracker`

Their output is stored under `skill_shadow` in the raw bundle, requires human review, and is never merged into the published report. Public Equity Investing and Data Analytics are Codex-side tools for manual follow-up research; they are not callable from GitHub Actions and are not CI dependencies. Claude models are not configured.

Set `DMD_SKILL_SHADOW_ENABLED=0` to disable the three extra DeepSeek calls without disabling the production MiniMax narrative layer.

The report content hierarchy and visual rules are documented in `docs/professional_report_design_system.md`.

## Installation

Python 3.10 or newer is required. GitHub Actions currently runs Python 3.12.

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

## Run Locally

Generate a professional briefing:

```bash
market-diary-professional --briefing-date 2026-05-23
```

Run for explicit review, global, and Hong Kong dates:

```bash
market-diary-professional \
  --review-date 2026-05-23 \
  --global-date 2026-05-23 \
  --hk-date 2026-05-23 \
  --briefing-date 2026-05-23
```

Run without charts or LLM calls:

```bash
market-diary-professional --briefing-date 2026-05-23 --skip-charts --no-llm
```

Useful flags:

- `--config PATH`: load a custom JSON config
- `--output-dir PATH`: override the output directory
- `--skip-charts`: skip all chart generation
- `--skip-dashboard`: skip only the dashboard
- `--skip-daily-chart`: skip Daily One Chart
- `--skip-trend-pack`: skip Hong Kong Trend Pack
- `--no-llm`: disable optional LLM sections
- `--debug`: save raw input payloads

## GitHub Actions

The scheduled workflow is defined in:

- [.github/workflows/morning_briefing_professional.yml](.github/workflows/morning_briefing_professional.yml)

It runs daily on a UTC schedule aligned to Hong Kong / Beijing morning time. The workflow:

- checks out and syncs `main`
- installs pinned dependencies
- runs the regression suite
- generates the professional briefing
- audits the generated output
- sends email when SMTP secrets are present
- archives published reports back to `main`
- verifies an immutable SHA-256 archive manifest and updates the signal ledger
- uploads reports and raw outputs as workflow artifacts

Manual workflow dispatch supports:

- `date`: explicit calendar review date
- `publish_archive`: commit the archive to `main`
- `include_raw_bundle`: include raw JSON in the committed archive

## Tests

Run the script-based regression suite:

```bash
python scripts/run_tests.py
```

Run the same suite plus pytest collection:

```bash
python scripts/run_tests.py --pytest
```

Run repository hygiene checks:

```bash
python scripts/audit_repo_hygiene.py
```

## Project Layout

```text
market_diary/
|-- main_professional.py        # professional pipeline entrypoint
|-- modules/                    # data adapters and shared data helpers
`-- professional/               # analytics, report rendering, charts, LLM overlay

reports_professional/
|-- latest/                     # stable latest report entry
|-- archive/                    # immutable dated report packages and manifests
`-- performance/                # tracked signal ledger, methodology, summary, and chart

scripts/                        # test, audit, archive, email helpers
tests/                          # regression suite
docs/                           # supporting documentation
```

More detailed documentation:

- [Quick Start](docs/QUICK_START.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)
- [Project Structure](docs/PROJECT_STRUCTURE.md)
- [Upgrade Guide](docs/UPGRADE_GUIDE.md)
- [Professional README](README_PROFESSIONAL.md)

## Configuration

The workbench ships with built-in defaults in `market_diary/professional/config.py`. You can add a local `config.json` to override report settings, watchlists, macro mappings, and LLM routes.

Important config areas:

- `watchlists.core_coverage`
- `watchlists.focus_pool`
- `watchlists.learning_pool`
- `macro_indicator_map`
- `source_health`
- `performance`
- `llm.routes`
- `llm.tasks`

Local secrets such as `.env` and `.apikey` are intentionally ignored and must not be committed.

## Output Policy

Published report pages and selected chart assets are committed under `reports_professional/archive/`. New date packages carry a deterministic SHA-256 manifest and cannot be silently overwritten by a rerun. A full-history integrity index covers legacy dates without rewriting their contents. Compact source-health and performance snapshots are archived even when the full raw bundle is not committed.

Raw bundles are committed only when `include_raw_bundle` is explicitly enabled for an archive publish run.

The tracked performance ledger uses the published risk regime, enters only at the next available close, applies turnover cost, excludes weekend pseudo-sessions, and records late-price conflicts. It is a research diagnostic rather than an investable track record. See [Archive, Source Health, and Backtest](docs/data_archive_backtest.md).
