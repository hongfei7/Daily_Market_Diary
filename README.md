# Daily Market Diary

Daily Market Diary is a professional morning research workbench for Hong Kong and offshore China market monitoring. It builds a structured pre-market briefing from market data, macro calendars, public flow indicators, watchlist news, chart features, and an optional LLM narrative layer.

The project is designed for a repeatable research-desk workflow: collect the overnight setup, score the local Hong Kong tape, generate charts, render a Markdown briefing, archive the result, and deliver it primarily through WeCom before the market opens.

> This repository is for research workflow automation. It is not investment advice.

## What It Produces

Each run writes a self-contained report package under `reports_professional/`.

- Markdown morning briefing
- Dashboard image
- Catalyst & Event Radar with explicit date confidence
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
5. Render Markdown, charts, archive pages, WeCom delivery assets, artifacts, and a secondary email copy.

The core report structure does not depend on an LLM. If the LLM layer is disabled or unavailable, the deterministic report still runs.

## LLM Provider Setup

`MiniMax-M3` (non-thinking mode for bounded JSON tasks) is the primary synthesis provider. DeepSeek `deepseek-v4-pro` is the independent fallback and continues to run the weekly financial-skill shadow pass. Claude is not called. MiniMax requests are serialized and separate reasoning from final content to protect the 07:30 delivery SLA.

For GitHub Actions, configure these repository secrets:

- `MINIMAX_API_KEY` for the primary MiniMax-M3 narrative provider
- `DEEPSEEK_API_KEY` for fallback synthesis and weekly skill shadow runs
- `WECOM_WEBHOOK_URL` for the required primary delivery channel
- SMTP secrets if the secondary email copy is enabled

The scheduled workflow maps them to:

```text
MiniMax:  MINIMAX_API_KEY, https://api.minimaxi.com/v1, MiniMax-M3
DeepSeek: DEEPSEEK_API_KEY, https://api.deepseek.com, deepseek-v4-pro
```

`LLM_REQUEST_TIMEOUT_SECONDS` optionally overrides the bounded per-request timeout (default 45 seconds); SDK retries are disabled because the application owns observable retries and MiniMax-to-DeepSeek failover.

For local development:

```bash
export DEEPSEEK_API_KEY="your_deepseek_key"
export LLM_BASE_URL="https://api.deepseek.com"
export LLM_MODEL="deepseek-v4-pro"
export LLM_PRIMARY_PROVIDER="deepseek"
```

To enable DeepSeek fallback and the provider-agnostic skill shadow run:

```bash
export DEEPSEEK_API_KEY="your_deepseek_key"
```

Use `--no-llm` when you want a deterministic run with no model calls.

### Financial skills and research plugins

Four project-local, provider-agnostic skills run in the weekly-review shadow pass when `DEEPSEEK_API_KEY` is available:

- `skills/morning-note`
- `skills/catalyst-calendar`
- `skills/thesis-tracker`
- `skills/report-evidence-qc`

Their output is contract-validated, stored under `skill_shadow` in the raw bundle, requires human review, and is never merged into the published report. Public Equity Investing and Data Analytics are Codex-side tools for manual follow-up research; they are not callable from GitHub Actions and are not CI dependencies. Claude models are not configured. See [the skill architecture](docs/skills_integration.md) for the external-skill review and promotion rules.

They are skipped on ordinary daily runs to protect the 07:30 delivery SLA and token budget. Set `DMD_SKILL_SHADOW_FORCE=1` for an explicit daily diagnostic, or `DMD_SKILL_SHADOW_ENABLED=0` to disable them entirely without disabling the production MiniMax narrative layer.

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
- `--skip-dashboard`: skip the dashboard and companion Catalyst & Event Radar
- `--skip-daily-chart`: skip Daily One Chart
- `--skip-trend-pack`: skip Hong Kong Trend Pack
- `--no-llm`: disable optional LLM sections
- `--debug`: save raw input payloads

## GitHub Actions

The scheduled workflow is defined in:

- [.github/workflows/morning_briefing_professional.yml](.github/workflows/morning_briefing_professional.yml)

It targets delivery before 07:30 Hong Kong / Beijing time. The primary run starts at 05:17 and a 06:47 recovery run executes only when the date archive is still absent. The workflow:

- checks out and syncs `main`
- installs pinned dependencies
- runs a narrow production preflight; the full suite is opt-in for manual dispatch and remains part of normal CI
- generates the professional briefing
- audits the generated output
- renders and audits the mobile WeCom summary, self-contained HTML attachment, and email preview before publication
- sends the decision brief and full report to WeCom first, with bounded retries
- treats WeCom as the required primary channel; a successful email copy does not hide a WeCom delivery failure
- uses the commute release policy so visible research caveats do not silently suppress the report, while broken files, invalid provenance, and delivery-asset failures still block publication
- sends a WeCom incident notice when no delivery-ready report can be produced, allowing the 06:47 recovery run to remain visible
- writes machine-readable success receipts for both the WeCom decision brief and HTML attachment
- archives published reports back to `main`
- verifies an immutable SHA-256 archive manifest and updates the signal ledger
- uploads reports and raw outputs as workflow artifacts

Manual workflow dispatch supports:

- `date`: explicit calendar review date
- `publish_archive`: commit the archive to `main`
- `deliver`: send the primary WeCom brief and attachment plus the secondary email copy for a manual run; defaults to `true`
- `run_full_tests`: run the full regression suite before a manual generation
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

scripts/                        # test, audit, archive, WeCom and email helpers
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

The tracked performance ledger uses the published risk regime, enters only at the first close on or after publication and after the market as-of date, applies turnover cost, excludes weekend pseudo-sessions, and excludes conflicting prices. Results remain exploratory until the minimum sample gate is met. See [Archive, Source Health, and Backtest](docs/data_archive_backtest.md).
