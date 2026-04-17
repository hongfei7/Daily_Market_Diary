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
|   |-- analytics.py
|   |-- config.py
|   |-- dashboard.py
|   |-- models.py
|   `-- report_builder.py
`-- modules/
    |-- data_fetcher.py
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

## Run

```bash
python market_diary/main_professional.py --date 2026-04-13 --no-llm
```

Useful flags:

- `--config PATH`: load a custom JSON config
- `--output-dir PATH`: override the output directory
- `--skip-charts`: skip chart generation
- `--skip-dashboard`: skip dashboard generation
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
python test_professional_workbench.py
```

This regression test validates the professional bundle, dashboard, and markdown renderer without depending on the full live run.
