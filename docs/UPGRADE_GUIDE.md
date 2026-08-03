# Upgrade Guide

## What Changed

The project is no longer a long, student-style market dump. It now behaves like a professional morning workbench designed for a Hong Kong sell-side research desk.

### Before

- One dense output stream.
- Limited market context.
- Weak separation between data collection, analytics, visuals, and writing.
- Mainland-style bias without enough offshore China or Hong Kong emphasis.

### Now

- English-only briefing output.
- A layered report format: `3-minute edition`, `15-minute edition`, and traceable appendix.
- Hong Kong-specific market lens built into the summary and dashboard.
- Modular architecture that separates adapters, analytics, visualization, and report assembly.

## Current Architecture

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
    |-- chart_features.py
    |-- data_fetcher.py
    |-- llm_client.py
    |-- macro_calendar.py
    |-- market_movers.py
    |-- risk_radar.py
    `-- sector_news.py
```

### Responsibility Split

- `main_professional.py`: orchestration and output writing.
- `professional/analytics.py`: narrative-ready structured bundle generation.
- `professional/dashboard.py`: visual summary image generation.
- `professional/report_builder.py`: deterministic English report assembly.
- `modules/*.py`: lightweight data adapters and fallback feed collectors.

## Hong Kong Research Desk Focus

The professional version now gives more weight to:

- Hang Seng Index, HSCEI, and Hang Seng Tech.
- USD/HKD and USD/CNH as funding and China-risk lenses.
- Offshore China leadership versus US and Europe.
- Coverage pools that default to Hong Kong-listed internet, exchange, insurer, and China proxy names.

## Migration Steps

### 1. Install dependencies

```bash
pip install -r market_diary/requirements.txt
```

### 2. Configure environment variables

The system can run without the LLM layer, but the AI overlay requires a model key.

```bash
cp .env.example .env
```

Then populate:

- `MINIMAX_API_KEY` as primary, with `DEEPSEEK_API_KEY` as fallback and skill-shadow provider
- optional `LLM_BASE_URL`
- optional `LLM_MODEL`

### 3. Run the professional launcher

Windows:

```bash
run_morning_briefing.bat
```

Linux or macOS:

```bash
./run_morning_briefing.sh
```

Direct Python entry:

```bash
python market_diary/main_professional.py --date 2026-04-13
```

### 4. Validate the workbench

```bash
python tests/test_github_actions.py
python tests/test_professional_workbench.py
```

## Customization Points

### Coverage pools and defaults

Edit:

- `market_diary/professional/config.py`

This is where you tune:

- core coverage names
- focus pool names
- learning pool names
- headline caps and section weights

### Hong Kong desk analytics logic

Edit:

- `market_diary/professional/analytics.py`

This is where you can refine:

- market priority logic
- mover attribution rules
- Hong Kong lens scoring
- high-frequency tracker selection

### Visual presentation

Edit:

- `market_diary/professional/dashboard.py`
- `market_diary/professional/report_builder.py`

This is where you can tune:

- the dashboard layout
- chart labels
- section ordering
- wording and briefing tone

### Data adapters

Edit:

- `market_diary/modules/data_fetcher.py`
- `market_diary/modules/macro_calendar.py`
- `market_diary/modules/sector_news.py`
- `market_diary/modules/market_movers.py`
- `market_diary/modules/risk_radar.py`

These modules are the right place to connect Bloomberg, Wind, FactSet, or internal feeds later.

## Operational Notes

### GitHub Actions

The live workflow is:

- `.github/workflows/morning_briefing_professional.yml`

It now tests both the CI smoke path and the professional workbench before committing fresh reports.

### Legacy student path

The obsolete student workflow and the unused legacy template have been removed from the active path so the repository has a single professional operating model.

## Recommended Next Upgrades

The current system is now good enough for daily production use, but the highest-value future upgrades would be:

1. Replace placeholder macro and event adapters with Bloomberg or Wind feeds.
2. Add structured announcement ingestion for your actual coverage list.
3. Add sector-specific high-frequency datasets by team.
4. Introduce internal consensus changes and broker-view aggregation.
