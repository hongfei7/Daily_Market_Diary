# Project Structure

This project keeps production code, operational scripts, tests, and generated research output in separate folders.

```text
Daily_Market_Diary/
|-- market_diary/
|   |-- main_professional.py          # Production entrypoint
|   |-- modules/                      # Data adapters and public-source fetchers
|   `-- professional/                 # Research analytics, report modes, rendering, charts, LLM orchestration
|-- reports_professional/
|   |-- README.md                    # Generated report gallery for GitHub browsing
|   `-- archive/
|       |-- README.md                # Generated archive index
|       `-- YYYY-MM-DD/
|           |-- morning_briefing.md   # GitHub-readable report
|           |-- charts/               # Charts and chart feature JSON for that report date
|           `-- raw/                  # Structured bundle/raw data for traceability
|-- scripts/                          # Operational helpers used locally and in GitHub Actions
|-- tests/                            # Script-based regression tests
|-- docs/                             # Project documentation
`-- .github/workflows/                # Scheduled GitHub Actions automation
```

## Output Rules

- Root-level `reports_professional/*.md`, `reports_professional/charts/*`, and `reports_professional/raw/*` are runtime output.
- GitHub-readable output must be copied into `reports_professional/archive/YYYY-MM-DD/`.
- The archive folder is allowed to contain the report, charts, chart feature JSON, and raw bundle for traceability.
- Test-generated charts and email previews should not be archived.
- `scripts/stage_report_archive.py` refreshes both report gallery indexes after each archive update.

## Report Modes

- `trading_daily`: used when the reviewed date is a Hong Kong trading day.
- `weekly_review`: used on Sunday morning when the reviewed date is Saturday.
- `non_trading_event_watch`: used on Monday morning when the reviewed date is Sunday.
- `holiday_event_watch`: used for Hong Kong market holidays that are not immediately followed by a reopen.
- `holiday_reopen_playbook`: used when the reviewed holiday is followed by a Hong Kong trading day on the briefing date.
