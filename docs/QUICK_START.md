# Quick Start

## Local Setup

```bash
pip install -r market_diary/requirements.txt
pip install -e . --no-deps
```

Optional LLM environment variables:

```bash
export DEEPSEEK_API_KEY="your_deepseek_api_key"
export LLM_BASE_URL="http://api.deepseek.com"
export LLM_MODEL="deepseek-v4-pro"
```

If DeepSeek is unavailable, the pipeline falls back to the existing MiniMax setup:

```bash
export MINIMAX_API_KEY="your_minimax_api_key"
export LLM_BASE_URL="https://api.minimaxi.com/v1"
export LLM_MODEL="MiniMax-M2.7"
```

The professional pipeline now uses a multi-call LLM design:

- `news_selection`
- `overnight_review`
- `hk_review`
- `macro_interpretation`
- `company_commentary`
- `theme_deep_dive`
- `final_framing`

By default, task outputs are cached under `reports_professional/raw/llm_cache/` so repeated reruns on the same inputs do not keep spending API budget.

## Core Commands

Run smoke tests:

```bash
python tests/test_github_actions.py
python tests/test_professional_workbench.py
python tests/test_market_data_resilience.py
```

Generate a briefing:

```bash
python market_diary/main_professional.py --date 2026-04-13
```

Skip the LLM layer:

```bash
python market_diary/main_professional.py --date 2026-04-13 --no-llm
```

## Report Structure

The current professional report is built around three commute-reading layers:

1. `Layer 1 | Scan`
   One-line market pulse, global dashboard, Hong Kong quick checks, and a short morning checklist.
2. `Layer 2 | Deep Read`
   Overseas recap, Hong Kong / A-share review, macro & policy tracking, company events, and watchlists.
3. `Layer 3 | Thinking`
   Rotating theme deep dive, forward calendar, daily one chart, and a personal view pad.

## GitHub Actions Email Delivery

The workflow can email the generated report to `hongfei_wu7@outlook.com` after each run.

Configure these GitHub repository secrets:

```text
SMTP_HOST
SMTP_PORT
SMTP_USERNAME
SMTP_PASSWORD
SMTP_FROM
SMTP_USE_TLS
```

Notes:

- `SMTP_FROM` can usually be the same as `SMTP_USERNAME`.
- If the SMTP secrets are not configured, the workflow will skip email delivery and still generate the report.
- The email contains a mobile-friendly HTML summary and attaches the full markdown report.

Preview the email locally without sending:

```bash
python scripts/send_report_email.py --report-date 2026-04-13 --output-dir reports_professional --to hongfei_wu7@outlook.com --dry-run
```

## Recommended Next Data Upgrades

- HKEX main-board turnover and short-selling ratio
- Southbound / Northbound Stock Connect flows
- AH premium index
- HIBOR and HKMA liquidity operations
- China 10Y government bond yield
