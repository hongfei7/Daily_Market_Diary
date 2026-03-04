# 📅 Daily Market Diary

> **LLM-powered daily macro report** — automatically generates a professional buy-side market diary every morning, complete with intraday charts and AI analysis.

[![Daily Market Diary](https://github.com/hongfei7/Daily_Market_Diary/actions/workflows/market_diary.yml/badge.svg)](https://github.com/hongfei7/Daily_Market_Diary/actions/workflows/market_diary.yml)

---

## What it does

Every day at **04:00 Beijing time** (via GitHub Actions), this project:

1. **Fetches market data** — intraday timeseries + daily closes for FX, rates, equities, commodities, vol, credit
2. **Fetches news headlines** — latest macro/market news
3. **Generates charts** — 8 intraday charts saved as PNG:
   - USD Strength (FX composite)
   - Gold vs Oil vs Bitcoin
   - UST 2Y / 10Y / 30Y rates (bps from open)
   - Curve: 2s10s slope
   - Real yield vs Breakeven
   - Global equities (US / EU / CN)
   - VIX vs MOVE vol
   - Credit: IG vs HY
4. **Extracts Chart Features** — converts intraday signals into LLM-readable text (turning points, net moves, divergence, correlations) and saves a JSON audit file
5. **Runs LLM analysis** (MiniMax M2.5) — generates a structured **Market Diary** in Markdown following a strict buy-side template
6. **Commits the report** — pushes `reports/YYYY-MM-DD.md` back to the repo automatically

---

## Report structure

Each report follows this template:

| Section | Content |
|---|---|
| `-1) Chart read` | USD trend + Gold/Oil/BTC analysis from Chart Features |
| `0) One-line takeaway` | Single-sentence macro narrative |
| `1) Market tape` | Session-by-session (Asia → Europe → US) |
| `2) Cross-asset dashboard` | Compact table: what moved + mechanism + signal quality |
| `3) Top 3 drivers` | Variable → Mechanism → Evidence → Action framework |
| `4) Rates & USD spine` | Curve, real yields, breakevens, USD reaction function |
| `5) Flows & positioning` | CTA/discretionary guess, vol mechanics |
| `6) Trading plan` | 2–4 trigger-based setups with entry/stop/target |
| `7) Watch tomorrow` | Catalysts, scenario map, invalidation checklist |

---

## Project structure

```
Daily_Market_Diary/
├── .github/workflows/
│   └── market_diary.yml       # GitHub Actions — runs daily at 04:00 BJT
├── market_diary/
│   ├── main.py                # Entry point: charts + features + LLM + save report
│   ├── requirements.txt
│   └── modules/
│       ├── data_fetcher.py    # Market data + news fetching
│       ├── chart_features.py  # Extract LLM-readable signals from timeseries
│       └── llm_client.py      # MiniMax/OpenAI API client + prompt
└── reports/                   # Generated reports (auto-committed by CI)
    ├── YYYY-MM-DD.md
    └── charts/
        ├── fx_YYYY-MM-DD.png
        ├── features_YYYY-MM-DD.json
        └── ...
```

---

## Setup

### 1. Clone & install dependencies

```bash
git clone https://github.com/hongfei7/Daily_Market_Diary.git
cd Daily_Market_Diary
pip install -r market_diary/requirements.txt
```

### 2. Set environment variables

```bash
# Required: MiniMax API key (or any OpenAI-compatible API)
export MINIMAX_API_KEY=your_key_here

# Optional overrides
export LLM_BASE_URL=https://api.minimaxi.com/v1   # default
export LLM_MODEL=MiniMax-M2.5                      # default
```

### 3. Run manually

```bash
python market_diary/main.py --date 2026-03-03
```

Output: `reports/2026-03-03.md` + `reports/charts/`

---

## GitHub Actions (automated)

The workflow runs daily and commits reports automatically.

**Required secrets** (set in `Settings → Secrets → Actions`):

| Secret | Description |
|---|---|
| `MINIMAX_API_KEY` | Your MiniMax API key |

**To trigger manually**: go to `Actions` tab → `Daily Market Diary` → `Run workflow`.

---

## Dependencies

| Package | Purpose |
|---|---|
| `pandas` / `numpy` | Timeseries processing |
| `matplotlib` | Chart generation |
| `openai` | LLM API client (OpenAI-compatible) |
| `yfinance` | Market data |
| `feedparser` / `requests` | News headlines |

---

## License

MIT
