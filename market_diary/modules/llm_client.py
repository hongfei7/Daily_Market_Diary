"""LLM client helpers for optional narrative overlays."""

import os
import re
from pathlib import Path

from openai import OpenAI


SYSTEM_PROMPT = """\
CRITICAL OUTPUT RULES:
1. Output only the final Markdown report body.
2. Do not include preamble, reasoning, or meta-commentary.
3. The first character must be `#`.
4. The first line must be exactly `# Market Diary - {DATE} (Beijing Time)`.
5. Use the supplied Chart Features block instead of inventing chart observations.
6. If a metric is unavailable, say so plainly instead of fabricating values.

You are a buy-side macro PM, event-driven trader, and risk manager.
Write concise, professional, actionable English.
"""


def _load_local_api_key() -> str:
    """Load a local development API key without printing or persisting it."""
    candidates = [
        Path.cwd() / ".apikey",
        Path(__file__).resolve().parents[2] / ".apikey",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            raw = path.read_text(encoding="utf-8").strip()
        except Exception:
            continue
        if not raw:
            continue
        for line in raw.splitlines():
            cleaned = line.strip()
            if not cleaned or cleaned.startswith("#"):
                continue
            if "=" in cleaned:
                key, value = cleaned.split("=", 1)
                if key.strip() in {"MINIMAX_API_KEY", "OPENAI_API_KEY", "API_KEY"} and value.strip():
                    return value.strip().strip('"').strip("'")
            return cleaned.strip('"').strip("'")
    return ""


def api_key_available() -> bool:
    """Return whether an environment or local development API key is present."""
    return bool((os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY") or _load_local_api_key() or "").strip())


def get_client() -> OpenAI:
    """Build an OpenAI-compatible client from environment variables."""
    api_key = (os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY") or _load_local_api_key() or "").strip()
    base_url = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()

    if not base_url:
        base_url = "https://api.minimaxi.com/v1"
    if not api_key:
        raise RuntimeError("API key missing: set MINIMAX_API_KEY or OPENAI_API_KEY")

    return OpenAI(api_key=api_key, base_url=base_url)


def format_market_data_for_prompt(summary_data):
    """Convert the summary market snapshot into a compact prompt block."""
    formatted_sections = []
    for category, items in (summary_data or {}).items():
        if not items:
            continue

        lines = [f"**{category}**"]
        for name, data in items.items():
            if isinstance(data, dict):
                price = data.get("Price", "N/A")
                change = data.get("Pct Change", "N/A")
                lines.append(f"- {name}: {price} ({change})")
            else:
                lines.append(f"- {name}: {data}")

        formatted_sections.append("\n".join(lines))

    return "\n\n".join(formatted_sections)


def _sanitize_output(text: str) -> str:
    """Strip any preamble before the first Markdown heading."""
    if not text:
        return text

    lines = text.splitlines()
    for index, line in enumerate(lines):
        if re.match(r"^#\s*Market Diary\s*[-:]", line.strip()):
            return "\n".join(lines[index:])
    return text


def generate_report(date_str, market_summary, news_headlines, chart_features_block: str = ""):
    """Generate a legacy market-diary style report through an OpenAI-compatible endpoint."""
    try:
        client = get_client()
    except Exception as exc:
        return f"Error: OpenAI client not initialized. {exc}"

    data_context = format_market_data_for_prompt(market_summary)
    news_context = "\n".join(news_headlines) if news_headlines else "No major news headlines fetched."
    chart_context = (
        chart_features_block.strip()
        if chart_features_block and chart_features_block.strip()
        else "[Chart Features: no intraday data available]"
    )

    user_prompt = f"""Date: {date_str}

{chart_context}

### Market Data Snapshot
{data_context}

### Latest News Headlines
{news_context}

### Required Structure
# Market Diary - {date_str} (Beijing Time)

## Chart Read
- USD chart: describe the turning points and regime signal.
- Gold/Oil/BTC: describe divergence or convergence and what it implies.

## One-line Takeaway
- One sentence linking the macro narrative to positioning bias.

## Market Tape
### Asia
### Europe
### US

## Cross-Asset Dashboard
- Rates
- FX
- Equities
- Credit
- Commodities
- Volatility

## Top Drivers
- Variable
- Mechanism
- Evidence
- Action
- Uncertainty
- Invalidation

## Rates and USD
## Flows and Positioning
## Trading Plan
## What to Watch Tomorrow
""".strip()

    try:
        model_name = os.getenv("LLM_MODEL", "MiniMax-M2.7")
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.7,
        )
        raw = response.choices[0].message.content
        return _sanitize_output(raw)
    except Exception as exc:
        return f"Error generating report: {exc}"
