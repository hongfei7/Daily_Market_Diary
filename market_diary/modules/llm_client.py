"""LLM client helpers for optional narrative overlays."""

import os
import re
from pathlib import Path

from openai import OpenAI


DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"
MINIMAX_API_KEY_ENV = "MINIMAX_API_KEY"
MINIMAX_BASE_URL = "https://api.minimaxi.com/v1"
MINIMAX_MODEL = "MiniMax-M3"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

_API_KEY_ENV_PROVIDERS = (
    (MINIMAX_API_KEY_ENV, "minimax"),
    (OPENAI_API_KEY_ENV, "minimax"),
    (DEEPSEEK_API_KEY_ENV, "deepseek"),
)
_PROVIDER_API_KEY_ENVS = {
    "deepseek": (DEEPSEEK_API_KEY_ENV,),
    "minimax": (MINIMAX_API_KEY_ENV, OPENAI_API_KEY_ENV),
}


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


def _load_local_api_key_with_provider() -> tuple[str, str]:
    """Load a local development API key without printing or persisting it."""
    named_providers = {
        DEEPSEEK_API_KEY_ENV: "deepseek",
        MINIMAX_API_KEY_ENV: "minimax",
        OPENAI_API_KEY_ENV: "minimax",
        "API_KEY": "minimax",
    }
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
                provider = named_providers.get(key.strip())
                if provider and value.strip():
                    return value.strip().strip('"').strip("'"), provider
            return cleaned.strip('"').strip("'"), "minimax"
    return "", ""


def _load_local_api_key() -> str:
    return _load_local_api_key_with_provider()[0]


def _resolve_api_key(provider: str = "") -> tuple[str, str]:
    if provider:
        for env_name in _PROVIDER_API_KEY_ENVS.get(provider, ()):
            value = (os.getenv(env_name) or "").strip()
            if value:
                return value, provider
        local_key, local_provider = _load_local_api_key_with_provider()
        if local_provider == provider:
            return local_key, provider
        return "", provider

    for env_name, env_provider in _API_KEY_ENV_PROVIDERS:
        value = (os.getenv(env_name) or "").strip()
        if value:
            return value, env_provider
    return _load_local_api_key_with_provider()


def get_default_base_url(provider: str = "") -> str:
    """Return the provider base URL implied by env overrides or key priority."""
    selected_provider = provider or get_default_provider()
    explicit_base_url = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()
    if explicit_base_url and selected_provider == get_default_provider():
        return explicit_base_url

    if selected_provider == "deepseek":
        return DEEPSEEK_BASE_URL
    return MINIMAX_BASE_URL


def get_default_provider() -> str:
    """Return the configured primary provider, preferring MiniMax when available."""
    explicit = (os.getenv("LLM_PRIMARY_PROVIDER") or "").strip().lower()
    if explicit in _PROVIDER_API_KEY_ENVS and _resolve_api_key(explicit)[0]:
        return explicit
    if _resolve_api_key("minimax")[0]:
        return "minimax"
    if _resolve_api_key("deepseek")[0]:
        return "deepseek"
    return "minimax"


def get_available_providers() -> list[str]:
    """Return configured providers in priority order."""
    providers = []
    for provider in ("minimax", "deepseek"):
        api_key, _ = _resolve_api_key(provider)
        if api_key:
            providers.append(provider)
    return providers


def get_default_model(provider: str = "") -> str:
    """Return the model implied by configured provider priority."""
    selected_provider = provider or get_default_provider()
    if selected_provider == "deepseek":
        return DEEPSEEK_MODEL
    return MINIMAX_MODEL


def get_completion_extra_body(provider: str = "", model: str = "") -> dict:
    """Return provider-specific request options for clean completion parsing."""
    selected_provider = (provider or get_default_provider()).strip().lower()
    selected_model = (model or get_default_model(selected_provider)).strip().lower()
    if selected_provider == "minimax" and selected_model == "minimax-m3":
        return {"reasoning_split": True}
    return {}


def api_key_available() -> bool:
    """Return whether an environment or local development API key is present."""
    return bool((_resolve_api_key()[0] or "").strip())


def get_client(provider: str = "") -> OpenAI:
    """Build an OpenAI-compatible client from environment variables."""
    api_key, selected_provider = _resolve_api_key(provider)
    api_key = api_key.strip()
    base_url = get_default_base_url(selected_provider)

    if not api_key:
        raise RuntimeError("API key missing: set DEEPSEEK_API_KEY, MINIMAX_API_KEY, or OPENAI_API_KEY")

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

    primary_provider = get_default_provider()
    providers = get_available_providers() or [primary_provider]
    ordered_providers = [primary_provider] + [provider for provider in providers if provider != primary_provider]
    last_error = ""

    for provider in ordered_providers:
        try:
            client = get_client(provider)
            if provider == primary_provider:
                model_name = os.getenv("LLM_MODEL", get_default_model(provider))
            else:
                model_name = get_default_model(provider)
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.7,
                extra_body=get_completion_extra_body(provider, model_name),
            )
            raw = response.choices[0].message.content
            return _sanitize_output(raw)
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

    if not api_key_available():
        return (
            "Error: OpenAI client not initialized. "
            "API key missing: set DEEPSEEK_API_KEY, MINIMAX_API_KEY, or OPENAI_API_KEY"
        )
    return f"Error generating report: {last_error}"
