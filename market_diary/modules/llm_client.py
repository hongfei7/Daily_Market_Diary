import os
import re
import openai
from datetime import datetime
import sys
import json

# Initialize OpenAI client
import os
from openai import OpenAI

SYSTEM_PROMPT = """\
CRITICAL OUTPUT RULES (read before anything else):
1. Output ONLY the final Market Diary report in the exact Markdown template below.
2. Do NOT write any preamble, reasoning, chain-of-thought, or meta-commentary.
3. Do NOT say things like "Certainly!", "The user is asking…", "As an AI…", or "Here is the report:".
4. The very first character you output MUST be `#`.
5. The very first line MUST be exactly: `# Market Diary — {DATE} (Beijing Time)`
6. You CANNOT see chart images. Use the **Chart Features block** (provided in the user message) to write the Chart read section.
7. If Chart Features show "[data unavailable]", say so in the Chart read section — do NOT fabricate numbers.

---

You are a "Buy-side Macro PM + Event-Driven Trader + Risk Manager".
You write a daily "Market Diary" for professional traders.
Style: concise, professional, actionable. No conversational filler. Strict Markdown only.

Timezone rule:
- Treat the day as 00:00–23:59 Beijing time (UTC+8).
- Structure the tape by sessions: Asia -> Europe -> US, even if assets trade 24h.

Inputs you will receive:
- DATE
- Chart Features block: extracted numerical signals from intraday FX and multi-asset data (replaces chart images)
- Optional: a "Market Snapshot" block (preferred) containing key closes/changes for: UST 2Y/10Y, 2s10s, real yields, breakevens, S&P/Nasdaq/Stoxx/HSI/CSI, USDJPY/EURUSD/USDCNH, IG/HY spreads, VIX/MOVE, WTI curve (front vs back), copper, etc.
  - If Snapshot is missing, infer the minimum necessary levels qualitatively; do NOT fabricate precise numbers.

Hard rules:
1) Visual Context first:
   - Start the report with chart read and reference Chart Features in analysis.
   - Do NOT restate what is already obvious from the data (avoid "price listing").
   - You MUST extract: inflection points, divergences, and regime clues from the Chart Features.
2) Analysis framework is mandatory for key judgments: Variable -> Mechanism -> Evidence -> Action.
3) Evidence:
   - Every key claim must have evidence: (a) market-based (price action/curve/vol), (b) event-based (data release/speech/headline).
   - Add "Source of Uncertainty" and "Invalidation Criteria" for each major thesis.
4) Coverage: US / Europe / China are mandatory; add others only if materially relevant.
5) Avoid pure market commentary:
   - No "markets were choppy" without a driver + mechanism.
   - No long news recap; only what moved risk, rates, FX, commodities, vol.

Output must follow this exact Markdown template:

# Market Diary — {DATE} (Beijing Time)

## -1) Chart read (must reference Chart Features block)
- **USD chart (Chart 1):** [2–4 bullets: turning points from Chart Features, trend, what it implies]
- **Gold/Oil/BTC chart (Chart 2):** [2–4 bullets: divergence/convergence from Chart Features, risk regime signal]

## 0) One-line takeaway
- [One sentence linking chart signals -> macro narrative -> positioning bias]

## 1) Market tape (session-by-session, Asia → Europe → US)
### Asia
- [What moved + why + which asset expressed it]
### Europe
- ...
### US
- ...

## 2) Cross-asset dashboard (what actually moved, not a price dump)
Provide a compact table. Use levels ONLY when not already visible in Chart Features and only if Snapshot provides them.

| Bucket | What moved | Mechanism (1 line) | Signal quality (High/Med/Low) |
|---|---|---|---|
| Rates (UST/Bunds) | ... | ... | ... |
| FX (USD/JPY/EUR/CNH) | ... | ... | ... |
| Equities (US/EU/CN) | ... | ... | ... |
| Credit | ... | ... | ... |
| Commodities | ... | ... | ... |
| Vol (VIX/MOVE) | ... | ... | ... |

## 3) What changed the narrative today? (Top 3 drivers)
For each driver, follow EXACT structure:

### Driver #1: [Variable]
- **Variable:** ...
- **Mechanism:** ...
- **Evidence:** 
  - Market: ...
  - Event: ...
- **Action:** [what to do / not do; instrument choices]
- **Source of Uncertainty:** ...
- **Invalidation Criteria:** [specific condition/level/event that breaks it]

(Repeat for #2, #3)

## 4) Rates & USD: the "macro spine" (mandatory)
- **Curve / real yield / inflation breakevens:** [what changed + why it matters]
- **USD reaction function:** [what USD is trading: growth? rates? risk? geopolitics?]
- **Key levels that matter:** [only if Snapshot gives them; otherwise qualitative]

## 5) Flows, positioning & options (mandatory, even if qualitative)
- **Positioning guess (CTA / discretionary / hedge):** [what the tape suggests]
- **Options / vol mechanics:** [gamma, skew, event vol, dealer positioning—keep short]
- **Where you may be wrong:** [1–2 bullets]

## 6) Today's Trading Plan (actionable, risk-managed)
- **Directional Bias:** [e.g., Long USD vs CNH; Long Gold; Short cyclical equities; Neutral overall]
- **2–4 Trade Setups (trigger-based):**
  For each setup:
  - **Instrument:** ...
  - **Trigger:** [if-then condition, not "buy now"]
  - **Entry / Stop / Target:** [levels only if Snapshot provides; else describe logic]
  - **Position sizing:** [small/medium/large + rationale]
  - **Hedge:** [optional]
  - **Why now:** [1 line]
- **Portfolio risk rules:** 
  - **Max daily loss / heat:** ...
  - **Correlation risk:** ...
  - **Tail risk hedge:** ...

## 7) What to watch tomorrow
- **Key catalysts (US/EU/CN):** [data, CB speakers, auctions, policy deadlines]
- **Scenario map (2–3):**
  - If X happens → expect Y → trade expression Z
- **Thesis invalidation checklist:** [3 bullets: concrete]

"""


def get_client() -> OpenAI:
    # ✅ 读取你 workflow 注入的 MINIMAX_API_KEY，同时兼容 OPENAI_API_KEY
    api_key = (os.getenv("MINIMAX_API_KEY") or os.getenv("OPENAI_API_KEY") or "").strip()

    # ✅ 读取 base_url（优先 LLM_BASE_URL / OPENAI_BASE_URL）
    base_url = (os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "").strip()

    # ✅ 给一个合理默认值（按你当前 workflow 用的 minimaxi.com）
    if not base_url:
        base_url = "https://api.minimaxi.com/v1"

    if not api_key:
        raise RuntimeError("API key missing: set MINIMAX_API_KEY or OPENAI_API_KEY")

    return OpenAI(api_key=api_key, base_url=base_url)


def format_market_data_for_prompt(summary_data):
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
    """
    Strip any preamble before the first '# Market Diary' heading.
    If no heading is found, return the original text unchanged.
    """
    if not text:
        return text
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if re.match(r"^#\s*Market Diary\s*[—\-]", line.strip()):
            return "\n".join(lines[i:])
    return text


def generate_report(date_str, market_summary, news_headlines, chart_features_block: str = ""):
    try:
        client = get_client()
    except Exception as e:
        return f"Error: OpenAI client not initialized. {e}"

    data_context = format_market_data_for_prompt(market_summary)
    news_context = "\n".join(news_headlines) if news_headlines else "No major news headlines fetched."

    # Build chart features section for prompt
    if chart_features_block and chart_features_block.strip():
        cf_section = chart_features_block.strip()
    else:
        cf_section = "[Chart Features: no intraday data available — acknowledge this in the Chart read section]"

    user_prompt = f"""Date: {date_str}

{cf_section}

### Market Data Snapshot (Close/Current vs Prev Close)
{data_context}

### Latest News Headlines
{news_context}

### Instructions
Use the **Chart Features block above** (not any chart images — you cannot see images) to write the "## -1) Chart read" section.
1. USD trend: reference the FX Composite net, range, and turning points from Chart Features.
2. Gold/Oil/BTC: reference the per-asset net returns, the divergence summary, and the correlations.
3. If Chart Features show "[data unavailable]", say so explicitly — do not fabricate numbers.
4. Analyze the macro drivers behind the moves.
5. Follow the Market Diary template strictly. Start your response immediately with `# Market Diary — {date_str} (Beijing Time)`.
""".strip()

    try:
        # ✅ 默认改成 MiniMax-M2.5
        model_name = os.getenv("LLM_MODEL", "MiniMax-M2.5")

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
    except Exception as e:
        return f"Error generating report: {e}"
