# Professional Report Design System

## Objective

The report is a decision brief, not a market-data catalogue. It should let a Hong Kong research professional answer four questions quickly:

1. What changed?
2. Why does it matter for Hong Kong and the coverage universe?
3. What evidence confirms or contradicts the read?
4. What must be checked next?

## Visual hierarchy

The production report uses three recurring visuals and one conditional visual.

### 1. Decision Dashboard — every run

Use one four-panel page:

- Global regime: direction and estimated Hong Kong transmission, not just raw returns.
- Hong Kong confirmation: turnover, funding, flow, short pressure, and A/H dispersion.
- Concentration: the names or channels carrying the flow signal.
- Catalyst ladder: only dated, sourced events that can change the view.

Use a white background, dark navy rules, blue for supportive signals, amber for adverse signals, no gradients, minimal borders, direct labels, and a short decision sequence in the footer. Never mix percentage returns and basis-point moves on one axis.

### 2. Daily One Chart — every run

Select exactly one decision-relevant relationship. Priority order:

1. Southbound or short-selling concentration when verified local data is available.
2. A/H dispersion or turnover conviction when relative-value evidence is strongest.
3. FX/rates versus Hong Kong growth when the external regime dominates.
4. Cross-asset attribution only as a fallback.

The right column must state the takeaway, evidence, confirmation, and invalidation. It must not repeat the dashboard.

### 3. Source-aware signal table — every run

Keep nine high-signal rows in the five-minute scan. Each row contains the last value and move, interpretation, and a confirmation or invalidation test. Indicator definitions and lower-priority monitors belong in the source bundle or appendix.

### 4. Hong Kong Trend Pack — weekly runs only

Use four shared-horizon panels: Southbound persistence, HIBOR/Aggregate Balance, HSI/HSCEI/3033.HK ETF-proxy leadership, and A/H dispersion. Daily reports should not include it because one-day evidence cannot support the same conclusions.

## Content budget

### Layer 1 — five-minute scan

- One top call.
- One Hong Kong implication.
- One confirmation test.
- One invalidation condition.
- Decision Dashboard.
- Nine cross-asset signals, up to six Hong Kong local checks, four risk components, and four checklist items.

Do not repeat a number already visible in the dashboard unless the sentence adds a new mechanism or decision test.

### Layer 2 — causal deep read

Target 25–30 minutes and keep the complete report near 2,200–3,200 English words. The runtime audit warns outside that range and blocks reports above 4,200 words.

Spend detail on cross-asset transmission, local-flow confirmation or contradiction, dated macro/company catalysts, disconfirming evidence, scenario paths, and sourced company implications.

Compress or omit generic indicator definitions, flat moves with no new information, repeated price recaps, undated watchlist catalysts, and headlines without a Hong Kong transmission channel.

### Layer 3 — analyst thinking

Keep the rotating theme, the one decisive chart, and the forward calendar. This layer should identify the next research question rather than restate Layer 2.

### Traceable appendix

Keep source URLs, as-of dates, quality diagnostics, unavailable fields, historical signal performance, and the full visual index here. Audit detail remains accessible without interrupting the scan.

The performance block must show the execution convention, sample size, net result, benchmark result, drawdown, hit rate, unresolved horizons, and data conflicts. Never present reconstructed legacy history as a live portfolio track record.

## Automated versus human research

- GitHub Actions uses deterministic code, MiniMax as the primary narrative provider, and DeepSeek as fallback plus a weekly non-publishing skill shadow runner.
- `morning-note`, `catalyst-calendar`, and `thesis-tracker` shadow outputs are stored in the raw bundle for human comparison and never enter the published report.
- Codex Public Equity Investing and Data Analytics plugins support manual follow-up research only. They are not available inside GitHub Actions and are not production dependencies.
- Claude models are not configured or called.

## Release standard

Every substantive statement must be traceable to a source URL and observation date. Questionable LLM fields are removed and replaced by deterministic copy. Missing provenance, failed critical-source freshness, unresolved critical numeric claims, truncated text that survives fallback, or a disabled look-ahead guard block automated delivery; review-only findings travel as caveats.

## Primary delivery contract

WeCom is the production reading surface. Its first message is a decision brief capped at 3,800 UTF-8 bytes and must retain the base case, confirmation test, invalidation condition, source dates, market tape, and a valid archive link. A self-contained mobile HTML file follows for the full 35–50 minute commute read. Both assets are rendered and audited before publication, sent with bounded retries, and treated as required delivery outcomes. Email is a secondary copy and cannot mask a WeCom failure.
