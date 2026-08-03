# Professional Report Design System

## Objective

The report is a decision brief, not a market-data catalogue. It should let a Hong Kong research professional answer four questions quickly:

1. What changed?
2. Why does it matter for Hong Kong and the coverage universe?
3. What evidence confirms or contradicts the read?
4. What must be checked next?

## Visual hierarchy

The production report uses four recurring visual roles and one conditional visual.

### 1. Decision Dashboard — every run

Use one mobile-first, vertically stacked page:

- Global regime: direction and estimated Hong Kong transmission, not just raw returns.
- Hong Kong confirmation: turnover, funding, flow, short pressure, and A/H dispersion.
- Concentration: the names or channels carrying the flow signal.

Use a white background, dark navy rules, blue for supportive signals, amber for adverse signals, no gradients, minimal borders, direct labels, and a short decision sequence in the footer. Never mix percentage returns and basis-point moves on one axis.

### 2. Catalyst & Event Radar — every run

Keep event confidence explicit: confirmed date, bounded window, or undated monitor. Formal calendars lead; configured watchlist triggers provide a non-fabricated weekend fallback. Recent issuer disclosures are a separate backward-looking strip and only receive individual cards when they match the configured watchlist.

### 3. Daily One Chart — every run

Select exactly one decision-relevant relationship. Priority order:

1. Southbound or short-selling concentration when verified local data is available.
2. A/H dispersion or turnover conviction when relative-value evidence is strongest.
3. FX/rates versus Hong Kong growth when the external regime dominates.
4. Cross-asset attribution only as a fallback.

The right column must state the takeaway, evidence, confirmation, and invalidation. It must not repeat the dashboard.

### 4. Source-aware signal table — every run

Keep nine high-signal rows in the five-minute scan. Each row contains the last value and move, interpretation, and a confirmation or invalidation test. Indicator definitions and lower-priority monitors belong in the source bundle or appendix.

### 5. Hong Kong Trend Pack — weekly runs only

Use four shared-horizon panels: Southbound persistence, HIBOR/Aggregate Balance, HSI/HSCEI/3033.HK ETF-proxy leadership, and A/H dispersion. Daily reports should not include it because one-day evidence cannot support the same conclusions.

## Content budget

### Layer 1 — five-minute scan

- One top call.
- One Hong Kong implication.
- One confirmation test.
- One invalidation condition.
- Decision Dashboard and Catalyst & Event Radar.
- Nine cross-asset signals, up to six Hong Kong local checks, four risk components, and four checklist items.

Do not repeat a number already visible in the dashboard unless the sentence adds a new mechanism or decision test.

### Layer 2 — causal deep read

Target 25–30 minutes and keep the complete report near 3,000–4,500 English words including traceability material. The runtime audit blocks reports above 5,200 words.

Spend detail on cross-asset transmission, local-flow confirmation or contradiction, dated macro/company catalysts, disconfirming evidence, scenario paths, and sourced company implications.

Compress or omit generic indicator definitions, flat moves with no new information, repeated price recaps, and headlines without a Hong Kong transmission channel. Undated watchlist catalysts may appear only as explicitly labelled monitoring triggers, never as scheduled events.

### Layer 3 — analyst thinking

Keep the rotating theme, the one decisive chart, and the forward calendar. This layer should identify the next research question rather than restate Layer 2.

### Traceable appendix

Keep source URLs, as-of dates, quality diagnostics, unavailable fields, historical signal performance, and the full visual index here. Audit detail remains accessible without interrupting the scan.

The performance block must show the execution convention, sample size, net result, benchmark result, drawdown, hit rate, unresolved horizons, and data conflicts. Never present reconstructed legacy history as a live portfolio track record.

## Automated versus human research

- GitHub Actions uses deterministic code, MiniMax as the primary narrative provider, and DeepSeek as fallback plus a weekly non-publishing skill shadow runner.
- `morning-note`, `catalyst-calendar`, `thesis-tracker`, and `report-evidence-qc` shadow outputs are contract-validated, stored in the raw bundle for human comparison, and never enter the published report.
- Codex Public Equity Investing and Data Analytics plugins support manual follow-up research only. They are not available inside GitHub Actions and are not production dependencies.
- Claude models are not configured or called.

## Release standard

Every substantive statement must be traceable to a source URL and observation date. Questionable LLM fields are removed and replaced by deterministic copy. The strict research policy continues to flag critical source-health and fact-check findings for human review. The commute delivery policy keeps those findings visible as release caveats instead of silently suppressing the morning report; malformed or missing artifacts, invalid provenance, broken tables, oversized content, and a disabled look-ahead guard remain hard blockers.

## Primary delivery contract

WeCom is the production reading surface. Its first message is a decision brief capped at 3,800 UTF-8 bytes and must retain the base case, confirmation test, invalidation condition, source dates, market tape, and a valid archive link. A self-contained mobile HTML file follows for the full 40–50 minute commute read. The working content budget is 3,000–4,500 words including traceability material, with a 5,200-word hard stop for genuine sprawl. Both assets are rendered and audited before publication, sent with bounded retries, and treated as required delivery outcomes. Each successful send writes a JSON receipt with the message kind, timestamp, report date, and WeCom response code. If a delivery-ready report cannot be produced, a short incident message explains the failed stage and the scheduled recovery behavior. Email is a secondary copy and cannot mask a WeCom failure.

## HTML design language

Use a conservative strategy-report system inspired by leading professional-services publications without copying their branding: black editorial rules, white paper, a single restrained blue accent, answer-first headlines, strong exhibit hierarchy, quiet metadata, no gradients, and no decorative dashboard cards. The masthead carries the issue, source dates, edition, quality state, and a three-step commute reading path. The executive summary uses three evidence-led columns on desktop and a single scan path on mobile. Full tables remain horizontally scrollable in WeCom, while the contents list collapses behind a native mobile disclosure control.
