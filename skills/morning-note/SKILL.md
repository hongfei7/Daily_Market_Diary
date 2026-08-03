---
name: morning-note
description: Create a decision-first institutional morning market note from a verified market bundle. Use for daily or weekly Hong Kong research briefings, overnight reviews, opening-call preparation, signal prioritization, and deciding which evidence belongs in the scan, deep-read, or appendix layers.
---

# Morning Note

## Workflow

1. Reject claims without a source URL, observation date, or supported value.
2. Separate the evidence into facts, bounded interpretation, and explicit unknowns.
3. Select one top call by materiality, freshness, Hong Kong relevance, and evidence quality.
4. Explain the mechanism: what changed, why it matters, where it transmits, and what confirms or invalidates the read.
5. Allocate content by decision value:
   - `scan`: the top call, four decisive signals, and today's confirmation checks.
   - `deep_read`: causal analysis, local-flow confirmation, scenarios, and company implications.
   - `appendix`: complete data tables, source metadata, and lower-priority observations.
6. Keep the top layer readable in five minutes. Do not repeat the same market move across layers.
7. Return strict JSON matching [the output contract](references/output-contract.md).

## Analytical standard

- Prefer a falsifiable read over a generic market label.
- Never use an indicator definition as the conclusion. Translate each signal into implication, confirmation, and invalidation.
- Treat correlation as a hypothesis unless the supplied evidence establishes causality.
- State when Hong Kong cash-market evidence is stale or unavailable.
- Do not issue investment recommendations, price targets, or trade instructions.
- Preserve source URLs and as-of dates for every factual claim.
