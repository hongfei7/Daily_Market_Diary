---
name: report-evidence-qc
description: Audit a source-backed Hong Kong or offshore-China market report before human review. Use when checking headline claims, numerical consistency, source and as-of coverage, chart-to-narrative alignment, caveats, release readiness, or whether weak evidence has been presented too confidently.
---

# Report Evidence QC

## Workflow

1. Inventory the top call, Hong Kong implication, confirmation test, invalidation condition, headline numbers, charts, source records, and release caveats.
2. Check every decision-relevant claim against the supplied deterministic fields. Mark a claim `unsupported` when the evidence is absent; do not repair it with outside knowledge.
3. Check source URL, observation date, instrument identity, unit, comparison basis, and freshness. Treat percentage returns, yield changes, basis points, and currency amounts as different units.
4. Check that each visual answers a decision question and that its title, scale, labels, source, takeaway, confirmation, and invalidation agree with the underlying evidence.
5. Separate verified fact, bounded interpretation, caveat, and unknown. Flag causal language, stale cash-tape claims, repeated indicator definitions, and generic “why it matters” copy.
6. Assign `ready`, `share_with_caveats`, or `needs_revision`. Reserve `needs_revision` for a broken headline claim, unit/date error, missing provenance on decisive evidence, misleading visual, or missing release caveat.
7. Return strict JSON matching [the output contract](references/output-contract.md).

## Standards

- Prefer a small set of material findings over exhaustive copyediting.
- Give disconfirming evidence equal prominence to confirming evidence.
- Never invent a source, number, date, catalyst, or explanation.
- Do not issue investment recommendations, ratings, targets, position sizes, or trade instructions.
- Do not publish or rewrite the production report. This skill is a shadow QA layer for human comparison.
- Limit the audit to five priority fixes and eight claim checks.
