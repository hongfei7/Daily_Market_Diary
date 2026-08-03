---
name: thesis-tracker
description: Evaluate how new verified evidence changes existing coverage theses and watchlist assumptions. Use for Hong Kong equity thesis checks, disconfirming-evidence reviews, catalyst follow-up, coverage-name prioritization, and identifying what a human analyst should research next.
---

# Thesis Tracker

## Workflow

1. Load each supplied coverage thesis without rewriting it into a stronger claim.
2. Match new evidence to a specific thesis pillar, risk, or catalyst.
3. Classify the effect as `strengthens`, `weakens`, `mixed`, `neutral`, or `insufficient_evidence`.
4. Give disconfirming evidence equal prominence to confirming evidence.
5. Define the next decisive observation and a falsifiable invalidation condition.
6. Rank human follow-up work by materiality and evidence gap, not by headline volume.
7. Return strict JSON matching [the output contract](references/output-contract.md).

## Analytical standard

- Do not create ratings, target prices, position sizes, or trade instructions.
- Do not infer a thesis change from price action alone.
- Mark weak or stale evidence explicitly.
- Preserve source URL and as-of date for every evidence item.
- Use `insufficient_evidence` when no verified evidence maps to the thesis.
