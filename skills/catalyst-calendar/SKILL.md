---
name: catalyst-calendar
description: Build and prioritize a source-verified catalyst calendar for Hong Kong and offshore-China research. Use when ranking macro releases, earnings, policy events, conferences, corporate actions, or next-open watch items and when defining scenario, confirmation, and invalidation paths for each event.
---

# Catalyst Calendar

## Workflow

1. Accept only dated events with a source URL and as-of timestamp. Put unverified items in `gaps`, never in the calendar.
2. Normalize each event into date, time zone, type, affected assets, expected transmission channel, and evidence status.
3. Rank events by expected market impact, uncertainty, proximity, Hong Kong relevance, and ability to change the current thesis.
4. For each high-priority event, define:
   - the consensus or known baseline, if supplied;
   - the upside and downside scenario paths;
   - the first observable confirmation signal;
   - the invalidation condition;
   - the exact follow-up question for human research.
5. Deduplicate recurring references to the same catalyst across company, macro, and watchlist data.
6. Return strict JSON matching [the output contract](references/output-contract.md).

## Analytical standard

- Do not invent dates, estimates, consensus values, or event importance.
- Distinguish scheduled catalysts from undated monitoring topics.
- Treat an event as actionable only when its transmission channel is explicit.
- Retain provenance fields on every event.
- Do not recommend position changes or execution.
