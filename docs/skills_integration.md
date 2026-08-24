# Research Skill Architecture

## Production boundary

The 07:30 Hong Kong delivery path must not depend on an interactive plugin, a Claude subscription, browser access, or a new third-party skill repository. Daily production remains:

1. deterministic source collection and analytics;
2. DeepSeek (deepseek-v4-pro) synthesis with MiniMax-M3 fallback;
3. deterministic fact fallback, provenance and source-health checks;
4. commute-policy release audit;
5. WeCom summary and self-contained HTML delivery with machine-readable receipts.

Public Equity Investing and Data Analytics remain Codex-side tools for manual follow-up research. Their useful standards—decision-first framing, claim/source tie-out, visualization integrity, explicit caveats, and release readiness—are encoded in project-owned contracts rather than imported as runtime dependencies.

## Weekly shadow stack

The weekly review runs four provider-agnostic skills through DeepSeek. Outputs are strict JSON, contract-validated, stored only in the raw `skill_shadow` bundle, and never merged into the published report.

| Skill | Job | Why it stays shadow-only |
| --- | --- | --- |
| `morning-note` | Rank one top call and allocate evidence across scan, deep read, and appendix | Its judgment should be compared with the production note before promotion |
| `catalyst-calendar` | Rank only dated, sourced catalysts with scenario and invalidation paths | Calendar evidence can be incomplete or change after the run |
| `thesis-tracker` | Map verified new evidence to supplied coverage theses and disconfirming evidence | Thesis changes require a human research owner |
| `report-evidence-qc` | Audit claims, sources, units, visuals, caveats, and release readiness | It evaluates the report but cannot override deterministic release controls |

Each response is quarantined when required JSON keys are missing or enum values violate its output contract. The weekly cadence protects the daily latency and token budget.

## External skills reviewed

- [Anthropic financial-services](https://github.com/anthropics/financial-services) is the closest public reference for equity-research workflows. Its morning-note, catalyst, thesis, earnings, and sector skills are valuable patterns, but the repository assumes human review and, for many workflows, connected filings, consensus data, models, or office-document tooling. Only the applicable source discipline and decision hierarchy are adapted here.
- [OpenAI role-specific Data Analytics plugins](https://github.com/openai/role-specific-plugins) provide strong validation and visualization standards. Those standards inform `report-evidence-qc` and the report design system; the interactive plugin itself is not callable by GitHub Actions.
- [GitHub awesome-copilot](https://github.com/github/awesome-copilot) contains useful `agentic-eval`, `agent-supply-chain`, and interface-review patterns. `agentic-eval` was not copied into production because an evaluator–optimizer loop would add model cost and latency to the morning SLA. Output-contract validation captures the highest-value reliability benefit deterministically.
- [OpenAI skills catalog](https://github.com/openai/skills) was reviewed. Its current curated install set is primarily developer, deployment, document, and browser oriented; no additional skill improves this financial morning workflow enough to justify another production dependency.

## Promotion rule

A shadow skill may influence the published report only after at least four weekly comparisons show that it improves a named quality metric without increasing unsupported claims, median runtime, or delivery failures. Promotion requires a deterministic parser, tests, explicit source fields, a rollback path, and no Claude-only dependency.
