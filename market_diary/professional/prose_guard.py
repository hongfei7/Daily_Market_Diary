"""Pre-publication prose checks for the rendered markdown report.

The existing test suite verifies that sections, tables and fields exist. It does
not verify that the resulting English is publishable, which is how sentence
fragments, unbalanced brackets and internal identifiers reached the executive
summary of shipped reports.

This module reads the final markdown and reports defects that a reader would
immediately notice. It is deliberately conservative: it only flags patterns that
cannot occur in well-formed research prose, so a finding is always actionable.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List

# Words that cannot end a sentence. A period straight after one of these is the
# signature of a hard truncation, e.g. "flagged as the dominant."
# Words that cannot end a sentence under any reading: determiners, conjunctions,
# bare auxiliaries and comparatives. A period after one of these is always the
# signature of a hard truncation, e.g. "flagged as the dominant."
# The lookbehind keeps hyphenated compounds ("Risk-On.") from matching.
_DANGLING_TAIL_RE = re.compile(
    r"(?<![-\w])(?:the|a|an|and|or|but|as|that|than|rather|while|which|"
    r"is|was|were|are|be|been|being|has|have|had|its|their|our|your|this|these|those|"
    r"could|can|would|may|might|should|will|shall|must|more|less|most|least|very|dominant|cleanest)\.(?:\s|$)",
    re.IGNORECASE,
)

# Prepositions legitimately end a sentence when stranded after a verb
# ("the catalyst to prepare for."). They only signal truncation when they follow
# a determiner, which means the cut severed the noun phrase they governed
# ("keeping the lens benign and removing the.").
_STRANDED_PREPOSITION_RE = re.compile(
    r"(?<![-\w])(?:the|a|an|this|that|these|those|its|their|our|your|any|current|next)\s+"
    r"(?:\w+\s+)?(?:with|without|to|from|for|of|in|on|at|by|into|onto|above|below|around|via|"
    r"through|against|despite|after|before|during|between)\.(?:\s|$)",
    re.IGNORECASE,
)

# Cut inside an enumeration: "…that growth, platform." severed from ", and …".
_SEVERED_LIST_RE = re.compile(r",\s+[\w/-]+\.(?:\s|$)")

# Internal plumbing that must never surface in reader-facing prose.
_INTERNAL_ID_RE = re.compile(
    r"(?:\battribution_v\d+\b|\btasks\.[a-z_]+|\b[a-z_]+_v\d+\b(?!\s*schema)|\b\w+\.\w+\[\d+\])",
    re.IGNORECASE,
)

# Lines that are structural rather than prose.
_SKIP_PREFIXES = ("|", "![", "#", "```", "> Date policy", "_Source:", "<div", "</div", "<p", "<span")

# Traceability lines in the audit appendix legitimately quote internal field
# paths; they are diagnostics for the desk, not reader-facing prose.
_DIAGNOSTIC_MARKERS = ("Deterministic fallback fields:", "Adapter status", "records checked:")


def _is_prose_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if stripped.startswith(_SKIP_PREFIXES):
        return False
    # Bare URLs and link-only bullets carry no prose to check.
    if stripped.startswith("- [") and stripped.endswith(")"):
        return False
    return True


def _strip_markdown(line: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", line)
    text = text.replace("**", "").replace("`", "")
    text = re.sub(r"^[-*]\s+", "", text.strip())
    return text.strip()


def _bracket_findings(text: str, line_no: int) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []
    for opener, closer, name in (("(", ")", "parenthesis"), ("[", "]", "bracket")):
        if text.count(opener) != text.count(closer):
            findings.append(
                {
                    "line": line_no,
                    "rule": "unbalanced_bracket",
                    "detail": f"Unbalanced {name}",
                    "text": text[:160],
                }
            )
    if text.count('"') % 2:
        findings.append(
            {"line": line_no, "rule": "unbalanced_quote", "detail": "Unbalanced double quote", "text": text[:160]}
        )
    return findings


def check_markdown(markdown: str) -> List[Dict[str, Any]]:
    """Return every prose defect found in the rendered report."""
    findings: List[Dict[str, Any]] = []
    previous_prose = ""

    for idx, raw_line in enumerate(str(markdown or "").splitlines(), start=1):
        if not _is_prose_line(raw_line):
            continue
        text = _strip_markdown(raw_line)
        if not text:
            continue

        if _DANGLING_TAIL_RE.search(text) or _STRANDED_PREPOSITION_RE.search(text):
            findings.append(
                {
                    "line": idx,
                    "rule": "sentence_fragment",
                    "detail": "Sentence ends on a function word",
                    "text": text[:160],
                }
            )
        if _SEVERED_LIST_RE.search(text):
            findings.append(
                {
                    "line": idx,
                    "rule": "severed_list",
                    "detail": "Sentence ends mid-enumeration",
                    "text": text[:160],
                }
            )
        match = None if any(marker in text for marker in _DIAGNOSTIC_MARKERS) else _INTERNAL_ID_RE.search(text)
        if match:
            findings.append(
                {
                    "line": idx,
                    "rule": "internal_identifier",
                    "detail": f"Internal identifier '{match.group(0)}' leaked into prose",
                    "text": text[:160],
                }
            )
        findings.extend(_bracket_findings(text, idx))

        # A verbatim repeat of the previous prose line is duplicated rendering.
        if len(text) > 60 and text == previous_prose:
            findings.append(
                {
                    "line": idx,
                    "rule": "duplicate_line",
                    "detail": "Line repeats the previous prose line verbatim",
                    "text": text[:160],
                }
            )
        previous_prose = text

    return findings


def summarize(findings: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate findings into a report-quality component payload."""
    by_rule: Dict[str, int] = {}
    for item in findings:
        rule = str(item.get("rule", "unknown"))
        by_rule[rule] = by_rule.get(rule, 0) + 1

    total = len(findings)
    # Each defect is individually visible to the reader, so the penalty is steep.
    score = max(0.0, 100.0 - 20.0 * total)
    if total == 0:
        read = "No prose defects detected in the rendered report."
    else:
        parts = ", ".join(f"{count} {rule}" for rule, count in sorted(by_rule.items()))
        read = f"{total} prose defect(s) detected: {parts}."

    return {
        "status": "ok" if total == 0 else "warning",
        "score": round(score, 1),
        "total": total,
        "by_rule": by_rule,
        "findings": findings[:20],
        "read": read,
    }
