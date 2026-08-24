from __future__ import annotations

import re
from typing import List


def _split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", str(text or "")).strip()
    if not normalized:
        return []
    protected = {
        "vs.": "vs<prd>",
        "Vs.": "Vs<prd>",
        "e.g.": "e<prd>g<prd>",
        "i.e.": "i<prd>e<prd>",
        "U.S.": "U<prd>S<prd>",
        "U.K.": "U<prd>K<prd>",
    }
    for needle, replacement in protected.items():
        normalized = normalized.replace(needle, replacement)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9`$])", normalized)
    restored = []
    for part in parts:
        for needle, replacement in protected.items():
            part = part.replace(replacement, needle)
        if part.strip():
            restored.append(part.strip())
    return restored


def _paragraph_chunks(text: str, max_sentences: int = 1, max_chars: int = 240, limit: int = 3) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for sentence in sentences:
        projected_len = current_len + len(sentence) + (1 if current else 0)
        if current and (len(current) >= max_sentences or projected_len > max_chars):
            chunks.append(" ".join(current))
            current = []
            current_len = 0
            if len(chunks) >= limit:
                break
        current.append(sentence)
        current_len += len(sentence) + 1

    if current and len(chunks) < limit:
        chunks.append(" ".join(current))
    return chunks[:limit]


def _brief_points(text: str, limit: int = 3, width: int = 190) -> List[str]:
    sentences = _split_sentences(text)
    if not sentences:
        stripped = str(text or "").strip()
        sentences = [stripped] if stripped else []
    return [_condense_sentence(sentence, width) for sentence in sentences[:limit] if str(sentence or "").strip()]


# Words that cannot legitimately end a sentence. Cutting in front of one of
# these leaves a fragment such as "flagged as the dominant." or
# "so the right lean is to keep.", which reads as a broken generator.
_DANGLING_TAIL_RE = re.compile(
    r"\b(?:the|a|an|and|or|but|with|without|to|from|for|of|in|on|at|by|as|that|than|rather|while|which|"
    r"is|was|were|are|be|been|being|has|have|had|its|their|his|her|our|your|this|these|those|"
    r"into|onto|above|below|around|via|through|against|despite|after|before|during|between|"
    r"could|can|would|may|might|should|will|shall|must|more|less|most|least|very|dominant|cleanest)$",
    re.IGNORECASE,
)


# A trailing "…, <word>" means the cut landed inside an enumeration, e.g.
# "growth, platform" severed from "…, and consumer-internet names".
_SEVERED_LIST_RE = re.compile(r",\s+[\w/-]+$")


def _is_well_formed(phrase: str) -> bool:
    """Reject clause fragments that would read as an unfinished sentence."""
    stripped = phrase.rstrip(" ,;:-").strip()
    if len(stripped) < 25:
        return False
    if _DANGLING_TAIL_RE.search(stripped):
        return False
    if _SEVERED_LIST_RE.search(stripped):
        return False
    # An opened bracket or quote that never closes is a truncation artefact.
    if stripped.count("(") != stripped.count(")"):
        return False
    if stripped.count("[") != stripped.count("]"):
        return False
    if stripped.count('"') % 2:
        return False
    return True


def _condense_sentence(text: str, width: int) -> str:
    """Shorten a sentence only where a clean clause boundary exists.

    Over-running the width budget is preferable to emitting a fragment: the
    width is a layout preference, but a broken sentence is a correctness defect
    that lands in the highest-visibility part of the report.
    """
    sentence = " ".join(str(text or "").split()).strip()
    if len(sentence) <= width:
        return sentence

    # ", and " / ", or " are excluded: they join serial list items far more often
    # than independent clauses, so cutting there severs an enumeration.
    boundary_markers = [", while ", ", which ", ", but ", "; ", ": "]
    candidates: List[str] = []
    for marker in boundary_markers:
        pos = sentence.find(marker)
        if 35 <= pos <= width:
            candidate = sentence[:pos].strip()
            if _is_well_formed(candidate):
                candidates.append(candidate)

    if not candidates:
        # No safe cut point: keep the sentence whole rather than mangle it.
        return sentence

    phrase = max(candidates, key=len).strip()
    if phrase and phrase[-1] not in ".!?":
        phrase = phrase.rstrip(" ,;:-") + "."
    return phrase


def _render_labeled_points(
    text: str,
    labels: List[str],
    *,
    fallback: str = "",
    limit: int = 3,
    width: int = 190,
) -> List[str]:
    points = _brief_points(text, limit=limit, width=width)
    if not points and fallback:
        return [f"- **Desk read:** {fallback}"]

    lines: List[str] = []
    for idx, point in enumerate(points):
        label = labels[idx] if idx < len(labels) else "Additional read"
        lines.append(f"- **{label}:** {point}")
    return lines


def _render_labeled_paragraphs(
    text: str,
    labels: List[str],
    *,
    fallback: str = "",
    limit: int = 3,
    width: int = 230,
) -> List[str]:
    points = _brief_points(text, limit=limit, width=width)
    if not points and fallback:
        points = [fallback]

    lines: List[str] = []
    for idx, point in enumerate(points):
        label = labels[idx] if idx < len(labels) else "Additional read"
        lines.append(f"**{label}.** {point}")
    return lines


def _compact_bullets(items: List[str], limit: int = 4, width: int = 150) -> List[str]:
    bullets: List[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            bullets.append(_condense_sentence(text, width))
        if len(bullets) >= limit:
            break
    return bullets


def _clean_report_spacing(text: str) -> str:
    """Keep generated markdown readable without changing table rows."""
    replacements = {
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": " - ",
        "\u2014": " - ",
        "\u2015": " - ",
        "\u00a0": " ",
        "\u200b": "",
        "\ufeff": "",
    }
    cleaned = text
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    # Do NOT strip CJK/Kana/Hangul: a HK/China product can legitimately carry
    # Chinese company names and filing titles, and silently deleting them mangled
    # facts (e.g. "小米集团 (1810.HK)" -> " (1810.HK)"). The runtime audit flags
    # non-English script as a warning instead of destroying it.
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)

    spaced_lines: List[str] = []
    previous_was_table = False
    for line in cleaned.splitlines():
        is_table = line.strip().startswith("|")
        if is_table and spaced_lines and spaced_lines[-1].strip() and not spaced_lines[-1].strip().startswith("|"):
            spaced_lines.append("")
        if previous_was_table and line.strip() and not is_table:
            spaced_lines.append("")
        spaced_lines.append(line.rstrip())
        previous_was_table = is_table

    cleaned = "\n".join(spaced_lines)
    cleaned = re.sub(r"\n{4,}", "\n\n\n", cleaned)
    return cleaned.strip() + "\n"
