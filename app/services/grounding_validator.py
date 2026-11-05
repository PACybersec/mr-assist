"""Deterministic, fail-closed grounding validator for Mr.Assist.

TASK 06.3.1 — Strict extractive grounding validator.

Design constraints (see TASK 06.3 / TASK 06.3-R):

- The candidate answer must be found as ONE contiguous span inside a single
  accepted evidence chunk.
- Matching uses conservative normalization (whitespace collapse + case folding).
- No fuzzy matching, lexical-overlap thresholds, stemming or LLM judge.
- Any unexpected or malformed input fails closed (supported=False).

Instruction filtering scope note:
    The injected-pattern filter below is intentionally narrow and deterministic
    for obvious prompt-injection signals (forged system/developer instructions,
    requests to ignore previous instructions, requests to reveal system prompts
    or internal secrets, assistant role overrides). It rejects suspicious spans
    instead of rewriting them.

    This is NOT a complete prompt-injection defense. Ordinary IT procedures
    containing shell commands are not treated as malicious by this filter.

    See TASK 06.3.1 scope: deterministic, fail-closed, extractive validation.
"""

import re
from dataclasses import dataclass


@dataclass
class GroundingResult:
    """Outcome of the deterministic grounding validation."""

    supported: bool
    reason: str
    matched_text: str | None = None


_INJECTION_PATTERNS = (
    re.compile(
        r"ignore\s+(?:all|the|previous|prior|above|these|those)?\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"disregard\s+(?:all|the|previous|prior|above|these|those)?\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"forget\s+(?:all|the|previous|prior|above|these|those)?\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"override\s+(?:all|the|previous|above|these|those)?\s+instructions",
        re.IGNORECASE,
    ),
    re.compile(
        r"reveal\s+(?:system\s+|internal\s+|your\s+|the\s+)*(?:prompt|instructions|secrets?)",
        re.IGNORECASE,
    ),
    re.compile(r"you\s+are\s+now", re.IGNORECASE),
    re.compile(r"from\s+now\s+on", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+(?:mode|message|instruction)", re.IGNORECASE),
)


def _normalize(text: str) -> str:
    """Conservative normalization: collapse whitespace, fold case."""
    return " ".join(text.split()).casefold()


def _normalize_with_offsets(text: str) -> tuple[str, list[int]]:
    """Normalize text like _normalize and map each normalized char to its origin.

    Returns a tuple ``(normalized_text, offsets)`` where each character of the
    normalized text is paired with the index of its originating character in
    ``text``. Whitespace runs collapse to a single normalized space mapped to
    the first whitespace position of the run. Case-folded characters keep the
    offset of their original character, including multi-character expansions
    such as ``ß`` → ``ss``.
    """
    normalized: list[str] = []
    offsets: list[int] = []
    pending_space = False
    started = False

    for index, char in enumerate(text):
        if char.isspace():
            if started:
                pending_space = True
            continue
        if pending_space:
            normalized.append(" ")
            offsets.append(index)
            pending_space = False
        for folded in char.casefold():
            normalized.append(folded)
            offsets.append(index)
        started = True

    return "".join(normalized), offsets


def _extract_original_span(text: str, normalized_answer: str) -> str | None:
    """Recover the exact original substring matching the normalized answer.

    Returns ``None`` when the span cannot be located or verified to round-trip
    exactly through normalization. Never guesses or substitutes input text.
    """
    normalized_text, offsets = _normalize_with_offsets(text)
    position = normalized_text.find(normalized_answer)
    if position == -1:
        return None

    start = offsets[position]
    end = offsets[position + len(normalized_answer) - 1] + 1
    span = text[start:end]

    # Verify the recovered span round-trips exactly through normalization and
    # remains an exact substring of the source text. Never accept a span that
    # cannot be independently verified; span reconstruction failures fail
    # closed instead of substituting the candidate text.
    if not span:
        return None
    if _normalize(span) != normalized_answer:
        return None

    return span


def _contains_injection(text: str) -> bool:
    """Deterministic check for obvious prompt-injection patterns."""
    return any(pattern.search(text) for pattern in _INJECTION_PATTERNS)


def validate_answer(question: str, answer: str, evidence: list) -> GroundingResult:
    """Validate that the candidate answer is an extractive grounded span.

    Args:
        question: User question. Must be a non-empty string (fail closed).
        answer: Candidate model answer. Must be a non-empty string.
        evidence: Accepted evidence chunks. Must be a non-empty list of dicts
            with a non-empty string ``text`` field.

    Returns:
        GroundingResult with ``supported``, ``reason`` and, on success, the
        exact original matched span in ``matched_text``.
    """
    try:
        if not isinstance(question, str) or not question.strip():
            return GroundingResult(supported=False, reason="invalid_question")

        if not isinstance(answer, str) or not answer.strip():
            return GroundingResult(supported=False, reason="empty_answer")

        if not isinstance(evidence, list) or len(evidence) < 1:
            return GroundingResult(supported=False, reason="missing_evidence")

        for item in evidence:
            if not isinstance(item, dict):
                return GroundingResult(supported=False, reason="malformed_evidence")
            text = item.get("text")
            if not isinstance(text, str) or not text.strip():
                return GroundingResult(supported=False, reason="malformed_evidence")

        if _contains_injection(answer):
            return GroundingResult(supported=False, reason="suspicious_instruction")

        normalized_answer = _normalize(answer)
        if not normalized_answer:
            return GroundingResult(supported=False, reason="empty_answer")

        for item in evidence:
            text = item.get("text")
            normalized_evidence = _normalize(text)
            if normalized_answer in normalized_evidence:
                span = _extract_original_span(text, normalized_answer)
                if span is None:
                    # Span reconstruction failed verification; fail closed.
                    # Never substitute the candidate text for an unverified span.
                    return GroundingResult(
                        supported=False,
                        reason="span_recovery_failed",
                    )
                return GroundingResult(
                    supported=True,
                    reason="supported",
                    matched_text=span,
                )

        return GroundingResult(
            supported=False,
            reason="answer_not_in_evidence",
        )

    except Exception:
        # Fail closed on any unexpected or malformed input.
        return GroundingResult(supported=False, reason="invalid_input")
