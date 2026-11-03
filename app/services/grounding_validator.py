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
        original matched evidence text in ``matched_text``.
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
                return GroundingResult(
                    supported=True,
                    reason="supported",
                    matched_text=text,
                )

        return GroundingResult(
            supported=False,
            reason="answer_not_in_evidence",
        )

    except Exception:
        # Fail closed on any unexpected or malformed input.
        return GroundingResult(supported=False, reason="invalid_input")
