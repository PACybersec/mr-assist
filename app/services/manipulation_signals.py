"""Local log of explicit manipulation signals (instruction override).

TASK 16 — BUILD (limited scope).

Design constraints:

- Only the ``instruction_override`` category is handled in this first version.
- Only the user question is inspected in this first version; evidence and
  candidates are explicitly out of scope.
- The detector is a small deterministic heuristic. It is presented as a
  heuristic, not infallible detection. It returns rule identifiers only, never
  user text.

- Events are written as JSON lines to a local rotating file.
- Failures in detection or writing never interrupt the recovery/answer flow.
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

DETECTOR_VERSION = "1.0.0"
CATEGORY_INSTRUCTION_OVERRIDE = "instruction_override"
LOGGER_NAME = "mr_assist.manipulation_signals"

DEFAULT_LOG_PATH = (
    Path(__file__).resolve().parents[2]
    / "app"
    / "data"
    / "security_events"
    / "manipulation_signals.jsonl"
)

ROTATION_MAX_BYTES = 1 * 1024 * 1024  # 1 MiB
ROTATION_BACKUP_COUNT = 3

_INSTRUCTION_OVERRIDE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override.es.ignora_instrucciones",
        re.compile(
            r"\bignora\s+(?:las\s+)?(?:cualquier\s+)?"
            r"(?:instrucciones|reglas|indicaciones|regla|indicacion|instruccion)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override.es.olvida_instrucciones",
        re.compile(
            r"\bolvida\s+(?:las\s+)?(?:cualquier\s+)?"
            r"(?:instrucciones|reglas|indicaciones|regla|indicacion|instruccion)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override.es.descarta_instrucciones",
        re.compile(
            r"\bdescarta\s+(?:las\s+)?(?:cualquier\s+)?"
            r"(?:instrucciones|reglas|indicaciones|regla|indicacion|instruccion)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override.en.ignore_previous",
        re.compile(
            r"\bignore\s+(?:(?:all|the|previous|above|prior|these|those)\s+)*"
            r"(?:instructions?|prompts?|rules|instruction|prompt|rule)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override.en.disregard_previous",
        re.compile(
            r"\bdisregard\s+(?:(?:all|the|previous|above|prior|these|those)\s+)*"
            r"(?:instructions?|prompts?|rules|instruction|prompt|rule)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "instruction_override.en.forget_previous",
        re.compile(
            r"\bforget\s+(?:(?:all|the|previous|above|prior|these|those)\s+)*"
            r"(?:instructions?|prompts?|rules|instruction|prompt|rule)\b",
            re.IGNORECASE,
        ),
    ),
)


def detect_instruction_override_rules(question: str) -> list[str]:
    """Return the identifiers of the instruction-override rules that match.

    Only explicit override signals are considered (verb + instruction object).
    Isolated words such as "firewall", "secrets", "jailbreak" or "actúa como" are
    deliberately not detected.

    Returns:
        A list of matched rule identifiers (empty when no signal is detected).
    """
    if not isinstance(question, str):
        return []

    matched: list[str] = []

    for rule_id, pattern in _INSTRUCTION_OVERRIDE_RULES:
        if pattern.search(question):
            matched.append(rule_id)

    return matched


def create_manipulation_event(
    category: str,
    rules: list[str],
    detector_version: str,
    timestamp_utc: datetime | str | None = None,
    event_id: str | None = None,
) -> dict:
    """Create the structured local event for one suspicious submission."""
    if isinstance(timestamp_utc, datetime):
        timestamp_utc = timestamp_utc.astimezone(timezone.utc).isoformat()

    return {
        "timestamp_utc": timestamp_utc if timestamp_utc is not None else datetime.now(timezone.utc).isoformat(),
        "event_id": event_id if event_id is not None else str(uuid.uuid4()),
        "category": category,
        "rules": list(rules),
        "detector_version": detector_version,
    }


def get_manipulation_logger(path: Path | str | None = None) -> logging.Logger:
    """Return the manipulation-signal logger, ensuring a single file handler."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    resolved = Path(path) if path is not None else DEFAULT_LOG_PATH
    resolved.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        str(resolved),
        maxBytes=ROTATION_MAX_BYTES,
        backupCount=ROTATION_BACKUP_COUNT,
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def emit_manipulation_event(event: dict, logger: logging.Logger) -> bool:
    """Write one event as a single JSON line. Fail-safe at the caller."""
    logger.info(json.dumps(event))
    return True


def record_suspicious_submission(
    question: str,
    logger: logging.Logger | None = None,
    path: Path | str | None = None,
) -> dict | None:
    """Detect, build and write one manipulation event per suspicious submission.

    Fail-safe: detection or writing errors are captured and no event is produced,
    so the normal answer flow is never interrupted.
    """
    try:
        rules = detect_instruction_override_rules(question)
        if not rules:
            return None

        event = create_manipulation_event(
            category=CATEGORY_INSTRUCTION_OVERRIDE,
            rules=rules,
            detector_version=DETECTOR_VERSION,
        )

        target_logger = logger if logger is not None else get_manipulation_logger(path)
        if not emit_manipulation_event(event, target_logger):
            return None

        return event
    except Exception:
        return None
