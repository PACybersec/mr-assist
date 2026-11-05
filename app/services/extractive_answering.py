"""Deterministic, conservative extractive answer selection.

This module is independent from Streamlit, Ollama and ChromaDB. It receives
evidence chunks that the caller has already accepted and returns, at most, one
exact contiguous excerpt from one of those chunks.

Limits of deterministic relevance assessment
--------------------------------------------
The selector is a lexical and structural heuristic, not a semantic engine. It
cannot reliably resolve paraphrases, synonyms with different stems,
cross-lingual questions, implicit answers, or contradictions between sources.
It therefore fails closed: when content coverage, question intent or ambiguity
checks are not satisfied, it returns ``None`` instead of guessing.

The prompt-injection filter is deliberately narrow. It rejects obvious
instruction-override patterns but is not a complete security control.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MIN_COVERAGE = 0.75
_AMBIGUITY_EPSILON = 1e-9

# Generic concept tokens that describe the kind of answer being requested
# rather than specific content that must appear in the evidence. A question
# can be answerable even when the generic concept word itself does not appear
# in the evidence (e.g. "¿Qué señales indican un problema de disco?" can be
# answered by disk-usage evidence that never uses the word "problema").
# These tokens are exempt from the document-level relevance guard. Only
# genuinely generic Spanish concepts are listed here — never IT-specific
# terms such as SLA, port, or command.
_GENERIC_CONCEPT_TOKENS = frozenset({"problem", "problema", "problemas", "procediment", "procedimiento", "procedimientos"})
_ACCENTS = str.maketrans("áéíóúüñ", "aeiouun")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")
_LIST_MARKER_RE = re.compile(r"^\s*(?:\d+[.)]|[-*•])\s+")
_SENTENCE_END_RE = re.compile(r"[.!?]+(?=\s|$)")

_STOPWORDS = frozenset(
    """
    a al algo algun alguna algunas alguno algunos ante antes como con contra
    cual cuales cuando de del desde donde durante e el ella ellas ellos en
    entre era eran es esa esas ese eso esos esta estaba estan estar estas este
    esto estos fue fueron ha habia han hasta hay la las le les lo los mas me
    mi mis mucho muy nada ni no nos nosotros o os otra otro para pero poco por
    porque que quien quienes se sea segun ser si sin sobre son su sus tambien
    tan tanto te tengo tiene tienen todo todos tu tus un una unas uno unos
    usted ustedes y ya yo debo debe deben deberia puedo puede pueden podria
    hacer hago hace hacen necesito necesita
    """.split()
)

# Generic question-framing terms describe the kind of answer being requested
# (symptoms, steps, recommendations, ...) rather than the IT topic itself.
# They are removed from the coverage denominator so that a question such as
# "¿Qué síntomas indican un problema de disco?" is judged on its topic terms
# (problema, disco) instead of being diluted by "síntomas"/"indican".
# Operational terms (comando, configuración, contraseña, firewall, SSH,
# systemctl, autorización, etc.) are deliberately NOT included here.
_GENERIC_QUESTION_FRAME_WORDS = frozenset(
    """
    sintoma sintomas sintomatico sintomatica sintomaticos sintomaticas
    sintomatologia signo signos senal senales indicador indicadores
    indica indican indicar indique indiquen
    paso pasos instruccion instrucciones
    recomendacion recomendaciones consejo consejos guia guias
    recomienda recomiendan recomendar recomiendo
    aconseja aconsejan aconsejar
    definicion significado
    seguir sigue siguen siga sigan
    """.split()
)

_SUFFIXES = (
    "aciones",
    "acion",
    "ciones",
    "cion",
    "mente",
    "idades",
    "idad",
    "ando",
    "iendo",
    "ados",
    "adas",
    "idos",
    "idas",
    "ado",
    "ada",
    "ido",
    "ida",
    "ar",
    "er",
    "ir",
    "es",
    "s",
)

_ACTION_VERBS = frozenset(
    {
        "comprueba",
        "comprueban",
        "verifica",
        "verifican",
        "revisa",
        "revisan",
        "ejecuta",
        "ejecutan",
        "ejecute",
        "ejecuten",
        "usa",
        "usan",
        "use",
        "usen",
        "utiliza",
        "utilizan",
        "confirma",
        "confirman",
        "cambia",
        "cambian",
        "desactiva",
        "desactivan",
        "reinicia",
        "reinician",
        "escala",
        "escalan",
        "registra",
        "registran",
        "evita",
        "evitan",
        "sigue",
        "siguen",
        "siga",
        "sigan",
        "aplica",
        "aplican",
        "configura",
        "configuran",
        "instala",
        "instalan",
        "actualiza",
        "actualizan",
        "elimina",
        "eliminan",
        "crea",
        "crean",
        "abre",
        "abren",
        "cierra",
        "cierran",
        "inicia",
        "inician",
        "detiene",
        "detienen",
        "contacta",
        "contactan",
        "solicita",
        "solicitan",
        "documenta",
        "documentan",
        "notifica",
        "notifican",
        "restaura",
        "restauran",
        "recupera",
        "recuperan",
        "valida",
        "validan",
        "monitoriza",
        "monitorizan",
        "hace",
        "hacen",
        "funciona",
        "funcionan",
    }
)

# A bare infinitive after one of these words is a purpose/complement, not an
# instruction: "para reiniciar", "de reiniciar", "sin reiniciar".
_INFINITIVE_BLOCKERS = frozenset(
    {"para", "de", "del", "al", "sin", "por", "que", "como", "cuando", "donde"}
)

# An action verb preceded by a subject or the reflexive "se" is descriptive:
# "el firewall se configura", "el equipo reinicia". It is not an instruction.
_ACTION_VERB_SUBJECT_BLOCKERS = frozenset(
    {"se", "que", "el", "la", "los", "las", "un", "una", "unos", "unas"}
)

_INJECTION_PATTERNS = (
    re.compile(
        r"\b(ignore|disregard|forget|override)\s+(?:all\s+)?(?:the\s+)?"
        r"(?:previous|prior|above|these|those)?\s*instructions?\b"
    ),
    re.compile(
        r"\b(ignora|olvida|descarta|anula|sobrescribe)\s+"
        r"(?:las?\s+)?(?:instrucciones|indicaciones)\b"
    ),
    re.compile(
        r"\b(reveal|show|print|expose)\s+(?:the\s+)?"
        r"(?:system\s+prompt|internal\s+instructions|secrets?)\b"
    ),
    re.compile(
        r"\b(revela|muestra|imprime|expone)\s+(?:el\s+)?"
        r"(?:prompt\s+del\s+sistema|instrucciones\s+internas|secretos?)\b"
    ),
    re.compile(
        r"\b(system\s+prompt|developer\s+(?:mode|message|instruction)|jailbreak)\b"
    ),
    re.compile(r"\b(you\s+are\s+now|from\s+now\s+on|act\s+as)\b"),
)

_DEFINITION_CUE_RE = re.compile(
    r"\b(es|son|se define|significa|consiste|se refiere|se entiende)\b"
)
_YES_NO_CUE_RE = re.compile(
    r"\b(no|nunca|jamas|puede|pueden|debe|deben|hay que|es necesario|"
    r"es obligatorio|autoriz\w*|permit\w*|prohib\w*|evit\w*|recomend\w*|"
    r"requiere|sin)\b"
)
_REASON_CUE_RE = re.compile(r"\b(porque|debido|a causa|motivo|razon|ya que)\b")
_TIME_CUE_RE = re.compile(
    r"\b(cuando|antes|despues|durante|hora|fecha|plazo|inmediatamente)\b"
)
_LOCATION_CUE_RE = re.compile(
    r"\b(donde|ubicad\w*|ruta|directorio|carpeta|servidor|equipo)\b"
)
_PERSON_CUE_RE = re.compile(
    r"\b(equipo|responsable|usuario|administrador|soporte|persona|area|"
    r"departamento)\b"
)
_QUANTITY_CUE_RE = re.compile(
    r"\b(\d+|uno|dos|tres|cuatro|cinco|minutos|horas|dias|veces)\b"
)


@dataclass(frozen=True)
class _Candidate:
    """A single extractive candidate taken from one evidence chunk."""

    text: str
    coverage: float
    score: float
    evidence_index: int
    order: int


def _normalize(text: str) -> str:
    """Fold case and accents, and collapse whitespace."""
    return " ".join(text.casefold().translate(_ACCENTS).split())


def _tokenize(text: str) -> list[str]:
    """Return lowercase alphanumeric tokens from text."""
    return _TOKEN_RE.findall(_normalize(text))


def _stem(token: str) -> str:
    """Apply a small, deterministic Spanish-oriented suffix reduction."""
    if len(token) <= 3:
        return token

    for suffix in _SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break

    if len(token) > 3 and token[-1] in "aeo":
        token = token[:-1]

    return token.replace("ue", "o").replace("ie", "e")


def _tokens_match(left: str, right: str) -> bool:
    """Match equal stems or sufficiently long shared prefixes."""
    if left == right:
        return True
    return (
        len(left) >= 5
        and len(right) >= 5
        and (left.startswith(right) or right.startswith(left))
    )


# Pronominal constructions such as "se ve afectado" / "se ve afectada" are
# grammatical copulas, not topical content. They are normalized locally so the
# pronominal tokens do not become spurious requirements in the coverage
# denominator. Negations ("no ...") and all other content tokens are preserved.
_PRONOUNIAL_AFFECT_RE = re.compile(
    r"\bse\s+ve\s+afectad[oa]s?\b",
    re.IGNORECASE,
)
_PRONOUNIAL_AFFECT_NEGATED_RE = re.compile(
    r"\b(?:no|nunca|jamas|tampoco)\s+se\s+ve\s+afectad[oa]s?\b",
    re.IGNORECASE,
)


def _question_tokens(question: str) -> tuple[str, ...]:
    """Extract unique topic stems, excluding generic question framing.

    Frame terms such as ``síntomas``, ``pasos`` or ``indican`` describe the
    expected answer shape, not the IT topic. Removing them from the coverage
    denominator keeps the 0.75 requirement focused on topic-specific terms.

    The pronominal affectation construction (``se ve afectado/afectada``) is
    removed before tokenizing so it does not add lexical requirements foreign
    to the content. When the construction is negated (``no se ve afectado``,
    ``nunca se ve afectado``, ``tampoco se ve afectado``), it is kept intact so
    the answer cannot fall into an affirmative branch. Other uses of ``ve`` are
    untouched.
    """
    normalized = _normalize(question)
    if _PRONOUNIAL_AFFECT_NEGATED_RE.search(normalized):
        # Negated affectation: keep the construction and its negation intact
        # so the answer cannot fall into an affirmative branch.
        cleaned = normalized
    else:
        cleaned = _PRONOUNIAL_AFFECT_RE.sub(" ", normalized)
    tokens = []
    for token in _TOKEN_RE.findall(cleaned):
        if len(token) < 2 or token in _STOPWORDS:
            continue
        if token in _GENERIC_QUESTION_FRAME_WORDS:
            continue
        tokens.append(_stem(token))
    return tuple(dict.fromkeys(tokens))


def _contains_injection(text: str) -> bool:
    """Detect obvious instruction-override patterns after normalization."""
    normalized = _normalize(text)
    return any(pattern.search(normalized) for pattern in _INJECTION_PATTERNS)


def _split_candidates(text: str):
    """Yield exact contiguous candidate excerpts from one evidence chunk."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or _HEADING_RE.match(stripped):
            continue

        if _LIST_MARKER_RE.match(stripped):
            yield stripped, True
            continue

        start = 0
        for match in _SENTENCE_END_RE.finditer(stripped):
            piece = stripped[start : match.end()].strip()
            if piece:
                yield piece, False
            start = match.end()

        tail = stripped[start:].strip()
        if tail:
            yield tail, False


def _detect_intent(question: str) -> str:
    """Classify the question into a small set of deterministic intents."""
    normalized = _normalize(question).strip(" ¿?¡!.")

    if normalized.startswith(
        (
            "que es ",
            "que son ",
            "que significa",
            "define ",
            "definicion de",
            "a que se refiere",
            "que se entiende por",
        )
    ):
        return "definition"

    if normalized.startswith(
        (
            "puedo ",
            "no puedo ",
            "se puede ",
            "no se puede ",
            "debo ",
            "no debo ",
            "deberia ",
            "no deberia ",
            "es necesario",
            "no es necesario",
            "hay que",
            "no hay que",
            "es posible",
            "no es posible",
            "tengo que",
            "no tengo que",
            "se debe ",
            "es obligatorio",
            "esta permitido",
        )
    ):
        return "yes_no"

    if (
        "procedimiento" in normalized
        or "procedimientos" in normalized
        or "pasos" in normalized
        or "instrucciones" in normalized
    ):
        return "procedural"

    if normalized.startswith(
        (
            "como ",
            "de que manera",
            "de que forma",
            "que debo",
            "que hacer",
            "que revisar",
            "que comprobar",
            "que pasos",
            "cuales son los pasos",
        )
    ):
        return "procedural"

    if normalized.startswith("por que"):
        return "reason"
    if normalized.startswith("cuando"):
        return "time"
    if normalized.startswith("donde"):
        return "location"
    if normalized.startswith(("quien", "quienes")):
        return "person"
    if normalized.startswith(("cuanto", "cuanta", "cuantos", "cuantas")):
        return "quantity"
    if normalized.startswith(("cual ", "cuales ")):
        return "selection"

    return "general"


def _looks_procedural(text: str, numbered: bool) -> bool:
    """Check for procedural structure or an instructional verb.

    A bare infinitive is only instructional when it is not the object of a
    preposition or nominalizer (``para reiniciar``, ``de reiniciar``). An
    action verb preceded by a subject or the reflexive ``se`` is descriptive
    (``el firewall se configura``), not an instruction. This prevents a
    sentence that merely mentions an action, such as "documented in another
    procedure for restarting the server", from satisfying a procedural
    question.
    """
    if numbered:
        return True

    tokens = _tokenize(text)
    for index, token in enumerate(tokens):
        previous = tokens[index - 1] if index else ""

        if token in _ACTION_VERBS:
            if previous not in _ACTION_VERB_SUBJECT_BLOCKERS:
                return True
            continue

        if len(token) >= 4 and token.endswith(("ar", "er", "ir")):
            if previous not in _INFINITIVE_BLOCKERS:
                return True

    return False


def _candidate_satisfies_intent(intent: str, text: str, numbered: bool) -> bool:
    """Apply the conservative intent-specific acceptance rule."""
    normalized = _normalize(text)

    if intent == "definition":
        return bool(_DEFINITION_CUE_RE.search(normalized))
    if intent == "yes_no":
        return bool(_YES_NO_CUE_RE.search(normalized))
    if intent == "procedural":
        return _looks_procedural(text, numbered)
    if intent == "reason":
        return bool(_REASON_CUE_RE.search(normalized))
    if intent == "time":
        return bool(_TIME_CUE_RE.search(normalized))
    if intent == "location":
        return bool(_LOCATION_CUE_RE.search(normalized))
    if intent == "person":
        return bool(_PERSON_CUE_RE.search(normalized))
    if intent == "quantity":
        return bool(_QUANTITY_CUE_RE.search(normalized))

    return True


def _candidate_tokens(text: str) -> frozenset[str]:
    """Return the set of stems present in a candidate."""
    return frozenset(_stem(token) for token in _tokenize(text))


def select_documentary_answer(
    question: str,
    accepted_evidence: list[dict],
) -> str | None:
    """Select one exact documentary excerpt that answers the question.

    Args:
        question: User question.
        accepted_evidence: Evidence chunks already accepted by the caller.
            Each item must be a dict with a non-empty string ``text`` field.

    Returns:
        An exact contiguous excerpt from one evidence chunk, or ``None`` when
        the question is malformed, unsupported, ambiguous, or adversarial.

    The function never calls a model, never queries a vector store, and never
    combines excerpts from different chunks or sentences.
    """
    if not isinstance(question, str) or not question.strip():
        return None
    if not isinstance(accepted_evidence, list) or not accepted_evidence:
        return None
    if _contains_injection(question):
        return None

    for item in accepted_evidence:
        if not isinstance(item, dict):
            return None
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            return None
        if _contains_injection(text):
            return None

    question_tokens = _question_tokens(question)
    if not question_tokens:
        return None

    # Document-level relevance guard: every question topic token must appear
    # somewhere in the accepted evidence. If any required concept is entirely
    # absent from all accepted chunks, the question cannot be answered.
    all_evidence_tokens: set[str] = set()
    for item in accepted_evidence:
        all_evidence_tokens |= _candidate_tokens(item["text"])

    for token in question_tokens:
        if token in _GENERIC_CONCEPT_TOKENS:
            continue
        if not any(_tokens_match(token, et) for et in all_evidence_tokens):
            return None

    intent = _detect_intent(question)
    candidates: list[_Candidate] = []
    seen: set[str] = set()

    for evidence_index, item in enumerate(accepted_evidence):
        evidence_text = item["text"]
        for order, (candidate_text, numbered) in enumerate(
            _split_candidates(evidence_text)
        ):
            if candidate_text not in evidence_text:
                continue

            key = _normalize(candidate_text)
            if not key or key in seen:
                continue
            seen.add(key)

            candidate_tokens = _candidate_tokens(candidate_text)
            matched_count = sum(
                1
                for token in question_tokens
                if any(
                    _tokens_match(token, candidate_token)
                    for candidate_token in candidate_tokens
                )
            )

            if matched_count < min(2, len(question_tokens)):
                continue

            coverage = matched_count / len(question_tokens)
            if coverage < _MIN_COVERAGE:
                continue

            if not _candidate_satisfies_intent(intent, candidate_text, numbered):
                continue

            score = coverage
            if intent == "procedural" and numbered:
                score += 0.03

            candidates.append(
                _Candidate(
                    text=candidate_text,
                    coverage=coverage,
                    score=score,
                    evidence_index=evidence_index,
                    order=order,
                )
            )

    if not candidates:
        return None

    candidates.sort(
        key=lambda candidate: (
            candidate.coverage,
            candidate.score,
            len(candidate.text),
            -candidate.order,
        ),
        reverse=True,
    )

    top = candidates[0]
    top_normalized = _normalize(top.text)

    for other in candidates[1:]:
        other_normalized = _normalize(other.text)
        if (
            other_normalized in top_normalized
            or top_normalized in other_normalized
        ):
            continue
        if abs(top.coverage - other.coverage) <= _AMBIGUITY_EPSILON:
            return None
        break

    return top.text
