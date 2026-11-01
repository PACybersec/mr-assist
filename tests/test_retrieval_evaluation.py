
# Consultas de referencia para evaluar la recuperación de Mr.Assist.

EVALUATION_CASES = [
    {
        "id": "R01",
        "question": "¿Qué debo comprobar si afecta a varios usuarios?",
        "expected_evidence": [
            "red",
            "estado del servicio compartido",
        ],
        "should_find_evidence": True,
    },
    {
        "id": "R02",
        "question": "¿Qué comando reinicia el servidor?",
        "expected_evidence": [],
        "should_find_evidence": False,
        "forbidden_evidence": ["reiniciar el servidor"],
    },
    {
        "id": "R03",
        "question": "¿Cómo comprobar la conectividad?",
        "expected_evidence": [
            "herramientas autorizadas",
        ],
        "should_find_evidence": True,
    },
    {
        "id": "R04",
        "question": "¿Puedo desactivar el firewall?",
        "expected_evidence": [
            "sin autorización",
        ],
        "should_find_evidence": True,
    },
    {
        "id": "R05",
        "question": "¿Qué hacer si no encuentro la causa?",
        "expected_evidence": [
            "escalar al equipo responsable",
        ],
        "should_find_evidence": True,
    },
]


# Validación del conjunto de evaluación

def test_evaluation_cases_have_unique_ids():
    ids = [case["id"] for case in EVALUATION_CASES]
    assert len(ids) == len(set(ids))


def test_evaluation_cases_have_required_fields():
    required_fields = {
        "id",
        "question",
        "expected_evidence",
        "should_find_evidence",
    }

    for case in EVALUATION_CASES:
        assert required_fields.issubset(case.keys())


def test_evaluation_cases_have_valid_expectations():
    for case in EVALUATION_CASES:
        if case["should_find_evidence"]:
            assert case["expected_evidence"]
        else:
            assert case["expected_evidence"] == []


# Evaluador de recuperación

def evaluate_case(case: dict, matches: list[dict]) -> dict:
    """Evalúa la presencia de evidencia en los fragmentos recuperados."""

    retrieved_count = len(matches)
    has_retrieved_context = retrieved_count > 0

    retrieved_text = " ".join(
        match["text"].lower()
        for match in matches
    )

    expected_evidence = case["expected_evidence"]
    forbidden_evidence = case.get("forbidden_evidence") or []

    matched_evidence = [
        evidence
        for evidence in expected_evidence
        if evidence.lower() in retrieved_text
    ]

    evidence_found = (
        bool(expected_evidence)
        and len(matched_evidence) == len(expected_evidence)
    )

    forbidden_evidence_found = any(
        evidence.lower() in retrieved_text
        for evidence in forbidden_evidence
    )

    if case["should_find_evidence"]:
        expectation_met = evidence_found
    else:
        expectation_met = not forbidden_evidence_found

    return {
        "case_id": case["id"],
        "retrieved_count": retrieved_count,
        "has_retrieved_context": has_retrieved_context,
        "evidence_found": evidence_found,
        "matched_evidence": matched_evidence,
        "forbidden_evidence_found": forbidden_evidence_found,
        "expectation_met": expectation_met,
    }


# Pruebas del evaluador

def test_evaluator_accepts_expected_evidence():
    case = EVALUATION_CASES[2]

    matches = [
        {"text": "Comprobar mediante las herramientas autorizadas."}
    ]

    result = evaluate_case(case, matches)

    assert result["retrieved_count"] == 1
    assert result["has_retrieved_context"] is True
    assert result["evidence_found"] is True
    assert result["expectation_met"] is True


def test_evaluator_detects_missing_evidence():
    case = EVALUATION_CASES[2]

    matches = [
        {"text": "Comprobar el estado general del servidor."}
    ]

    result = evaluate_case(case, matches)

    assert result["has_retrieved_context"] is True
    assert result["evidence_found"] is False
    assert result["expectation_met"] is False


def test_evaluator_distinguishes_missing_evidence_from_empty_retrieval():
    case = EVALUATION_CASES[1]

    empty_result = evaluate_case(case, [])

    unrelated_matches = [
        {"text": "El servidor presenta un error de conexión."}
    ]

    unrelated_result = evaluate_case(case, unrelated_matches)

    assert empty_result["retrieved_count"] == 0
    assert empty_result["has_retrieved_context"] is False

    assert unrelated_result["retrieved_count"] == 1
    assert unrelated_result["has_retrieved_context"] is True

    assert empty_result["evidence_found"] is False
    assert unrelated_result["evidence_found"] is False

    assert empty_result["expectation_met"] is True
    assert unrelated_result["expectation_met"] is True


def test_negative_case_with_empty_retrieval_is_satisfied():
    case = EVALUATION_CASES[1]

    result = evaluate_case(case, [])

    assert result["retrieved_count"] == 0
    assert result["has_retrieved_context"] is False
    assert result["evidence_found"] is False
    assert result["forbidden_evidence_found"] is False
    assert result["expectation_met"] is True


def test_negative_case_with_unrelated_context_is_satisfied():
    case = EVALUATION_CASES[1]

    matches = [
        {"text": "El servidor presenta un error de conexión."},
    ]

    result = evaluate_case(case, matches)

    assert result["retrieved_count"] == 1
    assert result["has_retrieved_context"] is True
    assert result["evidence_found"] is False
    assert result["forbidden_evidence_found"] is False
    assert result["expectation_met"] is True


def test_positive_case_still_detects_expected_evidence():
    case = EVALUATION_CASES[0]

    matches = [
        {
            "text": (
                "Si afecta a varios usuarios, comprobar la red "
                "y el estado del servicio compartido."
            ),
        },
    ]

    result = evaluate_case(case, matches)

    assert result["evidence_found"] is True
    assert result["matched_evidence"] == [
        "red",
        "estado del servicio compartido",
    ]
    assert result["expectation_met"] is True


def test_forbidden_evidence_is_detected():
    case = EVALUATION_CASES[1]

    matches = [
        {"text": "Para reiniciar el servidor, ejecute el procedimiento indicado."},
    ]

    result = evaluate_case(case, matches)

    assert result["forbidden_evidence_found"] is True


def test_negative_case_fails_when_forbidden_evidence_is_present():
    case = EVALUATION_CASES[1]

    matches = [
        {"text": "Para reiniciar el servidor, ejecute el procedimiento indicado."},
    ]

    result = evaluate_case(case, matches)

    assert result["forbidden_evidence_found"] is True
    assert result["expectation_met"] is False
