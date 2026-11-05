"""AppTest integration tests for the deterministic extractive default flow."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from app.services import extractive_answering as extractive_module
from app.services import grounding_validator as grounding_module
from app.services import knowledge_base as kb
from app.services import llm as llm_module


APP_PATH = str(Path(__file__).resolve().parents[1] / "app" / "main.py")

ABSTENTION_MESSAGE = (
    "No he encontrado evidencia documental "
    "suficientemente relevante para responder "
    "a tu consulta.\n\n"
    "No puedo proporcionar instrucciones "
    "sin respaldo documental. Consulta al "
    "equipo responsable si necesita "
    "información adicional."
)


@pytest.fixture
def app(monkeypatch) -> AppTest:
    monkeypatch.setattr(kb, "collection", SimpleNamespace(count=lambda: 0))
    monkeypatch.setattr(kb, "search_knowledge", lambda *args, **kwargs: [])
    monkeypatch.setattr(kb, "filter_relevant_matches", lambda *args, **kwargs: [])
    return AppTest.from_file(APP_PATH, default_timeout=30)


def _accepted_matches(monkeypatch, matches: list) -> None:
    monkeypatch.setattr(kb, "search_knowledge", lambda *args, **kwargs: list(matches))
    monkeypatch.setattr(
        kb,
        "filter_relevant_matches",
        lambda found, max_distance=None: list(matches),
    )


def _fake_select(monkeypatch, candidate=None, error=None) -> list:
    calls: list = []

    def _select(question: str, evidence: list):
        calls.append((question, list(evidence)))
        if error is not None:
            raise error
        return candidate

    monkeypatch.setattr(extractive_module, "select_documentary_answer", _select)
    return calls


def _fake_validate(
    monkeypatch,
    supported: bool = True,
    matched_text: str = "Texto validado.",
    error=None,
) -> list:
    calls: list = []

    def _validate(question: str, answer: str, evidence: list):
        calls.append((question, answer, list(evidence)))
        if error is not None:
            raise error
        if not supported:
            return SimpleNamespace(
                supported=False,
                reason="answer_not_in_evidence",
                matched_text=None,
            )
        return SimpleNamespace(
            supported=True,
            reason="supported",
            matched_text=matched_text,
        )

    monkeypatch.setattr(grounding_module, "validate_answer", _validate)
    return calls


def _fake_generate(monkeypatch) -> list:
    calls: list = []

    def _generate(question: str, evidence: list) -> str:
        calls.append((question, list(evidence)))
        return "Respuesta generada por LLM."

    monkeypatch.setattr(llm_module, "generate_answer", _generate)
    return calls


def _markdown_values(app: AppTest) -> list:
    return [element.value for element in app.markdown]


def _ask(app: AppTest, question: str) -> None:
    app.chat_input[0].set_value(question)
    app.run()


def _assert_no_internal_metadata(app: AppTest) -> None:
    rendered = "\n".join(_markdown_values(app))

    assert ".md" not in rendered
    assert "distance" not in rendered
    assert "Distancia" not in rendered
    assert "Fuente" not in rendered
    assert "procedimientos_it" not in rendered
    assert "chroma" not in rendered.lower()
    assert "/" not in rendered

    assert len(app.expander) == 0

    for caption in app.caption:
        value = caption.value
        assert "Fuente" not in value
        assert "Distancia" not in value
        assert ".md" not in value


def test_supported_factual_question_displays_validated_text(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    app.run()
    _ask(app, "¿El servidor está caído?")

    values = _markdown_values(app)
    assert "El servidor está caído." in values
    assert ABSTENTION_MESSAGE not in values
    assert app.session_state["messages"][1]["content"] == "El servidor está caído."
    assert len(app.exception) == 0


def test_supported_procedural_question_displays_validated_text(app, monkeypatch):
    evidence_text = (
        "1. Comprobar la conectividad con el servidor mediante las "
        "herramientas autorizadas."
    )
    matches = [
        {
            "text": evidence_text,
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    app.run()
    _ask(app, "¿Cómo comprobar la conectividad?")

    values = _markdown_values(app)
    assert evidence_text in values
    assert ABSTENTION_MESSAGE not in values
    assert len(app.exception) == 0


def test_supported_prohibition_preserves_negation(app, monkeypatch):
    evidence_text = (
        "No desactivar el firewall ni cambiar reglas de acceso para "
        "solucionar el problema sin autorización."
    )
    matches = [
        {
            "text": evidence_text,
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    app.run()
    _ask(app, "¿Puedo desactivar el firewall?")

    values = _markdown_values(app)
    assert evidence_text in values
    assert evidence_text.startswith("No ")
    assert ABSTENTION_MESSAGE not in values
    assert len(app.exception) == 0


def test_unsupported_question_abstains(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    app.run()
    _ask(app, "¿Cómo restablezco la contraseña del usuario?")

    values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in values
    assert "El servidor está caído." not in values
    assert len(app.exception) == 0


def test_no_accepted_evidence_abstains_without_selection(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    monkeypatch.setattr(kb, "search_knowledge", lambda *args, **kwargs: list(matches))
    monkeypatch.setattr(kb, "filter_relevant_matches", lambda *args, **kwargs: [])

    select_calls = _fake_select(monkeypatch, candidate="No debería usarse.")
    validate_calls = _fake_validate(
        monkeypatch,
        supported=True,
        matched_text="No debería mostrarse.",
    )

    app.run()
    _ask(app, "¿Qué red?")

    values = _markdown_values(app)
    assert select_calls == []
    assert validate_calls == []
    assert ABSTENTION_MESSAGE in values
    assert "No debería mostrarse." not in values
    assert len(app.exception) == 0


def test_selector_exception_abstains_safely(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_select(monkeypatch, error=RuntimeError("fallo simulado del selector"))

    app.run()
    _ask(app, "¿El servidor está caído?")

    values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in values
    assert "fallo simulado del selector" not in "\n".join(values)
    assert len(app.exception) == 0
    assert len(app.session_state["messages"]) == 2


def test_validator_rejection_abstains_safely(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_select(monkeypatch, candidate="El servidor está caído.")
    _fake_validate(monkeypatch, supported=False)

    app.run()
    _ask(app, "¿El servidor está caído?")

    values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in values
    assert "El servidor está caído." not in values
    assert len(app.exception) == 0


def test_validator_missing_matched_text_abstains_safely(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_select(monkeypatch, candidate="El servidor está caído.")
    _fake_validate(monkeypatch, supported=True, matched_text=None)

    app.run()
    _ask(app, "¿El servidor está caído?")

    values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in values
    assert "El servidor está caído." not in values
    assert app.session_state["messages"][1]["content"] == ABSTENTION_MESSAGE
    assert len(app.exception) == 0


def test_validator_exception_abstains_safely(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_select(monkeypatch, candidate="El servidor está caído.")
    _fake_validate(
        monkeypatch,
        error=RuntimeError("fallo simulado del validador"),
    )

    app.run()
    _ask(app, "¿El servidor está caído?")

    values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in values
    assert "fallo simulado del validador" not in "\n".join(values)
    assert len(app.exception) == 0
    assert len(app.session_state["messages"]) == 2


def test_exact_matched_text_is_displayed(app, monkeypatch):
    matches = [
        {
            "text": "Evidencia original.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_select(monkeypatch, candidate="Candidato extractivo.")
    _fake_validate(
        monkeypatch,
        supported=True,
        matched_text="Texto original exacto.\nSegunda línea.",
    )

    app.run()
    _ask(app, "¿El servidor está caído?")

    values = _markdown_values(app)
    assert "Texto original exacto.\nSegunda línea." in values
    assert "Candidato extractivo." not in values
    assert (
        app.session_state["messages"][1]["content"]
        == "Texto original exacto.\nSegunda línea."
    )
    assert len(app.exception) == 0


def test_default_flow_does_not_invoke_llm(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    llm_calls = _fake_generate(monkeypatch)

    app.run()
    _ask(app, "¿El servidor está caído?")

    assert llm_calls == []
    assert "El servidor está caído." in _markdown_values(app)
    assert len(app.exception) == 0


def test_chat_history_preserves_only_accepted_answers_or_abstentions(
    app,
    monkeypatch,
):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    app.run()
    _ask(app, "¿El servidor está caído?")
    _ask(app, "¿Cómo restablezco la contraseña del usuario?")

    messages = app.session_state["messages"]
    assert len(messages) == 4
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "El servidor está caído."
    assert messages[2]["role"] == "user"
    assert messages[3]["role"] == "assistant"
    assert messages[3]["content"] == ABSTENTION_MESSAGE
    assert len(app.exception) == 0


def test_normal_answer_contains_no_internal_metadata(app, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    app.run()
    _ask(app, "¿El servidor está caído?")

    _assert_no_internal_metadata(app)
    assert len(app.exception) == 0


def test_normal_answer_displays_only_validated_excerpt(app, monkeypatch):
    long_text = (
        "## Diagnóstico\n"
        "1. Confirmar el alcance del problema.\n"
        "2. Comprobar si el equipo tiene conexión de red.\n"
        "3. Verificar el nombre DNS del servidor.\n"
    )
    excerpt = "2. Comprobar si el equipo tiene conexión de red."
    matches = [
        {
            "text": long_text,
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_select(monkeypatch, candidate=excerpt)
    _fake_validate(monkeypatch, supported=True, matched_text=excerpt)

    app.run()
    _ask(app, "¿Qué comprobar si el equipo no tiene red?")

    values = _markdown_values(app)
    assert excerpt in values
    assert "1. Confirmar el alcance del problema." not in values
    assert "3. Verificar el nombre DNS del servidor." not in values
    assert "## Diagnóstico" not in values
    assert ABSTENTION_MESSAGE not in values
    assert (
        app.session_state["messages"][1]["content"] == excerpt
    )
    assert len(app.exception) == 0


def test_internal_audit_remains_available_in_development_mode(
    app,
    monkeypatch,
):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    app.run()
    app.checkbox[0].set_value(True)
    app.run()
    _ask(app, "¿El servidor está caído?")

    labels = [expander.label for expander in app.expander]
    assert any("Trazabilidad interna" in label for label in labels)

    captions = [caption.value for caption in app.caption]
    assert any("Fuente: procedimientos_it.md" in value for value in captions)
    assert any("Distancia: 0.2" in value for value in captions)
    assert len(app.exception) == 0


def test_sla_question_abstains_with_real_selector(app, monkeypatch):
    """Original SLA false-acceptance case (I03): must abstain end-to-end."""
    matches = [
        {
            "text": "- Si no se encuentra la causa, escalar al equipo responsable adjuntando hora, síntomas y evidencias.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    monkeypatch.setattr(kb, "search_knowledge", lambda *args, **kwargs: list(matches))
    monkeypatch.setattr(kb, "filter_relevant_matches", lambda *args, **kwargs: list(matches))
    # Do NOT mock select_documentary_answer: the real selector with the guard runs.

    app.run()
    _ask(app, "¿Cuál es el SLA de escalado del equipo responsable?")

    values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in values
    assert len(app.exception) == 0
