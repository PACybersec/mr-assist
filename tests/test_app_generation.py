"""AppTest tests for the extractive default answer path.

This file previously asserted that LLM generation was the default Streamlit
path. The default is now deterministic extractive selection; LLM service
coverage remains in tests/test_llm.py.
"""

from pathlib import Path
from types import SimpleNamespace

import pytest
from streamlit.testing.v1 import AppTest

from app.core.config import DEFAULT_MAX_DISTANCE
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
    monkeypatch.setattr(
        grounding_module,
        "validate_answer",
        lambda question, answer, evidence: SimpleNamespace(
            supported=True,
            reason="supported",
            matched_text=answer,
        ),
    )

    return AppTest.from_file(APP_PATH, default_timeout=30)


def _accepted_matches(monkeypatch, matches: list) -> None:
    monkeypatch.setattr(kb, "search_knowledge", lambda *args, **kwargs: list(matches))
    monkeypatch.setattr(
        kb,
        "filter_relevant_matches",
        lambda found, max_distance=None: [
            item
            for item in found
            if item.get("distance", float("inf"))
            <= (
                max_distance
                if max_distance is not None
                else DEFAULT_MAX_DISTANCE
            )
        ],
    )


def _fake_generate(monkeypatch, response=None) -> list:
    calls: list = []

    def _generate(question: str, evidence: list) -> str:
        calls.append((question, list(evidence)))
        return response

    monkeypatch.setattr(llm_module, "generate_answer", _generate)
    return calls


def _fake_validate(
    monkeypatch,
    supported: bool = True,
    matched_text: str = "Respuesta generada.",
) -> list:
    calls: list = []

    def _validate(question: str, answer: str, evidence: list) -> SimpleNamespace:
        calls.append((question, answer, list(evidence)))
        if not supported:
            return SimpleNamespace(
                supported=False,
                reason="answer_not_in_evidence",
                matched_text=None,
            )
        return SimpleNamespace(supported=True, reason="supported", matched_text=matched_text)

    monkeypatch.setattr(grounding_module, "validate_answer", _validate)
    return calls


def _markdown_values(app: AppTest) -> list:
    return [element.value for element in app.markdown]


def test_accepted_evidence_invokes_extractive_selection(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)

    select_calls: list = []

    def _select(question: str, evidence: list) -> str:
        select_calls.append((question, list(evidence)))
        return "El servidor está caído."

    monkeypatch.setattr(extractive_module, "select_documentary_answer", _select)
    validate_calls = _fake_validate(
        monkeypatch,
        supported=True,
        matched_text="El servidor está caído.",
    )

    app.run()
    app.chat_input[0].set_value("¿El servidor está caído?")
    app.run()

    assert len(select_calls) == 1
    assert select_calls[0][0] == "¿El servidor está caído?"
    assert select_calls[0][1] == matches
    assert len(validate_calls) == 1
    assert validate_calls[0][1] == "El servidor está caído."
    assert validate_calls[0][2] == matches
    assert "El servidor está caído." in _markdown_values(app)
    assert len(app.exception) == 0


def test_rejected_evidence_never_reaches_llm(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "Procedimiento no relacionado.",
            "source": "procedimientos_it.md",
            "distance": 0.9,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    calls = _fake_generate(monkeypatch, response="Respuesta generada.")

    app.run()
    app.chat_input[0].set_value("¿Qué red?")
    app.run()

    assert calls == []
    assert ABSTENTION_MESSAGE in _markdown_values(app)


def test_no_accepted_evidence_skips_generation(app: AppTest, monkeypatch):
    calls = _fake_generate(monkeypatch, response="Respuesta generada.")

    app.run()
    app.chat_input[0].set_value("¿Qué red?")
    app.run()

    assert calls == []
    assert ABSTENTION_MESSAGE in _markdown_values(app)


def test_retrieval_exception_displays_safe_abstention(app: AppTest, monkeypatch):
    def _raise(*args, **kwargs):
        raise RuntimeError("fallo simulado de recuperación")

    monkeypatch.setattr(kb, "search_knowledge", _raise)

    app.run()
    app.chat_input[0].set_value("¿El servidor está caído?")
    app.run()

    markdown_values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in markdown_values
    assert "fallo simulado de recuperación" not in "\n".join(markdown_values)
    assert len(app.exception) == 0
    assert len(app.session_state["messages"]) == 2


def test_filter_exception_displays_safe_abstention(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    monkeypatch.setattr(kb, "search_knowledge", lambda *args, **kwargs: list(matches))

    def _raise(*args, **kwargs):
        raise RuntimeError("fallo simulado del filtro")

    monkeypatch.setattr(kb, "filter_relevant_matches", _raise)

    app.run()
    app.chat_input[0].set_value("¿El servidor está caído?")
    app.run()

    markdown_values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in markdown_values
    assert "fallo simulado del filtro" not in "\n".join(markdown_values)
    assert len(app.exception) == 0
    assert len(app.session_state["messages"]) == 2


def test_supported_answer_displays_only_matched_text(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_generate(monkeypatch, response="Respuesta generada.")
    validate_calls = _fake_validate(
        monkeypatch,
        supported=True,
        matched_text="El servidor está caído.",
    )

    app.run()
    app.chat_input[0].set_value("¿El servidor está caído?")
    app.run()

    markdown_values = _markdown_values(app)
    assert "El servidor está caído." in markdown_values
    assert "Respuesta generada." not in markdown_values
    assert len(validate_calls) == 1

    messages = app.session_state["messages"]
    assert messages[1]["content"] == "El servidor está caído."
    assert len(app.exception) == 0


def test_supported_answer_with_fabricated_text_displays_abstention(
    app: AppTest,
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
    _fake_generate(monkeypatch, response="El servidor está caído. Ejecutar reinicio manual.")
    _fake_validate(monkeypatch, supported=False)

    app.run()
    app.chat_input[0].set_value("¿El servidor está caído?")
    app.run()

    markdown_values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in markdown_values
    assert "Ejecutar reinicio manual" not in markdown_values
    assert len(app.exception) == 0


def test_negation_contradiction_displays_abstention(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "No se debe reiniciar el servidor.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_generate(monkeypatch, response="El servidor debe reiniciarse.")
    _fake_validate(monkeypatch, supported=False)

    app.run()
    app.chat_input[0].set_value("¿Se debe reiniciar el servidor?")
    app.run()

    markdown_values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in markdown_values
    assert len(app.exception) == 0


def test_invented_shell_command_displays_abstention(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "sudo systemctl restart nginx",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_generate(monkeypatch, response="sudo systemctl restart nginx --force")
    _fake_validate(monkeypatch, supported=False)

    app.run()
    app.chat_input[0].set_value("¿Cómo reiniciar nginx?")
    app.run()

    markdown_values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in markdown_values
    assert "sudo systemctl restart nginx --force" not in markdown_values
    assert len(app.exception) == 0


def test_no_accepted_evidence_skips_generation_and_validation(
    app: AppTest,
    monkeypatch,
):
    calls = _fake_generate(monkeypatch, response="Respuesta generada.")
    validate_calls = _fake_validate(
        monkeypatch,
        supported=True,
        matched_text="Respuesta generada.",
    )

    app.run()
    app.chat_input[0].set_value("¿Qué red?")
    app.run()

    assert len(calls) == 0
    assert len(validate_calls) == 0
    assert ABSTENTION_MESSAGE in _markdown_values(app)
    assert len(app.exception) == 0


def test_validation_exception_displays_safe_abstention(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_generate(monkeypatch, response="Respuesta generada.")

    def _raise(question: str, answer: str, evidence: list) -> SimpleNamespace:
        raise RuntimeError("fallo simulado del validador")

    monkeypatch.setattr(grounding_module, "validate_answer", _raise)

    app.run()
    app.chat_input[0].set_value("¿El servidor está caído?")
    app.run()

    markdown_values = _markdown_values(app)
    assert ABSTENTION_MESSAGE in markdown_values
    assert len(app.exception) == 0
    assert len(app.session_state["messages"]) == 2


def test_supported_answer_contains_no_internal_data(app: AppTest, monkeypatch):
    matches = [
        {
            "text": "El servidor está caído.",
            "source": "procedimientos_it.md",
            "distance": 0.2,
        }
    ]
    _accepted_matches(monkeypatch, matches)
    _fake_generate(monkeypatch, response="Respuesta generada.")
    _fake_validate(
        monkeypatch,
        supported=True,
        matched_text="El servidor está caído.",
    )

    app.run()
    app.chat_input[0].set_value("¿El servidor está caído?")
    app.run()

    markdown_values = _markdown_values(app)
    rendered = "\n".join(markdown_values)
    assert ".md" not in rendered
    assert "distance" not in rendered
    assert "/" not in rendered
    assert len(app.exception) == 0
