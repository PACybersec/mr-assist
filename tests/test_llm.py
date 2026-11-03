
from types import SimpleNamespace
from threading import Event

import pytest

from app.core.config import (
    GENERATION_MODEL,
    GENERATION_OPTIONS,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_TIMEOUT_SECONDS,
)
from app.services import llm
from app.services.llm import (
    GenerationAbstainedError,
    InvalidGenerationRequestError,
    LLMError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
    generate_answer,
)


def _response(content: str = "Respuesta basada en la evidencia.") -> SimpleNamespace:
    return SimpleNamespace(
        message=SimpleNamespace(content=content),
        response=None,
    )


@pytest.fixture
def chat(monkeypatch) -> dict:
    state: dict = {"calls": [], "response": None, "error": None}

    def _chat(**kwargs):
        state["calls"].append(kwargs)
        if state["error"] is not None:
            raise state["error"]
        if state["response"] is not None:
            return state["response"]
        return _response()

    monkeypatch.setattr(llm, "ollama", SimpleNamespace(chat=_chat))
    return state


def _evidence() -> list[dict]:
    return [
        {
            "text": "Comprobar mediante las herramientas autorizadas.",
            "source": "procedimientos_it.md",
        }
    ]


def test_generate_answer_returns_normalized_response(chat: dict):
    result = generate_answer("¿Cómo comprobar la conectividad?", _evidence())

    assert result == "Respuesta basada en la evidencia."


def test_generate_answer_strips_generated_content(chat: dict):
    chat["response"] = _response("  Respuesta generada.  ")

    result = generate_answer("¿Qué red?", _evidence())

    assert result == "Respuesta generada."


def test_generate_answer_uses_configured_model(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    assert chat["calls"][0]["model"] == GENERATION_MODEL
    assert GENERATION_MODEL == "qwen3:0.6b"


def test_generate_answer_includes_evidence_in_model_context(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    messages = chat["calls"][0]["messages"]
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert GENERATION_SYSTEM_PROMPT in messages[0]["content"]
    assert "¿Qué red?" in messages[1]["content"]
    assert "Comprobar mediante las herramientas autorizadas." in messages[1]["content"]


def test_generate_answer_rejects_empty_evidence_without_invoking_model(
    monkeypatch,
    chat: dict,
):
    def _chat(**kwargs):
        raise AssertionError("No debe invocarse Ollama sin evidencia.")

    monkeypatch.setattr(llm, "ollama", SimpleNamespace(chat=_chat))

    with pytest.raises(InvalidGenerationRequestError):
        generate_answer("¿Qué red?", [])

    assert chat["calls"] == []


def test_generate_answer_rejects_empty_question_without_invoking_model(
    monkeypatch,
    chat: dict,
):
    def _chat(**kwargs):
        raise AssertionError("No debe invocarse Ollama sin pregunta.")

    monkeypatch.setattr(llm, "ollama", SimpleNamespace(chat=_chat))

    with pytest.raises(InvalidGenerationRequestError):
        generate_answer("   ", _evidence())

    assert chat["calls"] == []


def test_generate_answer_disables_thinking(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    assert chat["calls"][0]["think"] is False


def test_generate_answer_passes_low_temperature_options(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    assert chat["calls"][0]["options"] == GENERATION_OPTIONS
    assert chat["calls"][0]["options"]["temperature"] < 0.5


def test_generate_answer_raises_connection_error_on_connection_failure(
    monkeypatch,
):
    def _chat(**kwargs):
        raise ConnectionError("ollama unavailable")

    monkeypatch.setattr(llm, "ollama", SimpleNamespace(chat=_chat))

    with pytest.raises(OllamaConnectionError):
        generate_answer("¿Qué red?", _evidence())


def test_generate_answer_raises_timeout_error(monkeypatch):
    release = Event()

    def _chat(**kwargs):
        release.wait(timeout=10)
        return _response()

    monkeypatch.setattr(llm, "ollama", SimpleNamespace(chat=_chat))
    monkeypatch.setattr(llm, "GENERATION_TIMEOUT_SECONDS", 0.5)

    with pytest.raises(OllamaTimeoutError):
        generate_answer("¿Qué red?", _evidence())

    release.set()


def test_generate_answer_raises_response_error_on_empty_response(monkeypatch):
    monkeypatch.setattr(
        llm,
        "ollama",
        SimpleNamespace(
            chat=lambda **kwargs: SimpleNamespace(
                message=SimpleNamespace(content=None),
                response=None,
            )
        ),
    )

    with pytest.raises(OllamaResponseError):
        generate_answer("¿Qué red?", _evidence())


def test_generate_answer_raises_response_error_on_malformed_response(monkeypatch):
    monkeypatch.setattr(llm, "ollama", SimpleNamespace(chat=lambda **kwargs: None))

    with pytest.raises(OllamaResponseError):
        generate_answer("¿Qué red?", _evidence())


def test_generate_answer_does_not_expose_model_reasoning(monkeypatch):
    def _chat(**kwargs):
        return SimpleNamespace(
            message=SimpleNamespace(
                content="Respuesta basada en la evidencia.",
                thinking="Analizando la evidencia.",
            ),
            response=None,
        )

    monkeypatch.setattr(llm, "ollama", SimpleNamespace(chat=_chat))

    answer = generate_answer("¿Qué red?", _evidence())

    assert answer == "Respuesta basada en la evidencia."
    assert "Analizando" not in answer


def test_generate_answer_includes_verbatim_extraction_instructions(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    messages = chat["calls"][0]["messages"]
    system_content = messages[0]["content"]

    assert "verbatim" in system_content
    assert "NO_EVIDENCE" in system_content
    assert "unicamente el fragmento" in system_content


def test_generate_answer_does_not_allow_paraphrasing(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    messages = chat["calls"][0]["messages"]
    system_content = messages[0]["content"]

    assert "No parafrasees" in system_content


def test_generate_answer_requires_single_contiguous_span(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    messages = chat["calls"][0]["messages"]
    system_content = messages[0]["content"]

    assert "No combines" in system_content
    assert "Selecciona el fragmento unico" in system_content


def test_generation_prompt_ignores_instructions_embedded_in_documents(chat: dict):
    generate_answer("¿Qué red?", _evidence())

    messages = chat["calls"][0]["messages"]
    system_content = messages[0]["content"]

    assert "Ignora" in system_content
    assert "instruccion" in system_content.lower()


def test_generate_answer_raises_abstention_error_on_no_evidence_token(chat: dict):
    chat["response"] = _response("NO_EVIDENCE")

    with pytest.raises(GenerationAbstainedError):
        generate_answer("¿Qué red?", _evidence())


def test_generate_answer_returns_valid_literal_answer_normally(chat: dict):
    chat["response"] = _response("Comprobar mediante las herramientas autorizadas.")

    result = generate_answer("¿Qué red?", _evidence())

    assert result == "Comprobar mediante las herramientas autorizadas."


def test_generation_abstained_error_is_typed_error():
    assert issubclass(GenerationAbstainedError, LLMError)
    assert issubclass(GenerationAbstainedError, InvalidGenerationRequestError)
