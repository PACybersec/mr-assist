
"""Servicio local de generación de respuestas con Ollama."""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import ollama
from ollama import ResponseError

from app.core.config import (
    GENERATION_ABSTENTION_TOKEN,
    GENERATION_MODEL,
    GENERATION_OPTIONS,
    GENERATION_SYSTEM_PROMPT,
    GENERATION_TIMEOUT_SECONDS,
)


class LLMError(Exception):
    """Error base del servicio de generación local."""


class OllamaConnectionError(LLMError):
    """Error de conexión con el servicio Ollama."""


class OllamaTimeoutError(LLMError):
    """Error de timeout durante la generación local."""


class OllamaResponseError(LLMError):
    """Error en la respuesta devuelta por el modelo local."""


class InvalidGenerationRequestError(ValueError):
    """Error en la solicitud de generación: pregunta o evidencia inválidos."""


class GenerationAbstainedError(LLMError, InvalidGenerationRequestError):
    """Error en el resultado de generación: la documentación no contenía respuesta."""


def _build_user_content(question: str, evidence: list[dict]) -> str:
    """Construye el contenido del mensaje usuario con la pregunta y la evidencia."""
    evidence_blocks = []

    for item in evidence:
        text = item.get("text", "") if isinstance(item, dict) else str(item)
        evidence_blocks.append(f"<document>\n{text}\n</document>")

    content = f"Pregunta: {question}\n\nDocumentación recuperada:\n"

    if evidence_blocks:
        content += "\n\n".join(evidence_blocks)

    return content


def _build_messages(question: str, evidence: list[dict]) -> list[dict]:
    """Construye los mensajes separando instrucciones y evidencia no confiable."""
    return [
        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
        {"role": "user", "content": _build_user_content(question, evidence)},
    ]


def _map_generation_error(error: Exception) -> LLMError:
    """Mapea los errores del cliente Ollama a errores tipados del servicio."""
    if isinstance(error, (LLMError, InvalidGenerationRequestError)):
        return error

    if isinstance(error, TimeoutError) or isinstance(error, httpx.TimeoutException):
        return OllamaTimeoutError(
            "La generación de la respuesta excedió el tiempo configurado."
        )

    if isinstance(error, ResponseError):
        return OllamaResponseError(
            "El modelo de generación devolvió una respuesta inválida."
        )

    if isinstance(error, ConnectionError):
        return OllamaConnectionError(
            "No se pudo completar la generación de la respuesta."
        )

    return LLMError(f"Error durante la generación local: {type(error).__name__}")


def _request_generation(messages: list[dict]) -> Any:
    """Envía la generación al modelo local mediante Ollama."""
    try:
        return ollama.chat(
            model=GENERATION_MODEL,
            messages=messages,
            stream=False,
            think=False,
            options=GENERATION_OPTIONS,
        )
    except Exception as error:
        raise _map_generation_error(error) from error


def _normalize_response(response: Any) -> str:
    """Normaliza la respuesta del modelo y descarta contenido vacío."""
    content = ""

    message = getattr(response, "message", None)
    if message is not None:
        content = getattr(message, "content", None) or ""

    if not content:
        content = getattr(response, "response", None) or ""

    answer = content.strip() if isinstance(content, str) else ""

    if not answer:
        raise OllamaResponseError("El modelo de generación no devolvió contenido.")

    return answer


def generate_answer(question: str, evidence: list[dict]) -> str:
    """Genera una respuesta local a partir de la evidencia recibida.

    Args:
        question: Pregunta del usuario.
        evidence: Fragmentos de documentación recuperados y aceptados.

    Returns:
        Texto de respuesta normalizado.

    Raises:
        InvalidGenerationRequestError: Si la pregunta o la evidencia son inválidas.
        OllamaConnectionError: Si no se puede conectar con Ollama.
        OllamaTimeoutError: Si la generación excede el tiempo configurado.
        OllamaResponseError: Si la respuesta del modelo es inválida o vacía.
        LLMError: Error base ante fallos de generación.
    """
    question = (question or "").strip()
    if not question:
        raise InvalidGenerationRequestError("La pregunta no puede estar vacía.")

    if not evidence:
        raise InvalidGenerationRequestError(
            "No se recibió evidencia para generar la respuesta."
        )

    messages = _build_messages(question, evidence)

    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_request_generation, messages)
        try:
            response = future.result(timeout=GENERATION_TIMEOUT_SECONDS)
        except TimeoutError as error:
            raise OllamaTimeoutError(
                "La generación de la respuesta excedió el tiempo configurado."
            ) from error
    finally:
        executor.shutdown(wait=False, cancel_futures=True)

    answer = _normalize_response(response)

    if answer == GENERATION_ABSTENTION_TOKEN:
        raise GenerationAbstainedError(
            "La documentacion no contiene una respuesta para la pregunta."
        )

    return answer
