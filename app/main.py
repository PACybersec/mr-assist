
import streamlit as st

from app.core.config import DEFAULT_MAX_DISTANCE, DEFAULT_N_RESULTS
from app.services.knowledge_base import (
    collection,
    filter_relevant_matches,
    index_documents,
    search_knowledge,
)
from app.services.grounding_validator import validate_answer
from app.services.llm import (
    generate_answer,
    InvalidGenerationRequestError,
    OllamaConnectionError,
    OllamaResponseError,
    OllamaTimeoutError,
)


ABSTENTION_MESSAGE = (
    "No he encontrado evidencia documental "
    "suficientemente relevante para responder "
    "a tu consulta.\n\n"
    "No puedo proporcionar instrucciones "
    "sin respaldo documental. Consulta al "
    "equipo responsable si necesita "
    "información adicional."
)


st.set_page_config(
    page_title="Mr.Assist",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Mr.Assist")
st.subheader("Asistente inteligente de operaciones IT")

st.write(
    "Asistente local con recuperación de información "
    "desde procedimientos IT."
)

st.info(
    "Modo estricto: las respuestas se basan en fragmentos "
    "originales de la documentación. No se generan "
    "instrucciones que no estén respaldadas por ella."
)


# --------------------------------------------------
# 1. PANEL DE ESTADO Y ACTUALIZACIÓN DOCUMENTAL
# --------------------------------------------------

with st.sidebar:
    st.header("Base de conocimiento")

    st.write(
        f"Fragmentos indexados: {collection.count()}"
    )

    if st.button(
        "Actualizar documentos",
        use_container_width=True,
    ):
        try:
            with st.spinner("Indexando documentos..."):
                total = index_documents()

            st.success(
                f"Indexación completada: {total} fragmentos."
            )

            st.rerun()

        except Exception as error:
            st.error(
                "No se pudo actualizar la base de conocimiento."
            )
            st.caption(f"Detalle técnico: {error}")


# --------------------------------------------------
# 2. HISTORIAL VISUAL
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def display_message(message: dict) -> None:
    """Muestra un mensaje y su evidencia documental."""

    st.markdown(message["content"])

    evidence = message.get("evidence", [])

    if evidence:
        with st.expander("Evidencias documentales"):
            for item in evidence:
                st.caption(
                    f"Fuente: {item['source']}"
                )

                # Mostrar el contenido original sin
                # interpretarlo como instrucciones Markdown.
                st.text(item["text"])


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        display_message(message)


# --------------------------------------------------
# 3. RECEPCIÓN DE CONSULTAS
# --------------------------------------------------

question = st.chat_input("Describe tu consulta IT...")


if question:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:
            with st.spinner(
                "Buscando evidencia en la documentación..."
            ):

                # 1. Recuperar fragmentos de ChromaDB.
                matches = search_knowledge(
                    question,
                    n_results=DEFAULT_N_RESULTS,
                )

                # 2. Filtrar por distancia semántica.
                relevant = filter_relevant_matches(
                    matches,
                    max_distance=DEFAULT_MAX_DISTANCE,
                )

                # 3. Preparar respuesta documental.
                if relevant:

                    answer = generate_answer(question, relevant)

                    # TASK 06.3.2 — Validación determinista de anclaje (fail closed).
                    validation = None
                    try:
                        validation = validate_answer(question, answer, relevant)
                    except Exception:
                        validation = None

                    if (
                        validation is not None
                        and validation.supported
                        and validation.matched_text
                    ):
                        # Respuesta verificada: solo se muestra el fragmento
                        # original respaldado por la evidencia aceptada.
                        answer = validation.matched_text
                        evidence = []
                    else:
                        answer = ABSTENTION_MESSAGE
                        evidence = []

                else:

                    answer = ABSTENTION_MESSAGE
                    evidence = []

        except OllamaConnectionError:
            answer = (
                "No se pudo completar la respuesta en este momento. "
                "Comprueba que Ollama está disponible y vuelve a intentarlo."
            )
            evidence = []

        except OllamaTimeoutError:
            answer = (
                "La generación de la respuesta tardó demasiado de tiempo. "
                "Vuelve a intentarlo."
            )
            evidence = []

        except OllamaResponseError:
            answer = (
                "La respuesta del asistente no pudo ser procesada. "
                "Vuelve a intentarlo."
            )
            evidence = []

        except InvalidGenerationRequestError:
            answer = (
                "No se pudo generar una respuesta por falta de información. "
                "Consulta al equipo responsable si necesitas información adicional."
            )
            evidence = []

        except Exception as error:

            st.error(
                "No se pudo completar la consulta. "
                "Comprueba Ollama y la base de conocimiento."
            )

            st.caption(
                f"Detalle técnico: {error}"
            )

            answer = (
                "No se pudo completar la respuesta en este momento. "
                "Comprueba Ollama y la base de conocimiento."
            )
            evidence = []

        # 4. Mostrar respuesta.
        st.markdown(answer)

        if evidence:
            with st.expander("Evidencias documentales"):
                for item in evidence:

                    st.caption(
                        f"Fuente: {item['source']}"
                    )

                    st.text(item["text"])

        # 4b. Trazabilidad interna / auditoría.
        with st.expander(
            "🔍 Trazabilidad interna y auditoría",
            expanded=False,
        ):
            for item in relevant:
                st.caption(
                    f"Fuente: {item['source']}"
                )

                st.caption(
                    f"Distancia: {item['distance']}"
                )

                st.text(item["text"])

            if not relevant:
                st.caption(
                    "Sin fragmentos dentro del umbral de "
                    f"{DEFAULT_MAX_DISTANCE}."
                )

        # 5. Guardar historial visual.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "evidence": evidence,
            }
        )
