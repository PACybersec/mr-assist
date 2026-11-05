
import streamlit as st

from app.core.config import DEFAULT_MAX_DISTANCE, DEFAULT_N_RESULTS
from app.services.extractive_answering import select_documentary_answer
from app.services.grounding_validator import validate_answer
from app.services.knowledge_base import (
    collection,
    filter_relevant_matches,
    index_documents,
    search_knowledge,
)
from app.services.manipulation_signals import record_suspicious_submission


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

    st.divider()

    st.checkbox(
        "Mostrar trazabilidad interna (desarrollo)",
        key="show_internal_audit",
    )


# --------------------------------------------------
# 2. HISTORIAL VISUAL
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


def display_message(message: dict) -> None:
    """Muestra el contenido y, solo en modo desarrollo, su evidencia."""

    st.markdown(message["content"])

    if not st.session_state.get("show_internal_audit", False):
        return

    evidence = message.get(
        "audit_evidence",
        message.get("evidence", []),
    )

    if evidence:
        with st.expander("Evidencias documentales (desarrollo)"):
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

    # TASK 16 — registro local de señales de manipulación (fail-safe).
    try:
        record_suspicious_submission(question)
    except Exception:
        # Los fallos de detección/escritura no interrumpen la recuperación ni respuesta.
        pass

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        relevant = []
        candidate = None
        validation = None

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

            # 3. Selección extractiva determinista.
            if not relevant:
                answer = ABSTENTION_MESSAGE
            else:
                try:
                    candidate = select_documentary_answer(
                        question,
                        relevant,
                    )
                except Exception:
                    candidate = None

                if candidate is None:
                    answer = ABSTENTION_MESSAGE
                else:
                    # 4. Validación determinista de anclaje (fail closed).
                    try:
                        validation = validate_answer(
                            question,
                            candidate,
                            relevant,
                        )
                    except Exception:
                        validation = None

                    if (
                        validation is not None
                        and validation.supported
                        and validation.matched_text
                    ):
                        # Solo se muestra el texto original validado.
                        answer = validation.matched_text
                    else:
                        answer = ABSTENTION_MESSAGE

        except Exception:
            # Cualquier fallo de recuperación, selección o validación
            # se resuelve con abstención segura.
            answer = ABSTENTION_MESSAGE

        # 5. Mostrar respuesta verificada o abstención.
        st.markdown(answer)

        # 5b. Trazabilidad interna solo en modo desarrollo.
        if st.session_state.get("show_internal_audit", False):
            with st.expander(
                "🔍 Trazabilidad interna y auditoría",
                expanded=False,
            ):
                if relevant:
                    for item in relevant:
                        st.caption(
                            f"Fuente: {item['source']}"
                        )

                        st.caption(
                            f"Distancia: {item['distance']}"
                        )

                        st.text(item["text"])
                else:
                    st.caption(
                        "Sin fragmentos dentro del umbral de "
                        f"{DEFAULT_MAX_DISTANCE}."
                    )

        # 6. Guardar historial visual.
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "evidence": [],
                "audit_evidence": relevant,
            }
        )
