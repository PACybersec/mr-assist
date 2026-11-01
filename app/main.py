
import streamlit as st

from app.core.config import DEFAULT_MAX_DISTANCE, DEFAULT_N_RESULTS
from app.services.knowledge_base import (
    collection,
    filter_relevant_matches,
    index_documents,
    search_knowledge,
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

                    answer = (
                        "He encontrado fragmentos relacionados "
                        "con tu consulta en la documentación.\n\n"
                        "Para mantener el modo estricto, te muestro "
                        "el contenido original sin reformularlo "
                        "ni añadir instrucciones externas."
                    )

                    evidence = [
                        {
                            "source": item["source"],
                            "text": item["text"],
                        }
                        for item in relevant
                    ]

                else:

                    answer = (
                        "No he encontrado evidencia documental "
                        "suficientemente relevante para responder "
                        "a tu consulta.\n\n"
                        "No puedo proporcionar instrucciones "
                        "sin respaldo documental. Consulta al "
                        "equipo responsable si necesitas "
                        "información adicional."
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

            # 5. Guardar historial visual.
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "evidence": evidence,
                }
            )

        except Exception as error:

            st.error(
                "No se pudo completar la consulta. "
                "Comprueba Ollama y la base de conocimiento."
            )

            st.caption(
                f"Detalle técnico: {error}"
            )
