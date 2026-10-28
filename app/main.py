
import streamlit as st

st.set_page_config(
    page_title="MR.ASSIST",
    page_icon="🤖",
    layout="centered",
)

st.title("ASISTENCIA 24/7")
st.subheader("Asistente inteligente de operaciones IT")

st.write(
    "Bienvenido. Este asistente te ayudará a "
    "consultar procedimientos e incidencias."
)

question = st.text_input(
    "¿Qué necesitas saber?"
)

if st.button("Enviar pregunta"):
    if question.strip():
        st.info(
            "Tu pregunta se ha recibido. "
            "En el siguiente paso conectaremos "
            "el asistente con Ollama."
        )
    else:
        st.warning("Escribe una pregunta primero.")
