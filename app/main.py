
import streamlit as st
import ollama

MODEL_NAME = "qwen3:0.6b"

st.set_page_config(
    page_title="MR.ASSIST",
    page_icon="🤖",
    layout="centered",
)

st.title("🤖 Mr. Assist 24/7")
st.subheader("Asistente inteligente de operaciones IT")

st.write(
    "Tu asistente local para consultar conceptos, "
    "procedimientos y problemas de operaciones IT."
)

# Historial de conversación
if "messages" not in st.session_state:
    st.session_state.messages = []

# Mostrar mensajes anteriores
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Entrada del usuario
question = st.chat_input("¿En qué puedo ayudarte?")

if question:
    # Mostrar y guardar la pregunta
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    # Generar respuesta mediante Ollama
    with st.chat_message("assistant"):
        with st.spinner("ALTO Assist está pensando..."):
            try:
                response = ollama.chat(
                    model=MODEL_NAME,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "Eres ALTO Assist, un asistente "
                                "de operaciones IT y ciberseguridad. "
                                "Responde en español, con claridad, "
                                "precisión y ejemplos prácticos. "
                                "Si no sabes algo, dilo claramente. "
                                "No inventes datos ni afirmes haber "
                                "consultado sistemas a los que no "
                                "tienes acceso."
                            ),
                        },
                        *st.session_state.messages,
                    ],
                )

                answer = response["message"]["content"]
                st.markdown(answer)

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    "No se pudo obtener una respuesta de Ollama. "
                    "Comprueba que el servicio está activo y que "
                    f"el modelo '{MODEL_NAME}' está disponible."
                )
                st.caption(f"Detalle técnico: {error}")

