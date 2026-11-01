# Mr.Assist — Local AI IT Operations Assistant

Mr.Assist is a local AI-powered IT operations assistant built with Python, Streamlit, Ollama and ChromaDB. It uses retrieval over a knowledge base of documented IT procedures to support troubleshooting, and presents retrieved evidence to the user.

## Current Status

Mr.Assist is currently a **retrieval-focused prototype**, not a complete generative RAG system.

**Implemented:**

- Streamlit interface.
- ChromaDB persistent vector store.
- Ollama embeddings with `nomic-embed-text`.
- Document retrieval.
- Semantic distance filtering with a threshold of `0.35`.
- Strict mode: static response including retrieved evidence when fragments are accepted, or static abstention when none are accepted.

**Planned:**

- Response generation with a local LLM through Ollama, using retrieved documentation and conversation history, with strict abstention safeguards. This is **not implemented** in the current version.

Last confirmed test run: **16 passed**, with 1 unrelated ChromaDB deprecation warning, on November 1, 2026.

## Features

- **Local embeddings:** Ollama with the `nomic-embed-text` model.
- **Vector storage:** ChromaDB persistent client, cosine distance.
- **Retrieval:** semantic search over indexed Markdown chunks.
- **Relevance filtering:** distance threshold (`0.35` by default; pending calibration).
- **Strict mode:** static response from accepted evidence, or abstention.
- **Source transparency:** retrieved evidence is displayed to the user.
- **Knowledge base management:** re-index documents from the Streamlit interface.
- **Conversation history:** messages kept in the Streamlit session state.

## How It Works

The current flow is:

`ingest → chunk → embed → store → retrieve → filter → respond with static evidence or abstain`

1. **Knowledge ingestion:** Markdown IT procedures from `app/data/knowledge/`.
2. **Chunking:** simple paragraph-based splitting with a maximum of 800 characters.
3. **Embeddings:** Ollama `nomic-embed-text`.
4. **Storage:** ChromaDB collection `it_knowledge`, cosine distance.
5. **Retrieval:** embed the user question and query the collection.
6. **Filtering:** discard retrieved fragments whose distance exceeds the configured threshold.
7. **Response:** return a static response including the accepted evidence, or abstain.

## Technology Stack

- Python
- Streamlit
- ChromaDB
- Ollama
- Nomic Embed Text
- Pytest

## Getting Started

### Prerequisites

- Python 3.14.8 (used for the latest verified test run; other Python versions have not been validated).
- Ollama installed and running.
- Pull the current embedding model:

```bash
ollama pull nomic-embed-text
```

### Installation

```bash
git clone https://github.com/PACybersec/mr-assist.git
cd mr-assist
python -m venv .venv
source .venv/bin/activate
python -m pip install streamlit ollama chromadb pytest
```

### Run

```bash
streamlit run app/main.py
```

After running, the application can be accessed at `http://localhost:8501`.

### Tests

```bash
python -m pytest -v
```

## Knowledge Base

Place IT procedure documents in Markdown format in:

`app/data/knowledge/`

Use the **Update documents** button in the application sidebar to re-index the documents.

The current example knowledge base covers troubleshooting an unavailable internal server, including diagnosis, resolution and security considerations.

## Evaluation

Retrieval evaluation uses cases R01–R05. The evaluator checks expected evidence and explicitly forbidden evidence.

- The suite includes relevance-filtering tests and retrieval evaluation tests.
- Last confirmed run: **16 passed**, with 1 unrelated ChromaDB deprecation warning, on November 1, 2026.
- The evaluator relies on textual matching, not semantic understanding, and does not validate abstention behavior in the running application.

## Security Considerations

- Designed for local use.
- Retrieved documentation is treated as reference material rather than trusted instructions.
- No complete prompt-injection defense has been implemented and validated yet.

> [!WARNING]
> Do not commit or publish credentials, passwords, API keys, tokens, private logs, personal information, or confidential company documentation to the repository.

## Limitations

- Response generation with a local LLM is not implemented.
- The retrieval threshold (`0.35`) is pending calibration.
- Chunking is simple (paragraph-based, maximum 800 characters).
- Evaluation relies on textual matching, not semantic understanding.
- The project is not production-ready.

## Roadmap

- Implement local response generation through Ollama.
- Calibrate the retrieval threshold.
- Add end-to-end RAG tests.

## Disclaimer

This project is for educational and portfolio purposes. Responses should be reviewed before being used for operational decisions.

## Author

Cybersecurity enthusiast exploring local AI, IT operations, security engineering and practical automation.
