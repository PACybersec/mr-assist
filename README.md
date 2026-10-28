# Mr.Assist — Local AI IT Operations Assistant

Mr.Assist is a local AI-powered IT operations assistant built with Python, Streamlit, Ollama and ChromaDB. It uses retrieval-augmented generation (RAG) to answer IT support questions using a knowledge base of documented procedures.

The project explores how local language models and semantic search can support IT troubleshooting while reducing unsupported answers when relevant documentation is unavailable.

## Features

- **Local LLM inference:** generates responses using Ollama.
- **Retrieval-Augmented Generation (RAG):** retrieves relevant passages from IT documentation before generating an answer.
- **Vector search:** stores document embeddings in ChromaDB.
- **Source transparency:** displays the filenames used to support responses.
- **Relevance filtering:** rejects retrieved passages that exceed a configurable distance threshold.
- **Knowledge base management:** re-indexes Markdown documents from the Streamlit interface.
- **Conversation history:** preserves recent messages during a session.
- **Automated tests:** tests the relevance-filtering logic.

## Technology Stack

- Python
- Streamlit
- Ollama
- Qwen 3 (0.6B)
- Nomic Embed Text
- ChromaDB
- Pytest

## Architecture

1. **Knowledge ingestion:** Markdown IT procedures are loaded from the local knowledge directory.
2. **Chunking and embeddings:** documents are split into manageable passages and converted into embeddings.
3. **Vector storage:** ChromaDB stores the passages, embeddings and source metadata.
4. **Retrieval:** the user's question is embedded and compared with indexed passages.
5. **Relevance filtering:** passages outside the configured distance threshold are discarded.
6. **Response generation:** Ollama generates an answer using the retrieved documentation and recent conversation history.
7. **Source display:** the interface identifies the documents used for the response.

## Getting Started

### Prerequisites

- Python 3.11 or newer, subject to dependency compatibility.
- Ollama installed and running.
- Git.

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/mr-assist.git
cd mr-assist
```

Replace `YOUR-USERNAME` with your GitHub username.

### 2. Create and activate a virtual environment

On Linux with Fish shell:

```fish
python -m venv .venv
source .venv/bin/activate.fish
```

### 3. Install dependencies

Install the packages used by the project:

```bash
python -m pip install streamlit ollama chromadb pytest
```

### 4. Download the required models

```bash
ollama pull qwen3:0.6b
ollama pull nomic-embed-text
```

Ensure the Ollama service is running before launching the application.

### 5. Run the application

```bash
streamlit run app/main.py
```

Open the local address printed by Streamlit, usually `http://localhost:8501`.

### 6. Run the tests

```bash
python -m pytest -v
```

## Knowledge Base

Place IT procedure documents in Markdown format in:

`app/data/knowledge/`

Use the **Update documents** button in the application sidebar to index the documents.

The current example procedure covers troubleshooting an unavailable internal server, including initial diagnosis, resolution and security considerations.

## Security Considerations

- The application is designed to run locally.
- Retrieved documentation is treated as reference material rather than trusted instructions.
- The assistant is instructed not to invent internal policies when relevant documentation is unavailable.
- The example knowledge base contains fictional IT procedures.
- The project does not currently implement enterprise authentication, role-based access control or production-grade audit logging.
- The relevance threshold is configurable and requires evaluation against a representative dataset; it is not a calibrated confidence score.

Do not add credentials, API keys, private logs, personal information or confidential company documentation to the public repository.

## Current Limitations

- Response quality depends on the selected language model and available documentation.
- Retrieval quality depends on document chunking, embedding quality and the configured distance threshold.
- The current test suite focuses on relevance filtering rather than end-to-end model evaluation.
- The application is a learning and portfolio project, not a production-ready IT service management platform.

## Roadmap

- [ ] Add more troubleshooting procedures.
- [ ] Improve retrieval evaluation with a representative test dataset.
- [ ] Add end-to-end RAG tests.
- [ ] Improve error handling and application health checks.
- [ ] Add structured logging and response evaluation.
- [ ] Develop a security-focused evaluation suite for prompt injection and untrusted documents.
- [ ] Improve the interface and add demonstration screenshots.

## Disclaimer

This project is for educational and portfolio purposes. Responses should be reviewed before being used for operational decisions.

## Author

Cybersecurity enthusiast exploring local AI, IT operations, security engineering and practical automation.
