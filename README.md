# Mr.Assist

**Local AI operations assistant for IT support — grounded, deterministic, experimental.**

Mr.Assist is a local AI-powered IT operations assistant built with Python, Streamlit, Ollama and ChromaDB. It uses retrieval over a knowledge base of documented IT procedures to support troubleshooting, and makes retrieved evidence available for optional development traceability.

> [!WARNING]
> Experimental prototype. No readiness for production is claimed.

---

## Current Status

Mr.Assist is currently an **experimental 0.5 prototype** of an IT assistant based on RAG, with a retrieval-focused core and an experimental generative component. Validation so far is **limited**, and no readiness for production is claimed.

### Implemented

- Streamlit interface.
- ChromaDB persistent vector store.
- Ollama embeddings with `nomic-embed-text`.
- Document retrieval.
- Semantic distance filtering with a maximum distance of `0.40`.
- Deterministic extractive answering: retrieval, filtering, selection of an exact span from the original evidence, and fail-closed validation; responds with an extractive answer or abstains.
- Strict mode: deterministic extractive response from validated evidence, or abstention.

### Experimental generation

- Response generation with a local LLM through Ollama (`qwen3:0.6b`) is implemented as an experiment and is **not part of the default flow**; deterministic extractive answering remains the default.

### Selector adjustments after v0.5 (local work)

- Generic context terms (diagnosis phase, internal/application context, affected-user context) are excluded from content requirements.
- Generic question-framing verbs are excluded from content requirements.
- Ordinal step references select numbered candidates by their list marker.
- The pronominal affectation construction is normalized, including the plural form **`se ven afectados`**.
- Generic concept tokens (for example `procedimiento`) are excluded from the coverage denominator.

### Current parameters

| Parameter | Value |
| --- | --- |
| Maximum distance | `0.40` |
| Lexical content coverage | `0.75` |

---

## Current Verification

- Date: 2026-11-06 (local verification during the TASK 33–35 workflow).
- Full suite: **212 passed, 1 warning** (ChromaDB/asyncio deprecation, non-blocking); 0 failures; exit code 0.

### Historical status of tests (v0.5 and before)

- Automated regression (historical): last reported run **182 passed, 1 warning** (reported; an earlier documented figure of **169 passed** also exists and the discrepancy is not reconciled). TASK 20 run: **196 tests; 193 passed, 3 failed, 1 warning; exit code 1**.
- Isolated evaluation (historical): **11 passed, 3 failed** (12 functional cases and 2 auxiliary checks); failures A1, A3 and A4.

### Manual tests in Firefox (historical, communicated by the user)

- A1 (browser error) false rejection; A2 (single user) correct answer; A3 (service state) correct answer, retains "si se tienen permisos"; A4 (first step) false rejection; denied question: expected abstention; RTO not documented: correct abstention; override attempt plus firewall: safe abstention.

> [!NOTE]
> A3 failed in the isolated historical evaluation and worked in the manual Firefox test communicated by the user; the difference was not investigated further here and the isolated failure was not treated as resolved.

- Manual visual review of the traceability control (reported by the user): enabling and disabling it did not alter answers and did not duplicate messages.

### Development-set evaluation (synthetic, diagnostic)

- The development set (`evals/dev_set_cases.json`) is synthetic and is used for **development** of the selector.
- It was used to guide local adjustments (for example, TASK 33 for DEV-016 and DEV-035).
- It is **not** an independent evaluation and does not constitute a general quality or security guarantee.

### Limits of the current verification

- Functional evaluation with synthetic evidence.
- External services simulated in the relevant tests.
- This run did not verify real retrieval with embeddings and Ollama, nor does it constitute an end-to-end test with those services.
- A3 continues to differ from the historically communicated manual result.

## Features

- **Local embeddings:** Ollama with the `nomic-embed-text` model.
- **Vector storage:** ChromaDB persistent client, cosine distance.
- **Retrieval:** semantic search over indexed Markdown chunks.
- **Relevance filtering:** distance threshold (`0.40` maximum distance; lexical content coverage `0.75`).
- **Strict mode:** deterministic extractive response from validated evidence, or abstention.
- **Source transparency:** optional development traceability showing source, distance and fragment; disabled by default and distinct from the manipulation-signal log. The traceability control only shows or hides the evidence in the interface; it does not disable its processing, its storage, or the manipulation-signal log.
- **Knowledge base management:** re-index documents from the Streamlit interface.
- **Conversation history:** messages kept in the Streamlit session state.

## How It Works

The current flow is:

`ingest → chunk → embed → store → retrieve → filter → select → validate → respond with a documentary answer or abstain`

1. **Knowledge ingestion:** Markdown IT procedures from `app/data/knowledge/`.
2. **Chunking:** simple paragraph-based splitting with a maximum of 800 characters.
3. **Embeddings:** Ollama `nomic-embed-text`.
4. **Storage:** ChromaDB collection `it_knowledge`, cosine distance.
5. **Retrieval:** embed the user question and query the collection.
6. **Filtering:** discard retrieved fragments whose distance exceeds the configured threshold.
7. **Response:** return the validated extractive answer (exact original span), or abstain.

## Technology Stack

- Python
- Streamlit
- ChromaDB
- Ollama
- Nomic Embed Text
- Pytest

## Getting Started

### Prerequisites

- Python 3.14.9 (used for the TASK 20 validation run; an earlier historical run used Python 3.14.8; other Python versions have not been validated).
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

- Retrieval evaluation uses the **R01–R05** cases of the retrieval-focused evaluator (`tests/test_retrieval_evaluation.py`), which checks expected evidence and explicitly forbidden evidence.
- Functional evaluation uses the **A, B and C groups** of the full-flow evaluator (`tests/test_functional_evaluation.py`), which checks documented answers, abstentions and adversarial cases.

- The suite includes relevance-filtering tests and retrieval evaluation tests.
- The last reported regression run was **182 passed, 1 warning** (reported; see Current Status for the discrepancy with the earlier documented figure of **169 passed**).
- The evaluators rely on textual matching, not semantic understanding, and do not validate abstention behavior in the running application.

## Security Considerations

- Designed for local use.
- Retrieved documentation is treated as reference material rather than trusted instructions.
- The deterministic prompt-injection filter is narrow and experimental; the local manipulation-signal log is not a general prompt-injection protection.

### Local manipulation-signal log

- The local manipulation-signal log records explicit instruction-override signals.
- It is independent of the optional visual traceability of evidence.
- Path: `app/data/security_events/manipulation_signals.jsonl`.
- Each event stores a UTC timestamp, UUID, category `instruction_override`, matched rules and detector version; the question text is not stored.
- Rotation: 1 MiB maximum with 3 backups.
- Currently limited to a single process.
- This is a rule-based signal log; it is not a general prompt-injection defense.

> [!WARNING]
> Do not commit or publish credentials, passwords, API keys, tokens, private logs, personal information, or confidential company documentation to the repository.

## Limitations

- Response generation with a local LLM is implemented experimentally (qwen3:0.6b) but is not part of the default flow.
- The retrieval distance threshold is set to `0.40` (current parameter; no general quality calibration is claimed).
- Chunking is simple (paragraph-based, maximum 800 characters).
- Evaluation relies on textual matching, not semantic understanding.
- The fail-closed behavior applies to the deterministic extractive flow and the evaluated cases; it is not presented as a general security or correctness guarantee.
- Known remaining false rejections in the development set: **DEV-004, DEV-005, DEV-006, DEV-032 and DEV-033**; the selector still returns `None` where documentary answers were expected.
- The project is not production-ready.

## Roadmap

- Integrate the experimental local response generation (qwen3:0.6b) into the default flow, subject to further evaluation and a later explicit decision.
- Continue evaluation of the retrieval threshold and answer quality.
- Add end-to-end RAG tests.

## Disclaimer

This project is for educational and portfolio purposes. Responses should be reviewed before being used for operational decisions.

## Author

**Pablo Acosta Arango**

Cybersecurity enthusiast exploring local AI, IT operations, security engineering and practical automation.
