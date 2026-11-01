
from hashlib import sha256

import chromadb
import ollama

from app.core.config import (
    COLLECTION_NAME,
    DATABASE_DIR,
    DEFAULT_MAX_DISTANCE,
    DEFAULT_MAX_CHARS,
    DEFAULT_N_RESULTS,
    EMBEDDING_MODEL,
    KNOWLEDGE_DIR,
)

# Base vectorial persistente
client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list[str]:
    """Divide un documento en fragmentos manejables."""
    paragraphs = [
        paragraph.strip()
        for paragraph in text.split("\n\n")
        if paragraph.strip()
    ]

    chunks = []
    current = ""

    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""

            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:]

        candidate = (
            f"{current}\n\n{paragraph}"
            if current
            else paragraph
        )

        if len(candidate) > max_chars:
            if current:
                chunks.append(current)
            current = paragraph
        else:
            current = candidate

    if current:
        chunks.append(current)

    return chunks


def index_documents() -> int:
    """Indexa los documentos Markdown de la base de conocimiento."""
    if not KNOWLEDGE_DIR.exists():
        raise FileNotFoundError(
            f"No existe la carpeta: {KNOWLEDGE_DIR}"
        )

    documents = sorted(KNOWLEDGE_DIR.glob("*.md"))

    if not documents:
        raise ValueError(
            f"No hay documentos Markdown en {KNOWLEDGE_DIR}"
        )

    indexed = 0

    for document in documents:
        text = document.read_text(encoding="utf-8")
        chunks = split_text(text)

        # Eliminar los fragmentos anteriores de este documento.
        existing = collection.get(
            where={"source": document.name}
        )

        if not chunks:
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            continue

        embeddings = ollama.embed(
            model=EMBEDDING_MODEL,
            input=chunks,
        )["embeddings"]

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Los embeddings generados no coinciden con los fragmentos."
            )

        ids = [
            sha256(
                f"{document.name}:{i}:{chunk}".encode("utf-8")
            ).hexdigest()
            for i, chunk in enumerate(chunks)
        ]

        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[
                {"source": document.name}
                for _ in chunks
            ],
        )

        indexed += len(chunks)

    return indexed


def search_knowledge(
    question: str,
    n_results: int = DEFAULT_N_RESULTS,
) -> list[dict]:
    """Recupera fragmentos relevantes para una pregunta."""
    if not question.strip():
        return []

    total = collection.count()

    if total == 0:
        return []

    embedding = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question,
    )["embeddings"][0]

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, total),
        include=["documents", "metadatas", "distances"],
    )

    matches = []

    for document, metadata, distance in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        matches.append(
            {
                "text": document,
                "source": metadata["source"],
                "distance": distance,
            }
        )

    return matches


def filter_relevant_matches(
    matches: list[dict],
    max_distance: float = DEFAULT_MAX_DISTANCE,
) -> list[dict]:
    """Devuelve únicamente los fragmentos dentro del umbral."""
    return [
        match
        for match in matches
        if match["distance"] <= max_distance
    ]
