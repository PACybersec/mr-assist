
from contextlib import contextmanager
from hashlib import sha256
import json
import fcntl
from pathlib import Path

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

CONTROL_COLLECTION_NAME = "mr_assist_control"
COMMIT_KIND = "commit"
LEGACY_VERSION = 0
LOCK_FILE_NAME = "reindex.lock"

client = chromadb.PersistentClient(
    path=str(DATABASE_DIR)
)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)
control_collection = client.get_or_create_collection(
    name=CONTROL_COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _lock_path() -> Path:
    return Path(str(DATABASE_DIR)) / LOCK_FILE_NAME


@contextmanager
def _file_lock(mode: int):
    lock_path = _lock_path()
    with open(lock_path, "a+") as lock_file:
        fcntl.flock(lock_file.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _exclusive_lock():
    with _file_lock(fcntl.LOCK_EX):
        yield


@contextmanager
def _shared_lock():
    with _file_lock(fcntl.LOCK_SH):
        yield


def _marker_id(source: str, version: int) -> str:
    return sha256(
        f"mr-assist:commit:{source}:{version}".encode("utf-8")
    ).hexdigest()


def _publish_commit_marker(
    source: str,
    version: int,
    chunk_count: int,
) -> None:
    control_collection.add(
        ids=[_marker_id(source, version)],
        documents=[
            json.dumps(
                {
                    "source": source,
                    "version": version,
                    "chunk_count": chunk_count,
                }
            )
        ],
        metadatas=[
            {
                "source": source,
                "version": version,
                "kind": COMMIT_KIND,
                "chunk_count": chunk_count,
            }
        ],
    )


def _read_committed_versions() -> dict:
    markers = control_collection.get(
        where={"kind": COMMIT_KIND}
    )
    committed = {}
    for meta in markers.get("metadatas") or []:
        source = meta.get("source")
        version = meta.get("version")
        if source is None or version is None:
            continue
        committed[source] = max(
            committed.get(source, LEGACY_VERSION),
            int(version),
        )
    return committed


def _get_source_chunks(source: str) -> list:
    records = collection.get(where={"source": source})
    chunks = []
    ids = records.get("ids") or []
    documents = records.get("documents") or []
    metadatas = records.get("metadatas") or []
    for index, meta in enumerate(metadatas):
        chunks.append(
            {
                "id": ids[index],
                "text": documents[index],
                "version": (
                    int(meta.get("version"))
                    if meta.get("version") is not None
                    else LEGACY_VERSION
                ),
            }
        )
    return chunks


def _get_all_sources() -> set:
    records = collection.get()
    sources = set()
    for meta in records.get("metadatas") or []:
        source = meta.get("source")
        if source:
            sources.add(source)
    return sources


def _allocate_version(source: str) -> int:
    versions = [LEGACY_VERSION]
    committed = _read_committed_versions()
    if source in committed:
        versions.append(committed[source])
    for chunk in _get_source_chunks(source):
        versions.append(chunk["version"])
    return max(versions) + 1


def _get_chunk_ids(source: str) -> list:
    records = collection.get(where={"source": source})
    return list(records.get("ids") or [])


def _cleanup_deleted_sources(indexed_sources: set) -> None:
    all_sources = _get_all_sources()
    for source in sorted(all_sources - indexed_sources):
        ids = _get_chunk_ids(source)
        if ids:
            collection.delete(ids=ids)
        markers = control_collection.get(where={"source": source})
        marker_ids = list(markers.get("ids") or [])
        if marker_ids:
            control_collection.delete(ids=marker_ids)


def _load_active_versions() -> dict:
    active = dict(_read_committed_versions())
    for source in _get_all_sources():
        if source not in active:
            active[source] = LEGACY_VERSION
    return active


def _retrieve_source(
    embedding: list,
    source: str,
    version: int,
    n_results: int,
) -> list:
    if version > LEGACY_VERSION:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={
                "$and": [
                    {"source": {"$eq": source}},
                    {"version": {"$eq": version}},
                ]
            },
            include=["documents", "metadatas", "distances"],
        )
    else:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            where={"source": source},
            include=["documents", "metadatas", "distances"],
        )

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]
    matches = []
    for index, meta in enumerate(metadatas[0]):
        chunk_version = (
            int(meta.get("version"))
            if meta.get("version") is not None
            else LEGACY_VERSION
        )
        if chunk_version != version:
            continue
        matches.append(
            {
                "text": documents[0][index],
                "source": source,
                "distance": distances[0][index],
            }
        )
    return matches


def split_text(text: str, max_chars: int = DEFAULT_MAX_CHARS) -> list:
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
    with _exclusive_lock():
        if not KNOWLEDGE_DIR.exists():
            raise FileNotFoundError(
                f"No existe la carpeta: {KNOWLEDGE_DIR}"
            )

        documents = sorted(KNOWLEDGE_DIR.glob("*.md"))

        if not documents:
            raise ValueError(
                f"No hay documentos Markdown en {KNOWLEDGE_DIR}"
            )

        total = 0
        indexed_sources = set()

        for document in documents:
            indexed_sources.add(document.name)
            text = document.read_text(encoding="utf-8")
            chunks = split_text(text)
            version = _allocate_version(document.name)
            old_ids = _get_chunk_ids(document.name)

            if chunks:
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
                        f"{document.name}:{version}:{i}:{chunk}".encode(
                            "utf-8"
                        )
                    ).hexdigest()
                    for i, chunk in enumerate(chunks)
                ]

                collection.add(
                    ids=ids,
                    documents=chunks,
                    embeddings=embeddings,
                    metadatas=[
                        {"source": document.name, "version": version}
                        for _ in chunks
                    ],
                )

            _publish_commit_marker(
                document.name,
                version,
                len(chunks),
            )

            if old_ids:
                collection.delete(ids=old_ids)

            total += len(chunks)

        _cleanup_deleted_sources(indexed_sources)

        return total


def search_knowledge(
    question: str,
    n_results: int = DEFAULT_N_RESULTS,
) -> list[dict]:
    """Recupera fragmentos relevantes para una pregunta."""
    if not question.strip():
        return []

    with _shared_lock():
        embedding = ollama.embed(
            model=EMBEDDING_MODEL,
            input=question,
        )["embeddings"][0]

        active_versions = _load_active_versions()
        matches = []

        for source in sorted(active_versions):
            source_matches = _retrieve_source(
                embedding,
                source,
                active_versions[source],
                n_results,
            )
            matches.extend(source_matches)

        matches.sort(key=lambda item: item["distance"])
        return matches[:n_results]


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
