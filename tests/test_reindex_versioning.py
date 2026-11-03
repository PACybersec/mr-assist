
from hashlib import sha256
from types import SimpleNamespace

import chromadb
import pytest

from app.services import knowledge_base as kb


@pytest.fixture
def store(tmp_path, monkeypatch):
    chroma_dir = tmp_path / "chroma"
    chroma_dir.mkdir()
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(
        name="it_knowledge",
        metadata={"hnsw:space": "cosine"},
    )
    control_collection = client.get_or_create_collection(
        name="mr_assist_control",
        metadata={"hnsw:space": "cosine"},
    )

    monkeypatch.setattr(kb, "client", client)
    monkeypatch.setattr(kb, "collection", collection)
    monkeypatch.setattr(kb, "control_collection", control_collection)
    monkeypatch.setattr(kb, "DATABASE_DIR", chroma_dir)

    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    monkeypatch.setattr(kb, "KNOWLEDGE_DIR", knowledge_dir)

    embed_state = {"fail": False}

    def fake_embed(model, input):
        if embed_state["fail"]:
            raise RuntimeError("ollama unavailable")
        texts = input if isinstance(input, list) else [input]
        embeddings = []
        for text in texts:
            embeddings.append(
                [
                    float(len(text)),
                    float(sum(ord(char) for char in text)) / 997.0,
                    0.5,
                ]
            )
        return {"embeddings": embeddings}

    monkeypatch.setattr(kb, "ollama", SimpleNamespace(embed=fake_embed))

    return SimpleNamespace(
        tmp_path=tmp_path,
        knowledge_dir=knowledge_dir,
        collection=collection,
        control_collection=control_collection,
        client=client,
        embed_state=embed_state,
    )


def write_doc(store, name: str, text: str) -> None:
    (store.knowledge_dir / name).write_text(
        text,
        encoding="utf-8",
    )


def test_index_creates_committed_version_and_marker(store):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")

    total = kb.index_documents()

    assert total > 0
    assert kb._read_committed_versions().get(
        "procedimientos_it.md"
    ) == 1
    assert kb.search_knowledge("Procedimiento base uno")


def test_reindex_replaces_version_and_cleans_previous(store):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base dos.")
    kb.index_documents()

    assert kb._read_committed_versions().get(
        "procedimientos_it.md"
    ) == 2

    chunks = kb._get_source_chunks("procedimientos_it.md")
    assert chunks
    assert all(chunk["version"] == 2 for chunk in chunks)

    results = kb.search_knowledge("Procedimiento base dos")
    assert results
    assert any(
        "Procedimiento base dos" in item["text"]
        for item in results
    )


def test_embedding_failure_preserves_committed_version(store, monkeypatch):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    def boom(*args, **kwargs):
        raise RuntimeError("ollama unavailable")

    monkeypatch.setattr(kb, "ollama", SimpleNamespace(embed=boom))

    with pytest.raises(RuntimeError):
        kb.index_documents()

    assert kb._read_committed_versions().get(
        "procedimientos_it.md"
    ) == 1


def test_partial_insertion_failure_preserves_committed_version(store, monkeypatch):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    def boom(*args, **kwargs):
        raise RuntimeError("insert failed")

    monkeypatch.setattr(kb.collection, "add", boom)

    with pytest.raises(RuntimeError):
        kb.index_documents()

    assert kb._read_committed_versions().get(
        "procedimientos_it.md"
    ) == 1
    assert kb.search_knowledge("Procedimiento base uno")


def test_failure_before_marker_creation_leaves_previous_version_active(
    store,
    monkeypatch,
):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    def boom(*args, **kwargs):
        raise RuntimeError("control add failed")

    monkeypatch.setattr(kb.control_collection, "add", boom)

    with pytest.raises(RuntimeError):
        kb.index_documents()

    assert kb._read_committed_versions().get(
        "procedimientos_it.md"
    ) == 1
    assert kb.search_knowledge("Procedimiento base uno")


def test_ambiguous_marker_write_outcome_and_recovery(store, monkeypatch):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    def boom(*args, **kwargs):
        raise RuntimeError("ambiguous marker failure")

    monkeypatch.setattr(kb.control_collection, "add", boom)

    with pytest.raises(RuntimeError):
        kb.index_documents()

    committed = kb._read_committed_versions()
    assert committed.get("procedimientos_it.md") == 1
    assert kb.search_knowledge("Procedimiento base uno")


def test_uncommitted_chunks_never_in_search(store):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    staged_id = sha256(b"staged-chunk").hexdigest()
    kb.collection.add(
        ids=[staged_id],
        documents=["## Sintomas\nProcedimiento base dos."],
        embeddings=[[0.9, 0.9, 0.9]],
        metadatas=[{"source": "procedimientos_it.md", "version": 2}],
    )

    results = kb.search_knowledge("Procedimiento base dos")

    assert results
    assert all(
        "Procedimiento base uno" in item["text"]
        for item in results
    )


def test_legacy_version_zero_is_searchable(store):
    kb.collection.add(
        ids=["legacy-1"],
        documents=["Procedimiento historico legacy"],
        embeddings=[[0.3, 0.3, 0.3]],
        metadatas=[{"source": "historico.md"}],
    )

    active = kb._load_active_versions()

    assert active.get("historico.md") == kb.LEGACY_VERSION

    results = kb.search_knowledge("Procedimiento historico legacy")

    assert results


def test_no_mixed_versions_per_source(store):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base dos.")
    kb.index_documents()

    source = "procedimientos_it.md"
    committed = kb._read_committed_versions()
    assert committed.get(source) == 2

    chunks = kb._get_source_chunks(source)
    assert chunks
    assert all(chunk["version"] == 2 for chunk in chunks)


def test_staged_chunks_do_not_displace_valid_results(store):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    staged_id = sha256(b"staged-chunk-2").hexdigest()
    kb.collection.add(
        ids=[staged_id],
        documents=["## Sintomas\nProcedimiento base dos."],
        embeddings=[[0.9, 0.9, 0.9]],
        metadatas=[{"source": "procedimientos_it.md", "version": 2}],
    )

    results = kb.search_knowledge("Procedimiento base uno")

    assert results
    assert all(
        "Procedimiento base uno" in item["text"]
        for item in results
    )


def test_multiple_sources_different_versions(store):
    write_doc(store, "fuente_a.md", "## Contenido\nContenido fuente A.")
    kb.index_documents()

    write_doc(store, "fuente_b.md", "## Contenido\nContenido fuente B.")
    kb.index_documents()

    active = kb._load_active_versions()

    assert active.get("fuente_a.md") == 2
    assert active.get("fuente_b.md") == 1


def test_empty_document_publication_clears_source(store):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    (store.knowledge_dir / "procedimientos_it.md").write_text(
        "",
        encoding="utf-8",
    )
    total = kb.index_documents()

    assert total == 0
    assert kb._read_committed_versions().get(
        "procedimientos_it.md"
    ) == 2
    assert kb._get_source_chunks("procedimientos_it.md") == []
    assert kb.search_knowledge("Procedimiento base uno") == []


def test_deleted_source_cleanup_after_complete_pass(store):
    write_doc(store, "fuente_a.md", "## Contenido\nContenido fuente A.")
    write_doc(store, "fuente_b.md", "## Contenido\nContenido fuente B.")
    kb.index_documents()

    (store.knowledge_dir / "fuente_a.md").unlink()

    kb.index_documents()

    sources = kb._get_all_sources()
    assert "fuente_a.md" not in sources
    assert "fuente_b.md" in sources


def test_cleanup_failure_after_publication_keeps_committed_version(
    store,
    monkeypatch,
):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base dos.")

    def boom(*args, **kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(kb.collection, "delete", boom)

    with pytest.raises(RuntimeError):
        kb.index_documents()

    assert kb._read_committed_versions().get(
        "procedimientos_it.md"
    ) == 2
    assert kb.search_knowledge("Procedimiento base dos")


def test_failed_pass_does_not_cleanup_unrelated_sources(store, monkeypatch):
    write_doc(store, "fuente_a.md", "## Contenido\nContenido fuente A.")
    write_doc(store, "fuente_b.md", "## Contenido\nContenido fuente B.")
    kb.index_documents()

    (store.knowledge_dir / "fuente_a.md").unlink()

    monkeypatch.setattr(
        kb,
        "ollama",
        SimpleNamespace(
            embed=lambda *args, **kwargs: (_ for _ in ()).throw(
                RuntimeError("ollama down")
            )
        ),
    )

    with pytest.raises(RuntimeError):
        kb.index_documents()

    sources = kb._get_all_sources()
    assert "fuente_a.md" in sources


def test_file_locks_shared_and_exclusive(store):
    with kb._exclusive_lock():
        pass
    with kb._shared_lock():
        pass
    with kb._exclusive_lock():
        pass


def test_staged_chunks_cannot_displace_committed_results(store):
    write_doc(store, "procedimientos_it.md", "## Sintomas\nProcedimiento base uno.")
    kb.index_documents()

    query_embedding = (
        kb.ollama.embed(
            model=kb.EMBEDDING_MODEL,
            input="Procedimiento base dos",
        )["embeddings"][0]
    )

    staged_id = sha256(b"staged-displacement").hexdigest()
    kb.collection.add(
        ids=[staged_id],
        documents=["## Sintomas\nProcedimiento base dos."],
        embeddings=[query_embedding],
        metadatas=[{"source": "procedimientos_it.md", "version": 2}],
    )

    results = kb._retrieve_source(
        query_embedding,
        "procedimientos_it.md",
        1,
        3,
    )

    assert results
    assert all(
        "Procedimiento base uno" in item["text"]
        for item in results
    )


def test_legacy_version_zero_retrieval_is_isolated(store):
    kb.collection.add(
        ids=["legacy-3"],
        documents=["Procedimiento historico legacy"],
        embeddings=[[0.3, 0.3, 0.3]],
        metadatas=[{"source": "historico.md"}],
    )

    results = kb._retrieve_source(
        [0.3, 0.3, 0.3],
        "historico.md",
        0,
        3,
    )

    assert results
    assert all(
        item["source"] == "historico.md"
        for item in results
    )


def test_retrieval_multiple_sources_different_versions_isolated(store):
    write_doc(store, "fuente_a.md", "## Contenido\nContenido fuente A.")
    kb.index_documents()

    write_doc(store, "fuente_b.md", "## Contenido\nContenido fuente B.")
    kb.index_documents()

    active = kb._load_active_versions()
    assert active.get("fuente_a.md") == 2
    assert active.get("fuente_b.md") == 1

    embedding_a = (
        kb.ollama.embed(
            model=kb.EMBEDDING_MODEL,
            input="Contenido fuente A",
        )["embeddings"][0]
    )
    embedding_b = (
        kb.ollama.embed(
            model=kb.EMBEDDING_MODEL,
            input="Contenido fuente B",
        )["embeddings"][0]
    )

    results_a = kb._retrieve_source(embedding_a, "fuente_a.md", 2, 3)
    results_b = kb._retrieve_source(embedding_b, "fuente_b.md", 1, 3)

    assert results_a
    assert results_b


def test_merged_results_sorted_by_distance_and_limited_to_n_results(store):
    write_doc(store, "fuente_a.md", "## Contenido\nContenido fuente A.")
    write_doc(store, "fuente_b.md", "## Contenido\nContenido fuente B.")
    kb.index_documents()

    results = kb.search_knowledge("Contenido")

    assert len(results) <= 3
    distances = [item["distance"] for item in results]
    assert distances == sorted(distances)
