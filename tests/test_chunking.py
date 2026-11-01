
from app.services.knowledge_base import split_text


def test_split_text_returns_empty_list_for_empty_document():
    assert split_text("") == []


def test_split_text_returns_empty_list_for_blank_document():
    assert split_text("   \n\n   ") == []


def test_split_text_returns_single_chunk_for_short_document():
    assert (
        split_text("Procedimiento IT de ejemplo.")
        == ["Procedimiento IT de ejemplo."]
    )


def test_split_text_joins_paragraphs_within_limit():
    chunks = split_text(
        "Primer párrafo.\n\nSegundo párrafo.",
        max_chars=800,
    )

    assert chunks == ["Primer párrafo.\n\nSegundo párrafo."]


def test_split_text_splits_long_paragraph():
    chunks = split_text("a" * 1000, max_chars=800)

    assert chunks == ["a" * 800, "a" * 200]


def test_split_text_boundary_long_paragraph():
    chunks = split_text("a" * 800, max_chars=800)

    assert chunks == ["a" * 800]


def test_split_text_respects_max_chars_with_multiple_paragraphs():
    text = "\n\n".join(["a" * 30, "b" * 30, "c" * 30])

    chunks = split_text(text, max_chars=50)

    assert chunks == ["a" * 30, "b" * 30, "c" * 30]
    assert all(len(chunk) <= 50 for chunk in chunks)
