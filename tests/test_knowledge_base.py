
from app.services.knowledge_base import filter_relevant_matches


def test_accepts_relevant_fragment():
    matches = [
        {"text": "Procedimiento IT", "distance": 0.339},
    ]

    result = filter_relevant_matches(matches, max_distance=0.35)

    assert len(result) == 1
    assert result[0]["text"] == "Procedimiento IT"


def test_rejects_irrelevant_fragment():
    matches = [
        {"text": "Procedimiento IT", "distance": 0.405},
    ]

    result = filter_relevant_matches(matches, max_distance=0.35)

    assert result == []


def test_accepts_fragment_at_threshold():
    matches = [
        {"text": "Fragmento en el límite", "distance": 0.35},
    ]

    result = filter_relevant_matches(matches, max_distance=0.35)

    assert len(result) == 1


def test_handles_empty_results():
    assert filter_relevant_matches([]) == []


def test_filters_multiple_fragments():
    matches = [
        {"text": "Diagnóstico IT", "distance": 0.25},
        {"text": "Procedimiento relacionado", "distance": 0.34},
        {"text": "Documento irrelevante", "distance": 0.41},
    ]

    result = filter_relevant_matches(matches, max_distance=0.35)

    assert [item["text"] for item in result] == [
        "Diagnóstico IT",
        "Procedimiento relacionado",
    ]
