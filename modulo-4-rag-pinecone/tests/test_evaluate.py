import json

from evaluate import (
    GoldenItem,
    evaluate,
    load_golden_set,
    mean,
    precision_at_k,
    recall_at_k,
)
from fakes import make_chunk


def test_precision_counts_relevant_results_over_k():
    retrieved = ["a.md", "b.md", "a.md", "c.md", "a.md"]
    assert precision_at_k(retrieved, "a.md", 5) == 0.6


def test_precision_divides_by_k_even_if_fewer_results_come_back():
    assert precision_at_k(["a.md", "b.md"], "a.md", 5) == 0.2


def test_normalized_precision_divides_by_the_chunks_available():
    # El documento solo tiene 2 fragmentos: con los 2 recuperados la precisión es perfecta.
    assert precision_at_k(["a.md", "a.md", "b.md", "c.md", "d.md"], "a.md", 5, available=2) == 1.0


def test_normalized_precision_never_divides_by_more_than_k():
    assert precision_at_k(["a.md"] * 5, "a.md", 5, available=40) == 1.0


def test_recall_is_one_when_the_expected_document_is_in_the_top_k():
    assert recall_at_k(["b.md", "a.md"], "a.md", 5) == 1.0


def test_recall_is_zero_when_it_is_missing():
    assert recall_at_k(["b.md", "c.md"], "a.md", 5) == 0.0


def test_recall_ignores_results_after_position_k():
    assert recall_at_k(["b.md", "c.md", "a.md"], "a.md", 2) == 0.0


def test_mean():
    assert mean([1.0, 0.0, 0.5]) == 0.5


async def test_evaluate_scores_every_question():
    answers = {
        "p1": [make_chunk("a.md", "x"), make_chunk("b.md", "y")],
        "p2": [make_chunk("c.md", "z")],
    }

    async def retrieve(question):
        return answers[question]

    golden = [
        GoldenItem(pregunta="p1", documento_id_esperado="a.md"),
        GoldenItem(pregunta="p2", documento_id_esperado="a.md"),
    ]
    results = await evaluate(retrieve, golden, k=2, chunk_counts={"a.md": 1})
    assert [r.recall for r in results] == [1.0, 0.0]
    assert [r.precision for r in results] == [0.5, 0.0]
    assert [r.precision_norm for r in results] == [1.0, 0.0]
    assert results[0].retrieved == ["a.md", "b.md"]


def test_golden_set_file_is_valid(tmp_path):
    path = tmp_path / "golden.json"
    path.write_text(json.dumps([{"pregunta": "¿a?", "documento_id_esperado": "a.md"}]))
    assert load_golden_set(path)[0].documento_id_esperado == "a.md"


def test_project_golden_set_references_existing_documents():
    from config import BASE_DIR

    golden = load_golden_set(BASE_DIR / "golden_set.json")
    files = {p.name for p in (BASE_DIR / "data").iterdir()}
    assert len(golden) >= 5
    assert all(item.documento_id_esperado in files for item in golden)
