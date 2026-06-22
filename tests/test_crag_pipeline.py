"""Tests for additive standalone CRAG workflow."""

from __future__ import annotations

from src.crag_pipeline import CRAGResources, build_crag_workflow, crag_mermaid, run_crag_query
from src.hybrid_retriever import BiomedicalSparseIndex


def _resources() -> CRAGResources:
    sparse = BiomedicalSparseIndex()
    sparse._size = 1
    return CRAGResources(chroma_collection=object(), sparse_index=sparse)


def test_crag_mermaid_contains_key_nodes() -> None:
    graph = crag_mermaid()
    assert "Hybrid Retrieval" in graph
    assert "Query Correction" in graph
    assert "Judge Verification" in graph


def test_crag_happy_path_finalize(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.crag_pipeline.hybrid_search",
        lambda **kwargs: [
            {"id": "x1", "text": "Biomedical evidence", "metadata": {}, "score": 0.8, "source": "dense"}
        ],
    )
    monkeypatch.setattr(
        "src.crag_pipeline.grade_retrieval_quality",
        lambda query, docs: {"retrieval_quality": 0.9, "reason": "good", "missing_aspects": []},
    )
    monkeypatch.setattr(
        "src.crag_pipeline.grade_groundedness",
        lambda query, answer, context: {
            "groundedness": 0.95,
            "hallucination_risk": 0.05,
            "completeness": 0.9,
        },
    )
    monkeypatch.setattr(
        "src.crag_pipeline.ollama.chat",
        lambda **kwargs: {"message": {"content": "Grounded biomedical answer [1]."}},
    )

    app = build_crag_workflow(_resources())
    state = run_crag_query(app, "What does evidence report about diabetes?")
    assert state["final_answer"]
    assert "retrieve_hybrid" in state["trace"]
    assert "verify_answer" in state["trace"]
