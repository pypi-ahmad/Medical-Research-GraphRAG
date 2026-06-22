"""Tests for unified evaluation payload assembly."""

from src.evaluator import (
    GenerationExample,
    RetrievalExample,
    build_evaluation_bundle,
)


def _retrieval_examples() -> list[RetrievalExample]:
    return [
        RetrievalExample(retrieved_ids=["a", "b", "c"], relevant_ids=["a", "z"]),
        RetrievalExample(retrieved_ids=["x", "y", "z"], relevant_ids=["z"]),
    ]


def _generation_examples() -> list[GenerationExample]:
    return [
        GenerationExample(
            query="What is diabetes?",
            answer="Diabetes is a metabolic condition.",
            reference_answer="Diabetes is a metabolic condition.",
            context_chunks=["Diabetes is a metabolic condition."],
        )
    ]


def test_build_evaluation_bundle_metadata_defaults() -> None:
    bundle = build_evaluation_bundle(
        _retrieval_examples(),
        _generation_examples(),
        include_bertscore=False,
    )
    payload = bundle.to_dict()
    assert "retrieval_metrics" in payload
    assert "generation_metrics" in payload
    assert "rag_metrics" in payload
    assert payload["metadata"]["retrieval_example_count"] == 2
    assert payload["metadata"]["generation_example_count"] == 1
    assert payload["metadata"]["bertscore_enabled"] is False


def test_build_evaluation_bundle_respects_k_values() -> None:
    bundle = build_evaluation_bundle(
        _retrieval_examples(),
        _generation_examples(),
        k_values=[1, 3],
        include_bertscore=False,
    )
    metrics = bundle.retrieval_metrics
    assert "precision@1" in metrics
    assert "precision@3" in metrics
    assert "precision@5" not in metrics
