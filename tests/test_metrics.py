"""Unit tests for deterministic evaluation metric helpers."""

from src.evaluator import (
    RetrievalExample,
    compute_retrieval_metrics,
    exact_match,
    ndcg_at_k,
    reciprocal_rank,
)


def test_exact_match_normalization() -> None:
    assert exact_match("Hypertension.", "hypertension") == 1.0
    assert exact_match("Diabetes", "Hypertension") == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank([0, 1, 0]) == 0.5
    assert reciprocal_rank([0, 0, 0]) == 0.0


def test_ndcg_range() -> None:
    score = ndcg_at_k([1, 0, 1, 0], k=4)
    assert 0.0 <= score <= 1.0


def test_compute_retrieval_metrics_expected_keys() -> None:
    examples = [
        RetrievalExample(retrieved_ids=["a", "b", "c"], relevant_ids=["a", "z"]),
        RetrievalExample(retrieved_ids=["x", "y", "z"], relevant_ids=["z"]),
    ]

    metrics = compute_retrieval_metrics(examples, k_values=[1, 3])

    for key in ["precision@1", "recall@1", "f1@1", "ndcg@1", "precision@3", "mrr"]:
        assert key in metrics
        assert metrics[key] >= 0.0
