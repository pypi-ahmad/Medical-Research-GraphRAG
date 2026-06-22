"""Tests for additive biomedical hybrid retriever."""

from src.chunking import ChunkRecord
from src.hybrid_retriever import BiomedicalSparseIndex, weighted_score_fusion


def _chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="c1",
            pmid="p1",
            split="train",
            chunk_index=0,
            text="Diabetes mellitus is associated with insulin resistance.",
            title="A",
            entity_count=2,
            concept_ids=["C0011849", "C0021641"],
            entity_texts=["Diabetes", "Insulin"],
        ),
        ChunkRecord(
            chunk_id="c2",
            pmid="p2",
            split="train",
            chunk_index=0,
            text="Hypertension treatment includes lifestyle modification.",
            title="B",
            entity_count=1,
            concept_ids=["C0020538"],
            entity_texts=["Hypertension"],
        ),
    ]


def test_sparse_index_returns_ranked_results() -> None:
    index = BiomedicalSparseIndex()
    index.fit(_chunks())
    rows = index.search("diabetes insulin", top_k=3)
    assert rows
    assert rows[0]["id"] == "c1"
    assert rows[0]["score"] > 0


def test_sparse_abbreviation_expansion() -> None:
    index = BiomedicalSparseIndex()
    index.fit(_chunks())
    rows = index.search("dm", top_k=3)
    assert rows
    assert rows[0]["id"] == "c1"


def test_weighted_score_fusion_shape() -> None:
    dense = [
        {"id": "c1", "text": "A", "metadata": {}, "score": 0.9, "source": "dense"},
        {"id": "c2", "text": "B", "metadata": {}, "score": 0.3, "source": "dense"},
    ]
    sparse = [
        {"id": "c2", "text": "B", "metadata": {}, "score": 2.0, "source": "sparse"},
        {"id": "c3", "text": "C", "metadata": {}, "score": 1.0, "source": "sparse"},
    ]
    fused = weighted_score_fusion(dense, sparse, dense_weight=0.6, sparse_weight=0.4, top_k=5)
    assert fused
    ids = {row["id"] for row in fused}
    assert {"c1", "c2", "c3"}.issubset(ids)
