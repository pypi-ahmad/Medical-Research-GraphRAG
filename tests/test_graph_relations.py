"""Tests for relation extraction and relation-aware graph enrichment."""

from src.chunking import ChunkRecord
from src.graph_builder import build_entity_graph, extract_relationship_edges, relation_edge_statistics


def _sample_chunks() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="pm1_c0000",
            pmid="pm1",
            split="train",
            chunk_index=0,
            text="Insulin resistance is associated with diabetes in this cohort.",
            title="Sample 1",
            entity_count=2,
            concept_ids=["C0021641", "C0011849"],
            entity_texts=["Insulin", "Diabetes"],
        ),
        ChunkRecord(
            chunk_id="pm2_c0000",
            pmid="pm2",
            split="train",
            chunk_index=0,
            text="Metformin treats diabetes and improves glycemic control.",
            title="Sample 2",
            entity_count=2,
            concept_ids=["C0025598", "C0011849"],
            entity_texts=["Metformin", "Diabetes"],
        ),
    ]


def test_extract_relationship_edges_labels() -> None:
    edges = extract_relationship_edges(_sample_chunks())
    assert len(edges) >= 2
    labels = {edge.relation for edge in edges}
    assert "associated_with" in labels
    assert "treats" in labels


def test_relation_aware_graph_metadata() -> None:
    chunks = _sample_chunks()
    relation_edges = extract_relationship_edges(chunks)
    graph = build_entity_graph(
        chunks,
        min_entity_frequency=1,
        min_edge_weight=1,
        relation_edges=relation_edges,
    )
    assert graph.number_of_nodes() >= 3
    assert graph.number_of_edges() >= 2

    has_relation_metadata = False
    for _, _, data in graph.edges(data=True):
        if "relation_count" in data:
            has_relation_metadata = True
            assert data["relation_count"] >= 1
            assert isinstance(data.get("relation_labels"), list)
    assert has_relation_metadata


def test_relation_edge_statistics_shape() -> None:
    stats = relation_edge_statistics(extract_relationship_edges(_sample_chunks()))
    assert stats["total_edges"] >= 2
    assert "relation_type_counts" in stats
