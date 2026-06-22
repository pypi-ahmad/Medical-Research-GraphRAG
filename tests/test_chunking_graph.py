"""Unit tests for chunking and graph construction behavior."""

from src.chunking import build_chunks
from src.data_pipeline import EntityMention, MedRecord
from src.graph_builder import build_entity_graph, local_graph_expansion


def _sample_record() -> MedRecord:
    return MedRecord(
        pmid="1",
        split="train",
        title="Diabetes study",
        abstract="Diabetes is linked to insulin resistance.",
        text="Diabetes study\n\nDiabetes is linked to insulin resistance.",
        entities=[
            EntityMention(
                text="Diabetes",
                concept_id="C0011849",
                semantic_type_ids=["T047"],
                offsets=[[0, 8]],
            ),
            EntityMention(
                text="insulin",
                concept_id="C0021641",
                semantic_type_ids=["T116"],
                offsets=[[26, 33]],
            ),
        ],
    )


def test_build_chunks_retains_entities() -> None:
    chunks = build_chunks([_sample_record()], chunk_size=200, chunk_overlap=20)
    assert len(chunks) >= 1
    assert chunks[0].entity_count >= 1
    assert "C0011849" in chunks[0].concept_ids


def test_graph_and_local_expansion() -> None:
    chunks = build_chunks([_sample_record()], chunk_size=200, chunk_overlap=20)
    graph = build_entity_graph(chunks, min_entity_frequency=1, min_edge_weight=1)

    neighborhood = local_graph_expansion(graph, ["C0011849"], hops=1)
    assert "nodes" in neighborhood
    assert "edges" in neighborhood
    assert "C0011849" in neighborhood["nodes"]
