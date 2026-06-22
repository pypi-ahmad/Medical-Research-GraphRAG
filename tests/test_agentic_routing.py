"""Tests for agentic routing behavior in LangGraph workflow."""

from __future__ import annotations

import networkx as nx

from src.agentic_rag import AgentResources, build_agentic_workflow, run_agentic_query
from src.chunking import ChunkRecord


def _resources() -> AgentResources:
    graph = nx.Graph()
    graph.add_node("C0011849", label="Diabetes", frequency=10)
    graph.add_node("C0021641", label="Insulin", frequency=8)
    graph.add_edge("C0011849", "C0021641", weight=3)

    chunks = [
        ChunkRecord(
            chunk_id="pm1_c0000",
            pmid="pm1",
            split="train",
            chunk_index=0,
            text="Diabetes is associated with insulin resistance.",
            title="Demo",
            entity_count=2,
            concept_ids=["C0011849", "C0021641"],
            entity_texts=["Diabetes", "Insulin"],
        )
    ]

    summaries = [
        {
            "community_id": 0,
            "size": 2,
            "num_edges": 1,
            "top_entities": [
                {
                    "concept_id": "C0011849",
                    "label": "Diabetes",
                    "centrality": 1.0,
                    "frequency": 10,
                }
            ],
        }
    ]

    return AgentResources(
        chroma_collection=object(),
        chunks=chunks,
        graph=graph,
        partition={"C0011849": 0, "C0021641": 0},
        summaries=summaries,
    )


def _mock_chat(*args, **kwargs):  # noqa: ANN002, ANN003
    return {"message": {"content": "Grounded biomedical answer [1]."}}


def test_route_to_web_fallback(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agentic_rag.vector_search",
        lambda *args, **kwargs: [{"id": "x", "text": "Doc", "metadata": {}, "score": 0.1}],
    )
    monkeypatch.setattr("src.agentic_rag._judge_retrieval", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr("src.agentic_rag._hallucination_score", lambda *args, **kwargs: 0.95)
    monkeypatch.setattr(
        "src.agentic_rag._web_search",
        lambda *args, **kwargs: [{"title": "Result", "snippet": "Evidence", "url": "https://x.test"}],
    )
    monkeypatch.setattr("src.agentic_rag.ollama.chat", _mock_chat)

    app = build_agentic_workflow(_resources())
    state = run_agentic_query(app, "What about diabetes and insulin resistance?")
    assert state["route"] == "web_fallback"
    assert "web_search" in state["trace"]
    assert state["final_answer"]


def test_route_direct_to_graph(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.agentic_rag.vector_search",
        lambda *args, **kwargs: [{"id": "x", "text": "Doc", "metadata": {}, "score": 0.9}],
    )
    monkeypatch.setattr("src.agentic_rag._judge_retrieval", lambda *args, **kwargs: 0.9)
    monkeypatch.setattr("src.agentic_rag._hallucination_score", lambda *args, **kwargs: 0.95)
    monkeypatch.setattr("src.agentic_rag._web_search", lambda *args, **kwargs: [])
    monkeypatch.setattr("src.agentic_rag.ollama.chat", _mock_chat)

    app = build_agentic_workflow(_resources())
    state = run_agentic_query(app, "What about diabetes and insulin resistance?")
    assert state["route"] == "graph_traversal"
    assert "graph_traversal" in state["trace"]
