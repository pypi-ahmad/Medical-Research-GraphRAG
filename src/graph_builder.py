"""Biomedical knowledge graph construction and GraphRAG search primitives."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from typing import Any

import community as community_louvain
import networkx as nx
import numpy as np
from loguru import logger

from src.chunking import ChunkRecord
from src.utils import timer


@dataclass(slots=True)
class RelationEdge:
    """Typed relation evidence extracted from a chunk-level entity pair."""

    source_concept_id: str
    target_concept_id: str
    relation: str
    evidence_chunk_id: str
    evidence_text: str


RELATION_PATTERNS: dict[str, list[str]] = {
    "associated_with": [r"\bassociated with\b", r"\bcorrelated with\b", r"\blinked to\b"],
    "inhibits": [r"\binhibits?\b", r"\bsuppresses?\b", r"\bblocks?\b"],
    "activates": [r"\bactivates?\b", r"\bstimulates?\b", r"\binduces?\b"],
    "treats": [r"\btreats?\b", r"\btherapy for\b", r"\bused for\b"],
    "causes": [r"\bcauses?\b", r"\bleads to\b", r"\bresults in\b"],
}


def _infer_relation_label(text: str) -> str:
    """Infer a lightweight relation label from evidence text."""
    lowered = text.lower()
    for relation, patterns in RELATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, lowered):
                return relation
    return "co_occurs_with"


def _concept_label_lookup(chunks: list[ChunkRecord]) -> dict[str, str]:
    """Build concept-id to readable entity label map from chunk metadata."""
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    for chunk in chunks:
        for concept_id, entity_text in zip(chunk.concept_ids, chunk.entity_texts):
            labels[concept_id][entity_text] += 1

    return {
        concept_id: counter.most_common(1)[0][0]
        for concept_id, counter in labels.items()
        if counter
    }


@timer
def build_entity_graph(
    chunks: list[ChunkRecord],
    min_entity_frequency: int = 2,
    min_edge_weight: int = 1,
    relation_edges: list[RelationEdge] | None = None,
) -> nx.Graph:
    """Build weighted co-occurrence graph from chunk-level entity annotations."""
    graph = nx.Graph()
    concept_counts: Counter[str] = Counter()
    edge_counts: Counter[tuple[str, str]] = Counter()

    label_lookup = _concept_label_lookup(chunks)

    for chunk in chunks:
        concepts = sorted(set(chunk.concept_ids))
        if not concepts:
            continue
        concept_counts.update(concepts)

        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                edge_counts[(concepts[i], concepts[j])] += 1

    valid_concepts = {
        cid for cid, count in concept_counts.items() if count >= min_entity_frequency
    }

    for concept_id in valid_concepts:
        graph.add_node(
            concept_id,
            label=label_lookup.get(concept_id, concept_id),
            frequency=int(concept_counts[concept_id]),
        )

    for (c1, c2), weight in edge_counts.items():
        if c1 not in valid_concepts or c2 not in valid_concepts:
            continue
        if weight < min_edge_weight:
            continue
        graph.add_edge(c1, c2, weight=int(weight))

    relation_edges = relation_edges or []
    relation_groups: dict[tuple[str, str], list[RelationEdge]] = defaultdict(list)
    for edge in relation_edges:
        key = tuple(sorted([edge.source_concept_id, edge.target_concept_id]))
        relation_groups[key].append(edge)

    for (c1, c2), group in relation_groups.items():
        if c1 not in graph or c2 not in graph:
            continue
        if not graph.has_edge(c1, c2):
            graph.add_edge(c1, c2, weight=1)

        relation_counter = Counter(item.relation for item in group)
        graph[c1][c2]["relation_count"] = int(len(group))
        graph[c1][c2]["relation_labels"] = sorted(relation_counter.keys())
        graph[c1][c2]["primary_relation"] = relation_counter.most_common(1)[0][0]

    logger.info(
        "Graph built with {} nodes and {} edges",
        graph.number_of_nodes(),
        graph.number_of_edges(),
    )
    return graph


def graph_statistics(graph: nx.Graph) -> dict[str, float]:
    """Compute high-signal graph-level diagnostics."""
    if graph.number_of_nodes() == 0:
        return {}

    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)

    components = list(nx.connected_components(graph))
    largest_component = max((len(c) for c in components), default=0)

    return {
        "nodes": int(graph.number_of_nodes()),
        "edges": int(graph.number_of_edges()),
        "density": float(nx.density(graph)),
        "num_components": int(len(components)),
        "largest_component": int(largest_component),
        "avg_degree": float(np.mean(degrees)),
        "median_degree": float(np.median(degrees)),
    }


@timer
def detect_communities(graph: nx.Graph, resolution: float = 1.0) -> dict[str, int]:
    """Run Louvain community detection and return node->community mapping."""
    if graph.number_of_nodes() == 0:
        return {}

    partition = community_louvain.best_partition(graph, resolution=resolution, random_state=42)
    logger.info("Detected {} communities", len(set(partition.values())))
    return {str(node): int(cid) for node, cid in partition.items()}


def community_summaries(
    graph: nx.Graph,
    partition: dict[str, int],
    top_entities: int = 10,
) -> list[dict[str, Any]]:
    """Summarize communities for global GraphRAG search context."""
    grouped: dict[int, list[str]] = defaultdict(list)
    for node, cid in partition.items():
        if node in graph:
            grouped[cid].append(node)

    summaries: list[dict[str, Any]] = []
    for community_id, members in grouped.items():
        subgraph = graph.subgraph(members)
        centrality = nx.degree_centrality(subgraph)
        ranked = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:top_entities]

        summaries.append(
            {
                "community_id": int(community_id),
                "size": int(len(members)),
                "num_edges": int(subgraph.number_of_edges()),
                "top_entities": [
                    {
                        "concept_id": concept_id,
                        "label": graph.nodes[concept_id].get("label", concept_id),
                        "centrality": float(score),
                        "frequency": int(graph.nodes[concept_id].get("frequency", 0)),
                    }
                    for concept_id, score in ranked
                ],
            }
        )

    summaries.sort(key=lambda item: item["size"], reverse=True)
    return summaries


def local_graph_expansion(
    graph: nx.Graph,
    concept_ids: list[str],
    hops: int = 2,
) -> dict[str, Any]:
    """Expand neighborhood around query concepts for local GraphRAG search."""
    seeds = [cid for cid in concept_ids if cid in graph]
    if not seeds:
        return {"nodes": [], "edges": []}

    visited = set(seeds)
    frontier = set(seeds)

    for _ in range(max(1, hops)):
        next_frontier = set()
        for node in frontier:
            next_frontier.update(graph.neighbors(node))
        next_frontier -= visited
        if not next_frontier:
            break
        visited.update(next_frontier)
        frontier = next_frontier

    subgraph = graph.subgraph(visited)
    edges = [
        (u, v, float(subgraph[u][v].get("weight", 1.0)))
        for u, v in subgraph.edges()
    ]

    return {
        "nodes": list(subgraph.nodes()),
        "edges": edges,
    }


def rank_communities_for_query(
    concept_ids: list[str],
    partition: dict[str, int],
    summaries: list[dict[str, Any]],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Select most relevant communities for global GraphRAG context."""
    if not concept_ids:
        return summaries[:top_k]

    community_hits = Counter()
    for concept_id in concept_ids:
        if concept_id in partition:
            community_hits[partition[concept_id]] += 1

    if not community_hits:
        return summaries[:top_k]

    summary_lookup = {item["community_id"]: item for item in summaries}
    ranked = sorted(
        community_hits.items(),
        key=lambda x: x[1],
        reverse=True,
    )
    selected = [summary_lookup[cid] for cid, _ in ranked if cid in summary_lookup]
    return selected[:top_k]


def community_summary_text(summary: dict[str, Any]) -> str:
    """Render a concise text summary for LLM context injection."""
    lines = [
        f"Community {summary['community_id']} (size={summary['size']}, edges={summary['num_edges']}):",
        "Key biomedical concepts:",
    ]
    for entity in summary.get("top_entities", []):
        lines.append(
            f"- {entity['label']} (concept_id={entity['concept_id']}, freq={entity['frequency']}, centrality={entity['centrality']:.3f})"
        )
    return "\n".join(lines)


def extract_relationship_edges(
    chunks: list[ChunkRecord],
    max_evidence_chars: int = 320,
) -> list[RelationEdge]:
    """Extract pairwise relation evidence from chunk entity co-occurrence.

    This is a pragmatic heuristic extractor intended for GraphRAG context
    expansion and explainability, not a clinical relation-extraction model.
    """
    extracted: list[RelationEdge] = []

    for chunk in chunks:
        if len(chunk.concept_ids) < 2:
            continue

        unique_pairs: set[tuple[str, str]] = set()
        concepts = list(dict.fromkeys(chunk.concept_ids))
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                c1, c2 = concepts[i], concepts[j]
                pair = tuple(sorted([c1, c2]))
                if pair in unique_pairs:
                    continue
                unique_pairs.add(pair)

                extracted.append(
                    RelationEdge(
                        source_concept_id=pair[0],
                        target_concept_id=pair[1],
                        relation=_infer_relation_label(chunk.text),
                        evidence_chunk_id=chunk.chunk_id,
                        evidence_text=chunk.text[:max_evidence_chars],
                    )
                )

    logger.info("Extracted {} relationship evidence edges", len(extracted))
    return extracted


def relation_edge_statistics(relation_edges: list[RelationEdge]) -> dict[str, Any]:
    """Summarize extracted relationship evidence for notebook reporting."""
    if not relation_edges:
        return {"total_edges": 0, "relation_type_counts": {}}

    relation_counts = Counter(edge.relation for edge in relation_edges)
    return {
        "total_edges": int(len(relation_edges)),
        "relation_type_counts": {key: int(value) for key, value in relation_counts.items()},
    }


def serialize_relation_edges(relation_edges: list[RelationEdge]) -> list[dict[str, Any]]:
    """Convert typed relation edges into JSON-serializable dictionaries."""
    return [asdict(edge) for edge in relation_edges]


def concept_id_from_query(query: str, chunks: list[ChunkRecord], max_ids: int = 8) -> list[str]:
    """Map query text to known concept IDs using mention-string matching."""
    query_lower = query.lower()
    matches = []

    for chunk in chunks:
        for concept_id, text in zip(chunk.concept_ids, chunk.entity_texts):
            if text.lower() in query_lower:
                matches.append(concept_id)

    if not matches:
        return []
    counts = Counter(matches)
    return [concept_id for concept_id, _ in counts.most_common(max_ids)]
