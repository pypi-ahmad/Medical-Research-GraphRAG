"""ChromaDB indexing and retrieval helpers for the local GraphRAG workflow."""

from __future__ import annotations

import json
from typing import Any

import chromadb
from chromadb.config import Settings
from loguru import logger

from src.chunking import ChunkRecord
from src.config import settings
from src.embeddings import embed_query
from src.utils import timer


def create_chroma_client(path: str | None = None) -> chromadb.PersistentClient:
    """Create persistent Chroma client with telemetry disabled."""
    return chromadb.PersistentClient(
        path=path or str(settings.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )


def _metadata_from_chunk(chunk: ChunkRecord) -> dict[str, Any]:
    """Flatten chunk metadata into Chroma-supported scalar fields."""
    return {
        "pmid": chunk.pmid,
        "split": chunk.split,
        "chunk_index": int(chunk.chunk_index),
        "entity_count": int(chunk.entity_count),
        "concept_ids": "|".join(chunk.concept_ids),
        "entity_texts": "|".join(chunk.entity_texts),
        "title": chunk.title[:512],
    }


@timer
def create_or_replace_collection(
    client: chromadb.PersistentClient,
    collection_name: str,
) -> chromadb.Collection:
    """Create a fresh collection for deterministic runs."""
    existing = {c.name for c in client.list_collections()}
    if collection_name in existing:
        client.delete_collection(collection_name)

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


@timer
def index_chunks_to_chromadb(
    chunks: list[ChunkRecord],
    embeddings,
    collection_name: str = "medmentions_chroma",
    batch_size: int = 128,
) -> chromadb.Collection:
    """Index chunk embeddings and metadata into a Chroma collection."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding matrix rows must match")

    client = create_chroma_client()
    collection = create_or_replace_collection(client, collection_name)

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [_metadata_from_chunk(chunk) for chunk in chunks]

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        collection.add(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

    logger.info("Indexed {} chunks into Chroma collection {}", len(chunks), collection_name)
    return collection


def get_collection(collection_name: str = "medmentions_chroma") -> chromadb.Collection:
    """Fetch existing collection from persistent storage."""
    client = create_chroma_client()
    return client.get_collection(collection_name)


def vector_search(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Run vector similarity search with query embedding."""
    query_vec = embed_query(query)
    results = collection.query(
        query_embeddings=[query_vec.tolist()],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )

    items: list[dict[str, Any]] = []
    if not results.get("ids"):
        return items

    for i, chunk_id in enumerate(results["ids"][0]):
        distance = float(results["distances"][0][i]) if results.get("distances") else 1.0
        items.append(
            {
                "id": chunk_id,
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1.0 - distance,
                "source": "vector",
            }
        )
    return items


def entity_search(
    collection: chromadb.Collection,
    concept_ids: list[str],
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Retrieve chunks containing one or more concept IDs in metadata."""
    if not concept_ids:
        return []

    payload = collection.get(include=["metadatas", "documents"])

    results: list[dict[str, Any]] = []
    for idx, chunk_id in enumerate(payload["ids"]):
        meta = payload["metadatas"][idx]
        concepts_str = meta.get("concept_ids", "")
        matches = sum(1 for cid in concept_ids if cid and cid in concepts_str)
        if matches == 0:
            continue

        results.append(
            {
                "id": chunk_id,
                "text": payload["documents"][idx],
                "metadata": meta,
                "score": float(matches),
                "source": "entity",
            }
        )

    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]


def reciprocal_rank_fusion(
    result_sets: dict[str, list[dict[str, Any]]],
    top_k: int = 8,
    rrf_k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse ranked lists without requiring score normalization."""
    fused: dict[str, dict[str, Any]] = {}

    for source_name, items in result_sets.items():
        for rank, item in enumerate(items, start=1):
            chunk_id = item["id"]
            contribution = 1.0 / (rrf_k + rank)

            if chunk_id not in fused:
                fused[chunk_id] = {
                    "id": chunk_id,
                    "text": item["text"],
                    "metadata": item.get("metadata", {}),
                    "score": 0.0,
                    "sources": set(),
                }
            fused[chunk_id]["score"] += contribution
            fused[chunk_id]["sources"].add(source_name)

    ranked = sorted(fused.values(), key=lambda x: x["score"], reverse=True)
    for item in ranked:
        item["sources"] = sorted(item["sources"])

    return ranked[:top_k]
