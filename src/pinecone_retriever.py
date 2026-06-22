"""Pinecone indexing and retrieval for cloud GraphRAG benchmarking."""

from __future__ import annotations

from typing import Any

from loguru import logger
from pinecone import Pinecone, ServerlessSpec

from src.chunking import ChunkRecord
from src.config import settings
from src.embeddings import embed_query
from src.utils import timer


def _require_api_key() -> str:
    """Read Pinecone API key from settings/environment and fail fast if absent."""
    if not settings.pinecone_api_key:
        raise ValueError("PINECONE_API_KEY is required for Section B Pinecone benchmarking")
    return settings.pinecone_api_key


def pinecone_client() -> Pinecone:
    """Create Pinecone client."""
    return Pinecone(api_key=_require_api_key())


def _list_index_names(client: Pinecone) -> list[str]:
    """Version-compatible helper to get existing index names."""
    listing = client.list_indexes()

    if hasattr(listing, "names"):
        return list(listing.names())

    if isinstance(listing, dict) and "indexes" in listing:
        return [item["name"] for item in listing["indexes"]]

    names = []
    for item in listing:
        if isinstance(item, dict) and "name" in item:
            names.append(item["name"])
        elif hasattr(item, "name"):
            names.append(item.name)
    return names


@timer
def create_index(index_name: str, dimension: int, metric: str = "cosine") -> None:
    """Create Pinecone serverless index if it does not already exist."""
    client = pinecone_client()
    names = _list_index_names(client)
    if index_name in names:
        logger.info("Pinecone index {} already exists", index_name)
        return

    client.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    logger.info("Created Pinecone index {}", index_name)


@timer
def delete_index(index_name: str) -> None:
    """Delete Pinecone index for cost cleanup."""
    client = pinecone_client()
    names = _list_index_names(client)
    if index_name in names:
        client.delete_index(index_name)
        logger.info("Deleted Pinecone index {}", index_name)


@timer
def index_chunks_to_pinecone(
    chunks: list[ChunkRecord],
    embeddings,
    index_name: str,
    namespace: str = "default",
    batch_size: int = 100,
) -> None:
    """Upload chunk vectors with metadata into Pinecone."""
    if len(chunks) != len(embeddings):
        raise ValueError("Chunk count and embedding matrix rows must match")

    create_index(index_name=index_name, dimension=int(embeddings.shape[1]))

    client = pinecone_client()
    index = client.Index(index_name)

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        vectors = []
        for chunk, vector in zip(chunks[start:end], embeddings[start:end]):
            vectors.append(
                {
                    "id": chunk.chunk_id,
                    "values": vector.tolist(),
                    "metadata": {
                        "pmid": chunk.pmid,
                        "split": chunk.split,
                        "chunk_index": int(chunk.chunk_index),
                        "entity_count": int(chunk.entity_count),
                        "concept_ids": "|".join(chunk.concept_ids),
                        "text": chunk.text[:2000],
                    },
                }
            )

        index.upsert(vectors=vectors, namespace=namespace)

    logger.info("Indexed {} vectors to Pinecone index {}", len(chunks), index_name)


def query_pinecone(
    query: str,
    index_name: str,
    namespace: str = "default",
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Run vector search against Pinecone."""
    client = pinecone_client()
    index = client.Index(index_name)

    query_vector = embed_query(query)
    response = index.query(
        vector=query_vector.tolist(),
        top_k=top_k,
        include_metadata=True,
        namespace=namespace,
    )

    matches = response.get("matches", []) if isinstance(response, dict) else response.matches

    items: list[dict[str, Any]] = []
    for match in matches:
        metadata = match.get("metadata", {}) if isinstance(match, dict) else match.metadata
        match_id = match.get("id") if isinstance(match, dict) else match.id
        score = match.get("score") if isinstance(match, dict) else match.score

        items.append(
            {
                "id": match_id,
                "text": metadata.get("text", ""),
                "metadata": metadata,
                "score": float(score),
                "source": "pinecone",
            }
        )

    return items


def pinecone_cost_proxy(
    index_name: str,
    query_count: int,
    upsert_count: int,
    namespace: str = "default",
) -> dict[str, Any]:
    """Return workload-driven cost proxy fields for reproducible reporting.

    Direct dollar cost depends on plan and region. We therefore report real
    operation volumes and index stats to support transparent cost reasoning.
    """
    client = pinecone_client()
    index = client.Index(index_name)
    stats = index.describe_index_stats()

    namespaces = {}
    if isinstance(stats, dict):
        namespaces = stats.get("namespaces", {})
        total_vectors = stats.get("total_vector_count", 0)
    else:
        namespaces = getattr(stats, "namespaces", {})
        total_vectors = getattr(stats, "total_vector_count", 0)

    namespace_vectors = 0
    if isinstance(namespaces, dict) and namespace in namespaces:
        ns_obj = namespaces[namespace]
        if isinstance(ns_obj, dict):
            namespace_vectors = ns_obj.get("vector_count", 0)
        else:
            namespace_vectors = getattr(ns_obj, "vector_count", 0)

    return {
        "query_count": int(query_count),
        "upsert_count": int(upsert_count),
        "total_vectors": int(total_vectors),
        "namespace_vectors": int(namespace_vectors),
        "pricing_note": (
            "USD cost varies by Pinecone plan/region; report operation counts and vector footprint "
            "as auditable cost drivers."
        ),
    }
