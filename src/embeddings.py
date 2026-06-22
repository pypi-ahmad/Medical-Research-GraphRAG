"""Embedding utilities for local Ollama embedding model usage."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import os
import time

import numpy as np
import ollama
from loguru import logger

from src.config import settings
from src.utils import timer


@dataclass(slots=True)
class EmbeddingBundle:
    """Container for embedding matrix and aligned chunk IDs."""

    chunk_ids: list[str]
    matrix: np.ndarray


def _normalize_rows(vectors: np.ndarray) -> np.ndarray:
    """Normalize vectors row-wise for cosine-compatible dot products."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


@timer
def embed_texts(
    texts: Sequence[str],
    model: str | None = None,
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    """Embed many texts with Ollama in batches."""
    model_name = model or settings.embedding_model

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    client = ollama.Client(
        host=os.getenv("OLLAMA_HOST"),
        timeout=float(os.getenv("OLLAMA_EMBED_TIMEOUT_SECONDS", "900")),
    )

    vectors: list[list[float]] = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for batch_index, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = list(texts[start : start + batch_size])

        response = None
        last_exc: Exception | None = None
        for attempt in range(1, 4):
            try:
                response = client.embed(model=model_name, input=batch)
                break
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                logger.warning(
                    "Embedding batch {}/{} attempt {} failed: {}",
                    batch_index,
                    total_batches,
                    attempt,
                    exc,
                )
                time.sleep(min(5, attempt * 1.5))

        if response is None:
            # Fallback path: embed each item independently if batch request fails.
            if len(batch) > 1:
                logger.warning(
                    "Falling back to per-item embedding for batch {}/{}",
                    batch_index,
                    total_batches,
                )
                try:
                    fallback_dim = int(load_embedding_bundle().matrix.shape[1])
                except Exception:
                    fallback_dim = 2560
                batch_vectors = []
                for text in batch:
                    try:
                        single_response = client.embed(model=model_name, input=[text])
                        single_vectors = (
                            single_response.get("embeddings", [])
                            if hasattr(single_response, "get")
                            else []
                        )
                        if not single_vectors and hasattr(single_response, "embeddings"):
                            single_vectors = single_response.embeddings
                        if single_vectors:
                            batch_vectors.extend(single_vectors)
                        else:
                            logger.warning(
                                "Per-item embedding returned empty payload; using zero-vector fallback (dim={})",
                                fallback_dim,
                            )
                            batch_vectors.append([0.0] * fallback_dim)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning(
                            "Per-item embedding failed; using zero-vector fallback (dim={}): {}",
                            fallback_dim,
                            exc,
                        )
                        batch_vectors.append([0.0] * fallback_dim)
                vectors.extend(batch_vectors)
                continue

            # Single-item fallback: keep pipeline alive with a deterministic
            # zero vector when the embedding service repeatedly times out.
            fallback_dim = None
            try:
                fallback_dim = int(load_embedding_bundle().matrix.shape[1])
            except Exception:
                fallback_dim = 2560
            logger.error(
                "Embedding failed permanently for batch {}/{}; using zero-vector fallback (dim={})",
                batch_index,
                total_batches,
                fallback_dim,
            )
            vectors.append([0.0] * fallback_dim)
            continue

        batch_vectors = response.get("embeddings", [])
        if not batch_vectors and hasattr(response, "embeddings"):
            batch_vectors = response.embeddings

        vectors.extend(batch_vectors)
        if batch_index % 10 == 0 or batch_index == total_batches:
            logger.info(
                "Embedded batch {}/{} ({} texts)",
                batch_index,
                total_batches,
                len(vectors),
            )

    matrix = np.asarray(vectors, dtype=np.float32)
    if normalize and matrix.size:
        matrix = _normalize_rows(matrix)
    logger.info("Embedded {} texts with model {}", len(texts), model_name)
    return matrix


def embed_query(query: str, model: str | None = None) -> np.ndarray:
    """Embed one query string with the configured embedding model."""
    matrix = embed_texts([query], model=model, batch_size=1, normalize=True)
    return matrix[0]


def get_embedding_dimension(model: str | None = None) -> int:
    """Infer embedding dimension from a test query."""
    vec = embed_query("dimension check", model=model)
    return int(vec.shape[0])


class OllamaEmbeddingFunction:
    """Chroma-compatible embedding function wrapper."""

    def __init__(self, model: str | None = None):
        self.model = model or settings.embedding_model

    def __call__(self, input: list[str]) -> list[list[float]]:
        matrix = embed_texts(input, model=self.model, batch_size=max(1, len(input)), normalize=True)
        return matrix.tolist()


@timer
def persist_embedding_bundle(bundle: EmbeddingBundle, out_dir: Path | None = None) -> tuple[Path, Path]:
    """Persist embedding matrix and aligned IDs for reproducible indexing."""
    root = out_dir or settings.processed_dir
    root.mkdir(parents=True, exist_ok=True)

    matrix_path = root / "chunk_embeddings.npy"
    ids_path = root / "chunk_embedding_ids.json"

    np.save(matrix_path, bundle.matrix)
    ids_path.write_text(__import__("json").dumps(bundle.chunk_ids, ensure_ascii=True, indent=2), encoding="utf-8")

    logger.info("Saved embeddings to {} and {}", matrix_path, ids_path)
    return matrix_path, ids_path


def load_embedding_bundle(root: Path | None = None) -> EmbeddingBundle:
    """Load persisted embeddings and aligned chunk IDs."""
    base = root or settings.processed_dir
    matrix = np.load(base / "chunk_embeddings.npy")
    chunk_ids = __import__("json").loads((base / "chunk_embedding_ids.json").read_text(encoding="utf-8"))
    return EmbeddingBundle(chunk_ids=chunk_ids, matrix=matrix)
