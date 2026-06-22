"""Hybrid RAG retrieval (dense + sparse) for biomedical text."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from loguru import logger

from src.chroma_retriever import reciprocal_rank_fusion, vector_search
from src.chunking import ChunkRecord
from src.config import settings
from src.utils import timer


@dataclass(slots=True)
class SparseResult:
    """Sparse retriever row."""

    id: str
    score: float
    text: str
    metadata: dict[str, Any]
    source: str = "sparse"


def _tokenize(text: str) -> list[str]:
    """Tokenize biomedical text into lowercase lexical terms."""
    return re.findall(r"[a-z0-9][a-z0-9\\-]+", text.lower())


BIOMED_ABBREVIATION_MAP: dict[str, str] = {
    "htn": "hypertension",
    "dm": "diabetes mellitus",
    "ckd": "chronic kidney disease",
    "mi": "myocardial infarction",
    "copd": "chronic obstructive pulmonary disease",
    "cad": "coronary artery disease",
}


def _expand_biomedical_query_terms(tokens: list[str]) -> list[str]:
    """Expand a subset of common biomedical abbreviations."""
    expanded = list(tokens)
    for token in tokens:
        phrase = BIOMED_ABBREVIATION_MAP.get(token)
        if phrase:
            expanded.extend(_tokenize(phrase))
    return expanded


class BiomedicalSparseIndex:
    """BM25-style sparse index for chunk-level biomedical retrieval."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.doc_lengths: dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.term_freqs: dict[str, Counter[str]] = {}
        self.doc_payload: dict[str, dict[str, Any]] = {}
        self._size = 0

    @property
    def size(self) -> int:
        return self._size

    @timer
    def fit(self, chunks: list[ChunkRecord]) -> None:
        """Build sparse index structures from chunk text."""
        self.doc_freq.clear()
        self.doc_lengths.clear()
        self.term_freqs.clear()
        self.doc_payload.clear()

        total_len = 0
        for chunk in chunks:
            tokens = _tokenize(chunk.text)
            if not tokens:
                continue

            doc_id = chunk.chunk_id
            tf = Counter(tokens)
            self.term_freqs[doc_id] = tf
            self.doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)

            self.doc_payload[doc_id] = {
                "id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": {
                    "pmid": chunk.pmid,
                    "split": chunk.split,
                    "chunk_index": int(chunk.chunk_index),
                    "entity_count": int(chunk.entity_count),
                    "concept_ids": "|".join(chunk.concept_ids),
                    "entity_texts": "|".join(chunk.entity_texts),
                },
            }

            for term in tf.keys():
                self.doc_freq[term] += 1

        self._size = len(self.term_freqs)
        self.avg_doc_len = (total_len / self._size) if self._size else 0.0
        logger.info("Built sparse biomedical index over {} documents", self._size)

    def _idf(self, term: str) -> float:
        """Compute BM25 IDF with +1 smoothing for numeric stability."""
        n = self._size
        df = self.doc_freq.get(term, 0)
        if n == 0:
            return 0.0
        return math.log(1.0 + ((n - df + 0.5) / (df + 0.5)))

    def search(self, query: str, top_k: int = 8) -> list[dict[str, Any]]:
        """Run sparse BM25 retrieval for a biomedical query."""
        if self._size == 0:
            return []

        q_tokens = _expand_biomedical_query_terms(_tokenize(query))
        if not q_tokens:
            return []

        q_terms = Counter(q_tokens)
        scores: dict[str, float] = defaultdict(float)

        for doc_id, tf in self.term_freqs.items():
            doc_len = self.doc_lengths[doc_id]
            norm = self.k1 * (1 - self.b + self.b * (doc_len / max(self.avg_doc_len, 1e-8)))

            score = 0.0
            for term, qtf in q_terms.items():
                f = tf.get(term, 0)
                if f <= 0:
                    continue
                idf = self._idf(term)
                numer = f * (self.k1 + 1.0)
                denom = f + norm
                score += qtf * idf * (numer / max(denom, 1e-8))

            if score > 0:
                scores[doc_id] = score

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        rows: list[dict[str, Any]] = []
        for doc_id, score in ranked:
            payload = self.doc_payload[doc_id]
            rows.append(
                {
                    "id": payload["id"],
                    "text": payload["text"],
                    "metadata": payload["metadata"],
                    "score": float(score),
                    "source": "sparse",
                }
            )
        return rows


def weighted_score_fusion(
    dense_rows: list[dict[str, Any]],
    sparse_rows: list[dict[str, Any]],
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Fuse dense and sparse scores with min-max normalization."""
    dense_weight = settings.hybrid_dense_weight if dense_weight is None else dense_weight
    sparse_weight = settings.hybrid_sparse_weight if sparse_weight is None else sparse_weight
    total = dense_weight + sparse_weight
    if total <= 0:
        dense_weight, sparse_weight = 0.5, 0.5
    else:
        dense_weight /= total
        sparse_weight /= total

    def normalize(rows: list[dict[str, Any]]) -> dict[str, float]:
        if not rows:
            return {}
        values = [float(row.get("score", 0.0)) for row in rows]
        lo, hi = min(values), max(values)
        if hi - lo < 1e-12:
            return {row["id"]: 1.0 for row in rows}
        return {row["id"]: (float(row.get("score", 0.0)) - lo) / (hi - lo) for row in rows}

    dense_norm = normalize(dense_rows)
    sparse_norm = normalize(sparse_rows)
    by_id: dict[str, dict[str, Any]] = {}

    for row in dense_rows + sparse_rows:
        doc_id = row["id"]
        if doc_id not in by_id:
            by_id[doc_id] = {
                "id": doc_id,
                "text": row.get("text", ""),
                "metadata": row.get("metadata", {}),
                "score": 0.0,
                "sources": set(),
                "dense_norm": 0.0,
                "sparse_norm": 0.0,
            }
        by_id[doc_id]["sources"].add(row.get("source", "unknown"))

    for doc_id, item in by_id.items():
        d = dense_norm.get(doc_id, 0.0)
        s = sparse_norm.get(doc_id, 0.0)
        item["dense_norm"] = float(d)
        item["sparse_norm"] = float(s)
        item["score"] = float(dense_weight * d + sparse_weight * s)
        item["sources"] = sorted(item["sources"])

    ranked = sorted(by_id.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return ranked


def hybrid_search(
    *,
    collection: Any,
    sparse_index: BiomedicalSparseIndex,
    query: str,
    top_k: int = 8,
    dense_weight: float | None = None,
    sparse_weight: float | None = None,
    use_rrf: bool = False,
) -> list[dict[str, Any]]:
    """Run hybrid biomedical retrieval combining dense and sparse channels."""
    dense_rows = vector_search(collection, query, top_k=top_k * 2)
    sparse_rows = sparse_index.search(query, top_k=top_k * 2)

    if use_rrf:
        return reciprocal_rank_fusion(
            {"dense": dense_rows, "sparse": sparse_rows},
            top_k=top_k,
        )

    fused = weighted_score_fusion(
        dense_rows=dense_rows,
        sparse_rows=sparse_rows,
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        top_k=top_k,
    )
    return fused
