"""Chunking pipeline with entity-aware metadata propagation for GraphRAG."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings
from src.data_pipeline import EntityMention, MedRecord
from src.utils import timer


DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "; ", ", ", " ", ""]


@dataclass(slots=True)
class ChunkRecord:
    """Single retrievable chunk used by vector and graph retrieval."""

    chunk_id: str
    pmid: str
    split: str
    chunk_index: int
    text: str
    title: str
    entity_count: int
    concept_ids: list[str]
    entity_texts: list[str]


def recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str] | None = None,
) -> list[str]:
    """Split text using boundary-aware recursive splitting."""
    if not text:
        return []

    separators = separators or DEFAULT_SEPARATORS
    chunk_size = max(int(chunk_size), 200)
    chunk_overlap = max(0, min(int(chunk_overlap), chunk_size // 2))
    min_chunk_chars = max(120, int(chunk_size * 0.6))

    if len(text) <= chunk_size:
        return [text.strip()]

    chunks: list[str] = []
    start = 0

    while start < len(text):
        proposed_end = min(len(text), start + chunk_size)
        end = proposed_end

        if proposed_end < len(text):
            search_start = min(len(text), start + min_chunk_chars)
            for sep in separators:
                if sep == "":
                    break
                candidate = text.rfind(sep, search_start, proposed_end)
                if candidate > search_start:
                    end = candidate + len(sep)
                    break

        piece = text[start:end].strip()
        if piece and (len(piece) >= min_chunk_chars or end >= len(text)):
            chunks.append(piece)
        elif end >= len(text) and piece:
            chunks.append(piece)

        if end >= len(text):
            break

        next_start = max(start + 1, end - chunk_overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def _entities_for_chunk(record: MedRecord, chunk_text: str) -> tuple[list[str], list[str]]:
    """Match entities into chunk text using normalized mention matching."""
    text_lower = chunk_text.lower()
    concept_ids: list[str] = []
    entity_texts: list[str] = []
    seen = set()

    for entity in record.entities:
        mention = entity.text.lower().strip()
        if not mention:
            continue
        if mention in text_lower:
            key = (entity.concept_id, entity.text.lower())
            if key in seen:
                continue
            seen.add(key)
            concept_ids.append(entity.concept_id)
            entity_texts.append(entity.text)

    return concept_ids, entity_texts


@timer
def build_chunks(
    records: list[MedRecord],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkRecord]:
    """Build chunk records for all documents with entity metadata."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks: list[ChunkRecord] = []
    for record in records:
        text_chunks = recursive_split(
            text=record.text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for idx, chunk_text in enumerate(text_chunks):
            concept_ids, entity_texts = _entities_for_chunk(record, chunk_text)
            chunks.append(
                ChunkRecord(
                    chunk_id=f"{record.pmid}_c{idx:04d}",
                    pmid=record.pmid,
                    split=record.split,
                    chunk_index=idx,
                    text=chunk_text,
                    title=record.title,
                    entity_count=len(concept_ids),
                    concept_ids=concept_ids,
                    entity_texts=entity_texts,
                )
            )

    logger.info("Built {} chunks from {} records", len(chunks), len(records))
    return chunks


@timer
def chunks_to_dataframe(chunks: list[ChunkRecord]) -> pd.DataFrame:
    """Convert chunk objects into a dataframe for persistence and analysis."""
    rows = []
    for chunk in chunks:
        row = asdict(chunk)
        row["concept_ids"] = json.dumps(row["concept_ids"], ensure_ascii=True)
        row["entity_texts"] = json.dumps(row["entity_texts"], ensure_ascii=True)
        rows.append(row)
    return pd.DataFrame(rows)


@timer
def persist_chunks(chunks: list[ChunkRecord], path: Path | None = None) -> Path:
    """Persist chunk records to parquet."""
    target = path or settings.processed_dir / "medmentions_chunks.parquet"
    df = chunks_to_dataframe(chunks)
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target, index=False)
    logger.info("Saved chunk dataframe to {}", target)
    return target


def load_chunks(path: Path | None = None) -> list[ChunkRecord]:
    """Restore chunk records from parquet."""
    source = path or settings.processed_dir / "medmentions_chunks.parquet"
    df = pd.read_parquet(source)

    restored: list[ChunkRecord] = []
    for _, row in df.iterrows():
        restored.append(
            ChunkRecord(
                chunk_id=row["chunk_id"],
                pmid=row["pmid"],
                split=row["split"],
                chunk_index=int(row["chunk_index"]),
                text=row["text"],
                title=row.get("title", ""),
                entity_count=int(row["entity_count"]),
                concept_ids=json.loads(row["concept_ids"]),
                entity_texts=json.loads(row["entity_texts"]),
            )
        )
    return restored


def build_chunk_lookup(chunks: list[ChunkRecord]) -> dict[str, list[str]]:
    """Map PMID to the chunk IDs created from that source document."""
    mapping: dict[str, list[str]] = {}
    for chunk in chunks:
        mapping.setdefault(chunk.pmid, []).append(chunk.chunk_id)
    return mapping


def chunk_statistics(chunks: list[ChunkRecord]) -> dict[str, float]:
    """Compute summary statistics for chunking diagnostics."""
    if not chunks:
        return {}

    lengths = np.array([len(chunk.text) for chunk in chunks], dtype=float)
    entity_counts = np.array([chunk.entity_count for chunk in chunks], dtype=float)

    return {
        "total_chunks": int(len(chunks)),
        "avg_chunk_length": float(np.mean(lengths)),
        "median_chunk_length": float(np.median(lengths)),
        "p95_chunk_length": float(np.percentile(lengths, 95)),
        "avg_entity_count": float(np.mean(entity_counts)),
        "pct_chunks_with_entities": float(np.mean(entity_counts > 0) * 100),
    }
