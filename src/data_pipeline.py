"""Data ingestion and evaluation-set construction for MedMentions GraphRAG."""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from datasets import load_dataset
from loguru import logger

from src.config import settings
from src.utils import save_json, timer


@dataclass(slots=True)
class EntityMention:
    """Normalized biomedical entity annotation extracted from MedMentions."""

    text: str
    concept_id: str
    semantic_type_ids: list[str]
    offsets: list[list[int]]


@dataclass(slots=True)
class MedRecord:
    """Normalized record containing merged title+abstract and entity annotations."""

    pmid: str
    split: str
    title: str
    abstract: str
    text: str
    entities: list[EntityMention]


@dataclass(slots=True)
class EvalQuery:
    """Extractive evaluation query backed by real source sentence evidence."""

    query_id: str
    query: str
    reference_answer: str
    source_pmid: str
    supporting_chunk_ids: list[str]
    supporting_concept_ids: list[str]


def _first_text(value: Any) -> str:
    """Extract first text string from MedMentions passage/entity fields."""
    if isinstance(value, list):
        return str(value[0]) if value else ""
    return str(value)


def _extract_passages(record: dict[str, Any]) -> tuple[str, str]:
    """Extract title and abstract from the raw passage representation."""
    title, abstract = "", ""
    for passage in record.get("passages", []):
        p_text = _first_text(passage.get("text", "")).strip()
        p_type = _first_text(passage.get("type", "")).lower()

        if "title" in p_type and not title:
            title = p_text
        elif "abstract" in p_type and not abstract:
            abstract = p_text
        elif not abstract and p_text:
            abstract = p_text

    if not abstract:
        abstract = _first_text(record.get("text", "")).strip()

    return title, abstract


def _normalize_entity(entity: dict[str, Any]) -> EntityMention:
    """Normalize entity fields and enforce list-based offset representation."""
    offsets = entity.get("offsets", []) or []
    normalized_offsets: list[list[int]] = []
    for offset in offsets:
        if isinstance(offset, (list, tuple)) and len(offset) >= 2:
            normalized_offsets.append([int(offset[0]), int(offset[1])])

    semantic_type_raw = entity.get("semantic_type_id", []) or []
    if isinstance(semantic_type_raw, str):
        semantic_type_ids = [semantic_type_raw]
    else:
        semantic_type_ids = [str(x) for x in semantic_type_raw]

    return EntityMention(
        text=_first_text(entity.get("text", "")).strip(),
        concept_id=str(entity.get("concept_id", "")).strip(),
        semantic_type_ids=semantic_type_ids,
        offsets=normalized_offsets,
    )


@timer
def load_medmentions_records(max_records: int = 5000) -> list[MedRecord]:
    """Load and normalize MedMentions records from all official splits.

    The function reads train/validation/test splits and keeps only real records
    from the public dataset. No synthetic augmentation is applied.
    """
    dataset = load_dataset("bigbio/medmentions", trust_remote_code=True)

    normalized: list[MedRecord] = []
    for split_name in ["train", "validation", "test"]:
        for raw in dataset[split_name]:
            title, abstract = _extract_passages(raw)
            merged = "\n\n".join([s for s in [title, abstract] if s]).strip()
            if not merged:
                continue

            entities = [_normalize_entity(e) for e in raw.get("entities", [])]
            entities = [e for e in entities if e.text and e.concept_id]

            normalized.append(
                MedRecord(
                    pmid=str(raw.get("pmid", "")),
                    split=split_name,
                    title=title,
                    abstract=abstract,
                    text=merged,
                    entities=entities,
                )
            )

    if max_records and len(normalized) > max_records:
        rng = np.random.default_rng(settings.random_seed)
        sampled_idx = rng.choice(len(normalized), size=max_records, replace=False)
        normalized = [normalized[i] for i in sorted(sampled_idx.tolist())]

    logger.info(
        "Loaded {} normalized MedMentions records across all splits",
        len(normalized),
    )
    return normalized


@timer
def persist_records(records: list[MedRecord], path: Path | None = None) -> Path:
    """Persist normalized records to JSON for notebook reuse."""
    target = path or settings.processed_dir / "medmentions_records.json"
    save_json([asdict(r) for r in records], target)
    logger.info("Saved normalized records to {}", target)
    return target


def load_persisted_records(path: Path | None = None) -> list[MedRecord]:
    """Load previously persisted normalized records from disk."""
    source = path or settings.processed_dir / "medmentions_records.json"
    payload = __import__("json").loads(source.read_text(encoding="utf-8"))

    restored: list[MedRecord] = []
    for item in payload:
        entities = [EntityMention(**entity) for entity in item.get("entities", [])]
        restored.append(
            MedRecord(
                pmid=item["pmid"],
                split=item["split"],
                title=item.get("title", ""),
                abstract=item.get("abstract", ""),
                text=item["text"],
                entities=entities,
            )
        )
    return restored


def _split_sentences(text: str) -> list[str]:
    """Split text into candidate sentences for extractive reference generation."""
    candidates = re.split(r"(?<=[.!?])\s+", text)
    return [sent.strip() for sent in candidates if len(sent.strip()) >= 30]


def build_extractive_eval_queries(
    records: list[MedRecord],
    chunk_lookup: dict[str, list[str]],
    sample_size: int = 100,
) -> list[EvalQuery]:
    """Construct evaluation queries with real references from source abstracts.

    Query templates are deterministic and use entity mentions from the record.
    The reference answer is an exact sentence from the abstract, ensuring
    evaluation remains fully grounded in real data.
    """
    rng = np.random.default_rng(settings.random_seed)

    candidates: list[EvalQuery] = []
    for record in records:
        if not record.abstract or not record.entities:
            continue

        sentences = _split_sentences(record.abstract)
        if not sentences:
            continue

        unique_entities = []
        seen = set()
        for entity in record.entities:
            key = (entity.text.lower(), entity.concept_id)
            if key in seen:
                continue
            seen.add(key)
            unique_entities.append(entity)

        for entity in unique_entities[:3]:
            sentence = next(
                (s for s in sentences if entity.text.lower() in s.lower()),
                "",
            )
            if not sentence:
                continue

            query = f"What does this biomedical abstract report about {entity.text}?"
            chunk_ids = chunk_lookup.get(record.pmid, [])
            candidates.append(
                EvalQuery(
                    query_id=f"{record.pmid}_{entity.concept_id}",
                    query=query,
                    reference_answer=sentence,
                    source_pmid=record.pmid,
                    supporting_chunk_ids=chunk_ids,
                    supporting_concept_ids=[entity.concept_id],
                )
            )

    if not candidates:
        return []

    if len(candidates) > sample_size:
        selected = rng.choice(len(candidates), size=sample_size, replace=False)
        candidates = [candidates[i] for i in sorted(selected.tolist())]

    logger.info("Built {} extractive evaluation queries", len(candidates))
    return candidates


@timer
def persist_eval_queries(queries: list[EvalQuery], path: Path | None = None) -> Path:
    """Persist evaluation query definitions to disk."""
    target = path or settings.eval_dir / "extractive_eval_queries.json"
    save_json([asdict(q) for q in queries], target)
    logger.info("Saved eval queries to {}", target)
    return target
