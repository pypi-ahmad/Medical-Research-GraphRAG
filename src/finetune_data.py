"""Dataset builders for optional biomedical SFT in NB11.

This module intentionally stays additive and side-effect light:
- it transforms existing real MedMentions-derived artifacts,
- it does not generate synthetic rows,
- it writes JSONL artifacts that TRL SFTTrainer can consume.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

from src.chunking import ChunkRecord
from src.config import settings
from src.data_pipeline import EvalQuery
from src.utils import save_json


@dataclass(slots=True)
class SFTExample:
    """One supervised fine-tuning example derived from real biomedical records."""

    example_id: str
    query: str
    answer: str
    source_pmid: str
    supporting_chunk_ids: list[str]
    supporting_chunk_texts: list[str]
    prompt: str
    completion: str
    messages: list[dict[str, str]]


def build_chunk_id_lookup(chunks: list[ChunkRecord]) -> dict[str, ChunkRecord]:
    """Map chunk IDs to chunk records for fast lookup."""
    return {chunk.chunk_id: chunk for chunk in chunks}


def _build_prompt(query: str, context_chunks: list[str]) -> str:
    """Build a retrieval-grounded training prompt."""
    context = "\n\n".join(f"[{idx + 1}] {text[:900]}" for idx, text in enumerate(context_chunks[:8]))
    return (
        "You are a biomedical research assistant. Use only the provided context.\n"
        "If the context is insufficient, say what is missing.\n\n"
        f"Question: {query}\n\n"
        f"Context:\n{context}"
    )


def build_biomedical_sft_examples(
    eval_queries: list[EvalQuery],
    chunk_lookup: dict[str, ChunkRecord],
    max_examples: int | None = None,
) -> list[SFTExample]:
    """Create SFT examples from real extractive eval queries.

    Args:
        eval_queries: Real query/reference pairs generated from MedMentions abstracts.
        chunk_lookup: Mapping from chunk_id to chunk content.
        max_examples: Optional cap for generated examples.

    Returns:
        A list of SFTExample rows in conversational and plain-text formats.

    Example:
        ```python
        examples = build_biomedical_sft_examples(eval_queries, chunk_by_id, max_examples=2000)
        ```
    """
    rows: list[SFTExample] = []
    cap = max_examples if max_examples is not None else settings.finetune_max_train_examples

    for item in eval_queries:
        support_ids = [cid for cid in item.supporting_chunk_ids if cid in chunk_lookup]
        if not support_ids:
            continue

        support_texts = [chunk_lookup[cid].text for cid in support_ids[:8]]
        prompt = _build_prompt(item.query, support_texts)
        completion = item.reference_answer.strip()
        if not completion:
            continue

        rows.append(
            SFTExample(
                example_id=item.query_id,
                query=item.query,
                answer=item.reference_answer,
                source_pmid=item.source_pmid,
                supporting_chunk_ids=support_ids,
                supporting_chunk_texts=support_texts,
                prompt=prompt,
                completion=completion,
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ],
            )
        )
        if cap and len(rows) >= cap:
            break

    logger.info("Built {} biomedical SFT examples", len(rows))
    return rows


def train_eval_split_sft(
    examples: list[SFTExample],
    eval_fraction: float = 0.1,
    seed: int | None = None,
) -> tuple[list[SFTExample], list[SFTExample]]:
    """Split examples into train/eval sets with deterministic shuffling."""
    if not examples:
        return [], []

    eval_fraction = min(max(eval_fraction, 0.0), 0.5)
    rng = np.random.default_rng(settings.random_seed if seed is None else seed)
    indices = np.arange(len(examples))
    rng.shuffle(indices)
    eval_size = int(round(len(indices) * eval_fraction))
    eval_ids = set(indices[:eval_size].tolist())

    train_rows = [examples[i] for i in range(len(examples)) if i not in eval_ids]
    eval_rows = [examples[i] for i in range(len(examples)) if i in eval_ids]
    return train_rows, eval_rows


def examples_to_trl_rows(examples: list[SFTExample]) -> list[dict[str, Any]]:
    """Convert examples into TRL-friendly conversational rows."""
    rows: list[dict[str, Any]] = []
    for item in examples:
        rows.append(
            {
                "id": item.example_id,
                "source_pmid": item.source_pmid,
                "messages": item.messages,
                "text": f"{item.prompt}\n\n{item.completion}",
            }
        )
    return rows


def persist_sft_jsonl(
    train_examples: list[SFTExample],
    eval_examples: list[SFTExample],
    out_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist train/eval SFT rows to JSONL for reproducible NB11 runs."""
    target_dir = out_dir or settings.finetune_dataset_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    train_jsonl = target_dir / "sft_train.jsonl"
    eval_jsonl = target_dir / "sft_eval.jsonl"
    train_preview_json = target_dir / "sft_train_preview.json"
    eval_preview_json = target_dir / "sft_eval_preview.json"

    def _write_jsonl(path: Path, rows: list[SFTExample]) -> None:
        with path.open("w", encoding="utf-8") as handle:
            for row in examples_to_trl_rows(rows):
                handle.write(json.dumps(row, ensure_ascii=True))
                handle.write("\n")

    _write_jsonl(train_jsonl, train_examples)
    _write_jsonl(eval_jsonl, eval_examples)

    save_json([asdict(row) for row in train_examples[:50]], train_preview_json)
    save_json([asdict(row) for row in eval_examples[:50]], eval_preview_json)

    manifest = {
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_jsonl": str(train_jsonl),
        "eval_jsonl": str(eval_jsonl),
    }
    save_json(manifest, target_dir / "sft_manifest.json")
    logger.info("Saved SFT JSONL artifacts to {}", target_dir)
    return {
        "train_jsonl": train_jsonl,
        "eval_jsonl": eval_jsonl,
        "manifest": target_dir / "sft_manifest.json",
    }
