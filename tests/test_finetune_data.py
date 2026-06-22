"""Tests for optional fine-tuning dataset construction."""

from __future__ import annotations

from pathlib import Path

from src.chunking import ChunkRecord
from src.data_pipeline import EvalQuery
from src.finetune_data import (
    build_biomedical_sft_examples,
    build_chunk_id_lookup,
    persist_sft_jsonl,
    train_eval_split_sft,
)


def _chunk_rows() -> list[ChunkRecord]:
    return [
        ChunkRecord(
            chunk_id="p1_c0000",
            pmid="p1",
            split="train",
            chunk_index=0,
            text="Diabetes mellitus is associated with insulin resistance.",
            title="Paper 1",
            entity_count=2,
            concept_ids=["C0011849", "C0021641"],
            entity_texts=["Diabetes mellitus", "insulin resistance"],
        ),
        ChunkRecord(
            chunk_id="p2_c0000",
            pmid="p2",
            split="train",
            chunk_index=0,
            text="Hypertension and obesity are linked in this cohort.",
            title="Paper 2",
            entity_count=2,
            concept_ids=["C0020538", "C0028754"],
            entity_texts=["Hypertension", "obesity"],
        ),
    ]


def _eval_rows() -> list[EvalQuery]:
    return [
        EvalQuery(
            query_id="p1_q1",
            query="What does the abstract report about diabetes mellitus?",
            reference_answer="Diabetes mellitus is associated with insulin resistance.",
            source_pmid="p1",
            supporting_chunk_ids=["p1_c0000"],
            supporting_concept_ids=["C0011849"],
        ),
        EvalQuery(
            query_id="p2_q1",
            query="What does the abstract report about hypertension?",
            reference_answer="Hypertension and obesity are linked in this cohort.",
            source_pmid="p2",
            supporting_chunk_ids=["p2_c0000"],
            supporting_concept_ids=["C0020538"],
        ),
    ]


def test_build_biomedical_sft_examples_uses_real_fields() -> None:
    chunks = _chunk_rows()
    lookup = build_chunk_id_lookup(chunks)
    rows = build_biomedical_sft_examples(_eval_rows(), lookup, max_examples=10)
    assert rows
    assert rows[0].query
    assert rows[0].completion
    assert rows[0].messages[0]["role"] == "user"
    assert rows[0].messages[1]["role"] == "assistant"


def test_train_eval_split_sft_shape() -> None:
    rows = build_biomedical_sft_examples(_eval_rows(), build_chunk_id_lookup(_chunk_rows()), max_examples=10)
    train_rows, eval_rows = train_eval_split_sft(rows, eval_fraction=0.5, seed=123)
    assert len(train_rows) + len(eval_rows) == len(rows)
    assert eval_rows


def test_persist_sft_jsonl_creates_files(tmp_path: Path) -> None:
    rows = build_biomedical_sft_examples(_eval_rows(), build_chunk_id_lookup(_chunk_rows()), max_examples=10)
    train_rows, eval_rows = train_eval_split_sft(rows, eval_fraction=0.5, seed=123)
    paths = persist_sft_jsonl(train_rows, eval_rows, out_dir=tmp_path)
    assert paths["train_jsonl"].exists()
    assert paths["eval_jsonl"].exists()
    assert paths["manifest"].exists()
