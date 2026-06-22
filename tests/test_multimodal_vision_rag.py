"""Tests for vision-model multimodal RAG helpers."""

from __future__ import annotations

from pathlib import Path

from src.multimodal_vision_rag import build_vision_multimodal_documents, vision_multimodal_search


def test_build_vision_multimodal_documents_image(tmp_path: Path, monkeypatch) -> None:
    image_path = tmp_path / "figure.png"
    image_path.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        "src.multimodal_vision_rag.extract_vision_evidence_with_qwen",
        lambda image_path, prompt=None, model=None: "vision evidence",
    )

    docs = build_vision_multimodal_documents(image_paths=[image_path], table_paths=[])
    assert len(docs) == 1
    assert docs[0].asset_id.startswith("vimg_")
    assert docs[0].metadata["source_type"] == "vision_model"


def test_vision_multimodal_search_source_tag(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.multimodal_vision_rag.multimodal_vector_search",
        lambda collection, query, top_k=8: [
            {
                "id": "c1",
                "text": "evidence",
                "metadata": {},
                "score": 0.7,
                "source": "multimodal_dense",
            }
        ],
    )

    rows = vision_multimodal_search(collection=object(), query="q", top_k=3)
    assert rows
    assert rows[0]["source"] == "multimodal_vision_dense"
