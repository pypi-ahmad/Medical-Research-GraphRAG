"""Tests for multimodal OCR/table ingestion helpers."""

from __future__ import annotations

from pathlib import Path

from src.multimodal_rag import (
    MultimodalDocument,
    multimodal_documents_to_chunks,
    table_to_biomedical_text,
)


def test_table_to_biomedical_text_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "lab_values.csv"
    csv_path.write_text("marker,value,unit\\nHbA1c,7.2,%\\nLDL,130,mg/dL\\n", encoding="utf-8")
    text = table_to_biomedical_text(csv_path)
    assert "Biomedical table columns" in text
    assert "HbA1c" in text
    assert "LDL" in text


def test_multimodal_documents_to_chunks() -> None:
    docs = [
        MultimodalDocument(
            asset_id="img_a",
            modality="image",
            source_path="/tmp/a.png",
            title="Figure A",
            extracted_text="Diabetes biomarkers are elevated in cohort A. " * 40,
            metadata={"source_type": "ocr"},
        )
    ]
    chunks = multimodal_documents_to_chunks(docs, chunk_size=200, chunk_overlap=20)
    assert chunks
    assert chunks[0].asset_id == "img_a"
    assert chunks[0].metadata["modality"] == "image"
