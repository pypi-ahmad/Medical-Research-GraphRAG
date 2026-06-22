"""Vision-model-driven multimodal RAG for biomedical assets."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import ollama
from loguru import logger

from src.config import settings
from src.multimodal_rag import (
    MultimodalChunk,
    MultimodalDocument,
    index_multimodal_chunks_to_chromadb,
    multimodal_documents_to_chunks,
    multimodal_vector_search,
    table_to_biomedical_text,
)
from src.utils import timer


def _slug(path: Path) -> str:
    """Create deterministic IDs from filenames."""
    stem = path.stem.lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in stem)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "asset"


@timer
def extract_vision_evidence_with_qwen(
    image_path: Path,
    prompt: str | None = None,
    model: str | None = None,
) -> str:
    """Extract biomedical evidence from images using qwen3.5 vision."""
    query = prompt or (
        "Analyze this biomedical image and extract clinically relevant evidence: "
        "findings, measurements, axis labels, legends, subgroup differences, trend "
        "direction, confidence intervals, and safety signals. Return plain text only."
    )
    model_name = model or settings.multimodal_vision_model
    retries = max(1, int(os.getenv("MULTIMODAL_VISION_RETRIES", "2")))
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = ollama.chat(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": query,
                        "images": [str(image_path)],
                    }
                ],
                options={"temperature": 0.0},
            )
            text = response["message"]["content"].strip()
            if text:
                return text
            raise RuntimeError("empty vision extraction response")
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Vision extraction attempt {}/{} failed for {}: {}",
                attempt,
                retries,
                image_path.name,
                exc,
            )
            if attempt < retries:
                time.sleep(min(2 * attempt, 6))
    logger.error(
        "Vision extraction failed for {} after {} attempts: {}",
        image_path.name,
        retries,
        last_exc,
    )
    return ""


@timer
def build_vision_multimodal_documents(
    *,
    image_paths: list[Path] | None = None,
    table_paths: list[Path] | None = None,
    vision_model: str | None = None,
) -> list[MultimodalDocument]:
    """Create multimodal documents with image evidence from a vision model."""
    image_paths = image_paths or []
    table_paths = table_paths or []
    docs: list[MultimodalDocument] = []

    for image_path in image_paths:
        try:
            text = extract_vision_evidence_with_qwen(image_path=image_path, model=vision_model)
        except Exception as exc:
            logger.warning("Skipping vision image asset {} due to extraction failure: {}", image_path.name, exc)
            continue
        if not text.strip():
            logger.warning("Skipping vision image asset {} due to empty extraction", image_path.name)
            continue
        docs.append(
            MultimodalDocument(
                asset_id=f"vimg_{_slug(image_path)}",
                modality="image",
                source_path=str(image_path),
                title=image_path.stem,
                extracted_text=text,
                metadata={
                    "source_type": "vision_model",
                    "filename": image_path.name,
                    "vision_model": vision_model or settings.multimodal_vision_model,
                },
            )
        )

    for table_path in table_paths:
        try:
            text = table_to_biomedical_text(table_path=table_path)
        except Exception as exc:
            logger.warning("Skipping vision table asset {} due to parse failure: {}", table_path.name, exc)
            continue
        if not text.strip():
            logger.warning("Skipping vision table asset {} because parsed text is empty", table_path.name)
            continue
        docs.append(
            MultimodalDocument(
                asset_id=f"vtbl_{_slug(table_path)}",
                modality="table",
                source_path=str(table_path),
                title=table_path.stem,
                extracted_text=text,
                metadata={"source_type": "tabular", "filename": table_path.name},
            )
        )

    logger.info("Built {} vision multimodal documents", len(docs))
    return docs


def vision_documents_to_chunks(
    docs: list[MultimodalDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[MultimodalChunk]:
    """Chunk vision-derived multimodal documents."""
    return multimodal_documents_to_chunks(
        docs=docs,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )


@timer
def index_vision_chunks_to_chromadb(
    chunks: list[MultimodalChunk],
    *,
    collection_name: str = "medical_multimodal_qwen_vision",
    batch_size: int = 64,
):
    """Index vision-derived multimodal chunks in Chroma."""
    return index_multimodal_chunks_to_chromadb(
        chunks=chunks,
        collection_name=collection_name,
        batch_size=batch_size,
    )


def vision_multimodal_search(
    collection,
    query: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Retrieve from vision-derived multimodal collection."""
    rows = multimodal_vector_search(collection=collection, query=query, top_k=top_k)
    for row in rows:
        row["source"] = "multimodal_vision_dense"
    return rows
