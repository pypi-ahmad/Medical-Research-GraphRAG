"""Multimodal RAG ingestion and retrieval for biomedical assets."""

from __future__ import annotations

import io
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chromadb
import numpy as np
import ollama
import pandas as pd
from loguru import logger

from src.chunking import DEFAULT_SEPARATORS, recursive_split
from src.chroma_retriever import create_chroma_client, create_or_replace_collection
from src.config import settings
from src.embeddings import embed_query, embed_texts
from src.utils import save_json, timer


@dataclass(slots=True)
class MultimodalDocument:
    """Raw multimodal medical asset transformed into text evidence."""

    asset_id: str
    modality: str
    source_path: str
    title: str
    extracted_text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class MultimodalChunk:
    """Chunk-level retrievable unit from multimodal evidence."""

    chunk_id: str
    asset_id: str
    modality: str
    chunk_index: int
    text: str
    metadata: dict[str, Any]


@dataclass(slots=True)
class MultimodalEvalQuery:
    """Evaluation query grounded in a real multimodal asset."""

    query_id: str
    query: str
    reference_answer: str
    asset_id: str
    modality: str
    relevant_chunk_ids: list[str]


def _slug(path: Path) -> str:
    """Create deterministic IDs from filenames."""
    stem = path.stem.lower()
    safe = "".join(ch if ch.isalnum() else "-" for ch in stem)
    while "--" in safe:
        safe = safe.replace("--", "-")
    return safe.strip("-") or "asset"


def _glm_ocr_cli_command(
    *,
    image_path: Path,
    query: str,
    model: str,
) -> list[str]:
    """Build deterministic CLI command for `ollama run` OCR execution."""
    return [
        "ollama",
        "run",
        model,
        str(image_path),
        query,
    ]


@timer
def extract_text_with_glm_ocr_with_backend(
    image_path: Path,
    prompt: str | None = None,
    model: str | None = None,
    allow_fallback: bool | None = None,
    timeout_seconds: int | None = None,
) -> tuple[str, str]:
    """Extract chart/figure text with CLI-first OCR and API fallback.

    Returns:
        Tuple of (extracted_text, backend_used), where backend is one of:
        - "ollama_run"
        - "ollama_chat_fallback"
    """
    query = prompt or (
        "Extract all medically relevant visible text, axis labels, units, table cells, "
        "captions, and legend entries. Return plain text only."
    )
    model_name = model or settings.multimodal_ocr_model
    allow_fallback = settings.ocr_cli_allow_fallback if allow_fallback is None else allow_fallback
    timeout_seconds = timeout_seconds or settings.ocr_cli_timeout_seconds

    command = _glm_ocr_cli_command(image_path=image_path, query=query, model=model_name)
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip(), "ollama_run"
        error_text = result.stderr.strip() or result.stdout.strip() or "empty OCR response"
        raise RuntimeError(f"OCR CLI failed with code {result.returncode}: {error_text}")
    except Exception as exc:
        if not allow_fallback:
            raise RuntimeError("OCR CLI execution failed and fallback is disabled.") from exc
        logger.warning("Falling back to Ollama chat OCR due to CLI failure: {}", exc)
        retries = max(1, int(os.getenv("MULTIMODAL_OCR_CHAT_RETRIES", "2")))
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
                extracted = response["message"]["content"].strip()
                if extracted:
                    return extracted, "ollama_chat_fallback"
                raise RuntimeError("empty OCR fallback response")
            except Exception as chat_exc:
                last_exc = chat_exc
                logger.warning(
                    "OCR fallback attempt {}/{} failed for {}: {}",
                    attempt,
                    retries,
                    image_path.name,
                    chat_exc,
                )
                if attempt < retries:
                    time.sleep(min(2 * attempt, 6))
        logger.error(
            "OCR extraction failed for {} after CLI and fallback attempts: {}",
            image_path.name,
            last_exc,
        )
        return "", "ocr_failed"


@timer
def extract_text_with_glm_ocr(
    image_path: Path,
    prompt: str | None = None,
    model: str | None = None,
    allow_fallback: bool | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """Compatibility wrapper returning OCR text only."""
    text, _ = extract_text_with_glm_ocr_with_backend(
        image_path=image_path,
        prompt=prompt,
        model=model,
        allow_fallback=allow_fallback,
        timeout_seconds=timeout_seconds,
    )
    return text


def table_to_biomedical_text(
    table_path: Path,
    max_rows: int = 100,
) -> str:
    """Convert CSV/TSV/XLSX biomedical table to narrative text."""
    sep = ","
    suffix = table_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        df = pd.read_csv(table_path, sep=sep).head(max_rows)
    elif suffix in {".xlsx", ".xls"}:
        df = pd.read_excel(table_path).head(max_rows)
    else:
        raise ValueError(f"Unsupported table format: {table_path.suffix}")

    # Some upstream generators emit escaped newlines ("\\n") in plain-text CSV.
    # Reparse those files to recover row structure for robust ingestion/testing.
    if df.empty and suffix in {".csv", ".tsv"}:
        try:
            raw_text = table_path.read_text(encoding="utf-8")
            if "\\n" in raw_text:
                reparsed = pd.read_csv(io.StringIO(raw_text.replace("\\n", "\n")), sep=sep).head(max_rows)
                if not reparsed.empty:
                    df = reparsed
        except Exception as exc:  # noqa: BLE001
            logger.warning("Escaped-newline CSV reparsing failed for {}: {}", table_path.name, exc)

    if df.empty:
        return ""

    header = ", ".join(str(col) for col in df.columns.tolist())
    rows = []
    for idx, row in df.iterrows():
        values = [f"{col}={row[col]}" for col in df.columns.tolist()]
        rows.append(f"Row {idx + 1}: " + "; ".join(values))

    return (
        f"Biomedical table columns: {header}\n"
        + "\n".join(rows)
    )


@timer
def build_multimodal_documents(
    *,
    image_paths: list[Path] | None = None,
    table_paths: list[Path] | None = None,
    ocr_model: str | None = None,
    ocr_allow_fallback: bool | None = None,
    ocr_timeout_seconds: int | None = None,
) -> list[MultimodalDocument]:
    """Create text-bearing multimodal medical documents from assets."""
    image_paths = image_paths or []
    table_paths = table_paths or []
    docs: list[MultimodalDocument] = []

    for image_path in image_paths:
        try:
            text, backend = extract_text_with_glm_ocr_with_backend(
                image_path=image_path,
                model=ocr_model,
                allow_fallback=ocr_allow_fallback,
                timeout_seconds=ocr_timeout_seconds,
            )
        except Exception as exc:
            logger.warning("Skipping image asset {} due to OCR failure: {}", image_path.name, exc)
            continue
        if not text.strip():
            logger.warning(
                "Skipping image asset {} because extracted OCR text is empty (backend={})",
                image_path.name,
                backend,
            )
            continue
        docs.append(
            MultimodalDocument(
                asset_id=f"img_{_slug(image_path)}",
                modality="image",
                source_path=str(image_path),
                title=image_path.stem,
                extracted_text=text,
                metadata={
                    "source_type": "ocr",
                    "filename": image_path.name,
                    "ocr_backend": backend,
                    "ocr_model": ocr_model or settings.multimodal_ocr_model,
                },
            )
        )

    for table_path in table_paths:
        try:
            text = table_to_biomedical_text(table_path=table_path)
        except Exception as exc:
            logger.warning("Skipping table asset {} due to parse failure: {}", table_path.name, exc)
            continue
        if not text.strip():
            logger.warning("Skipping table asset {} because parsed text is empty", table_path.name)
            continue
        docs.append(
            MultimodalDocument(
                asset_id=f"tbl_{_slug(table_path)}",
                modality="table",
                source_path=str(table_path),
                title=table_path.stem,
                extracted_text=text,
                metadata={"source_type": "tabular", "filename": table_path.name},
            )
        )

    logger.info("Built {} multimodal documents", len(docs))
    return docs


def multimodal_documents_to_chunks(
    docs: list[MultimodalDocument],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[MultimodalChunk]:
    """Chunk multimodal extracted text for retrieval indexing."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    chunks: list[MultimodalChunk] = []
    for doc in docs:
        parts = recursive_split(
            doc.extracted_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=DEFAULT_SEPARATORS,
        )
        for idx, part in enumerate(parts):
            chunks.append(
                MultimodalChunk(
                    chunk_id=f"{doc.asset_id}_c{idx:04d}",
                    asset_id=doc.asset_id,
                    modality=doc.modality,
                    chunk_index=idx,
                    text=part,
                    metadata={
                        "title": doc.title,
                        "source_path": doc.source_path,
                        "modality": doc.modality,
                        **doc.metadata,
                    },
                )
            )
    return chunks


@timer
def index_multimodal_chunks_to_chromadb(
    chunks: list[MultimodalChunk],
    *,
    collection_name: str = "medical_multimodal",
    batch_size: int = 64,
) -> chromadb.Collection:
    """Index multimodal chunks in a dedicated Chroma collection."""
    client = create_chroma_client(str(settings.chroma_dir))
    collection = create_or_replace_collection(client, collection_name)
    if not chunks:
        logger.warning("No multimodal chunks available for indexing in collection {}", collection_name)
        return collection

    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, model=settings.embedding_model, batch_size=batch_size)

    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        block = chunks[start:end]
        collection.add(
            ids=[chunk.chunk_id for chunk in block],
            documents=[chunk.text for chunk in block],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[chunk.metadata for chunk in block],
        )
    return collection


def multimodal_vector_search(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 8,
) -> list[dict[str, Any]]:
    """Run vector search against multimodal evidence collection."""
    q_vec = embed_query(query, model=settings.embedding_model)
    response = collection.query(
        query_embeddings=[q_vec.tolist()],
        n_results=top_k,
        include=["metadatas", "documents", "distances"],
    )
    rows: list[dict[str, Any]] = []
    if not response.get("ids"):
        return rows
    for i, item_id in enumerate(response["ids"][0]):
        distance = float(response["distances"][0][i]) if response.get("distances") else 1.0
        rows.append(
            {
                "id": item_id,
                "text": response["documents"][0][i],
                "metadata": response["metadatas"][0][i],
                "score": 1.0 - distance,
                "source": "multimodal_dense",
            }
        )
    return rows


def persist_multimodal_manifest(
    docs: list[MultimodalDocument],
    chunks: list[MultimodalChunk],
    out_path: Path | None = None,
) -> Path:
    """Persist multimodal document/chunk manifest for reproducibility."""
    target = out_path or (settings.multimodal_dir / "multimodal_manifest.json")
    payload = {
        "documents": [asdict(doc) for doc in docs],
        "chunks": [asdict(chunk) for chunk in chunks],
    }
    save_json(payload, target)
    return target


def _first_sentence(text: str) -> str:
    """Extract a robust first sentence-like segment for references."""
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    for sep in [". ", "? ", "! ", "\n"]:
        if sep in cleaned:
            candidate = cleaned.split(sep, 1)[0].strip()
            if len(candidate) >= 24:
                return candidate
    return cleaned[:280].strip()


def _default_query_for_doc(doc: MultimodalDocument) -> str:
    """Generate a deterministic question when manifest query is unavailable."""
    if doc.modality == "table":
        return f"What key biomedical values are reported in table {doc.title}?"
    return f"What biomedical findings are shown in figure {doc.title}?"


def load_pmc_multimodal_manifest(
    manifest_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Load PMC asset manifest if present; return empty list otherwise."""
    source = manifest_path or (settings.multimodal_dir / "pmc_asset_manifest.json")
    if not source.exists():
        return []
    payload = json.loads(source.read_text(encoding="utf-8"))
    rows = payload.get("assets", [])
    return rows if isinstance(rows, list) else []


def build_multimodal_eval_queries(
    docs: list[MultimodalDocument],
    chunks: list[MultimodalChunk],
    *,
    manifest_rows: list[dict[str, Any]] | None = None,
    max_queries: int = 80,
) -> list[MultimodalEvalQuery]:
    """Build real multimodal eval queries from manifest-backed assets.

    If manifest rows are missing, the function falls back to deterministic
    query/reference extraction from multimodal documents themselves.
    """
    by_asset: dict[str, list[str]] = {}
    for chunk in chunks:
        by_asset.setdefault(chunk.asset_id, []).append(chunk.chunk_id)

    docs_by_path = {str(Path(doc.source_path).resolve()): doc for doc in docs}
    docs_by_asset = {doc.asset_id: doc for doc in docs}

    queries: list[MultimodalEvalQuery] = []
    rows = manifest_rows or []
    for idx, row in enumerate(rows):
        local_path = str(Path(row.get("local_path", "")).resolve())
        doc = docs_by_path.get(local_path)
        if doc is None:
            continue
        relevant = by_asset.get(doc.asset_id, [])
        if not relevant:
            continue

        ref = (
            str(row.get("reference_answer", "")).strip()
            or str(row.get("caption", "")).strip()
            or _first_sentence(doc.extracted_text)
        )
        question = str(row.get("question", "")).strip() or _default_query_for_doc(doc)
        if not ref:
            continue

        queries.append(
            MultimodalEvalQuery(
                query_id=f"mm_manifest_{idx:04d}",
                query=question,
                reference_answer=ref,
                asset_id=doc.asset_id,
                modality=doc.modality,
                relevant_chunk_ids=relevant,
            )
        )

    if not queries:
        for idx, doc in enumerate(docs):
            relevant = by_asset.get(doc.asset_id, [])
            if not relevant:
                continue
            ref = _first_sentence(doc.extracted_text)
            if not ref:
                continue
            queries.append(
                MultimodalEvalQuery(
                    query_id=f"mm_doc_{idx:04d}",
                    query=_default_query_for_doc(doc),
                    reference_answer=ref,
                    asset_id=doc.asset_id,
                    modality=doc.modality,
                    relevant_chunk_ids=relevant,
                )
            )

    if not queries:
        return []

    if len(queries) > max_queries:
        rng = np.random.default_rng(settings.random_seed)
        keep = rng.choice(len(queries), size=max_queries, replace=False)
        queries = [queries[i] for i in sorted(keep.tolist())]

    return queries
