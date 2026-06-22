# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: Python 3.12
#     language: python
#     name: python3
# ---

# %% [markdown]
# # NB08: Multimodal RAG for Biomedical Evidence (Text + Figures/Tables)
#
# ## What
# Multimodal RAG extends standard text RAG by ingesting evidence from medical charts, figures, and tables.
# In this notebook, multimodal evidence is converted to text via:
# - OCR for figure/chart/diagram images (`ollama run glm-ocr` primary path with API fallback),
# - structured table-to-text conversion.
#
# ## Why
# Biomedical knowledge frequently appears in figures/tables not fully captured by abstract text.
# A text-only retriever can miss this evidence.
#
# ## When
# Use multimodal RAG when:
# - key findings are in survival curves/plots/tables,
# - dosage, cohort, biomarker values, and subgroup results are visual/tabular,
# - downstream QA requires numeric or figure-grounded context.
#
# ## Tradeoffs
# - Additional ingestion complexity and preprocessing cost.
# - OCR noise can reduce retrieval precision if not cleaned.
# - Requires provenance-aware storage to preserve traceability.
#
# ## Alternatives
# - **Text-only RAG**: simpler but can under-utilize non-text biomedical evidence.
# - **Vision-model end-to-end QA**: potentially richer, but introduces extra model dependencies and serving complexity.
# - **Manual extraction**: high quality but not scalable.
#
# ## Production Considerations
# - Version OCR outputs and source assets together.
# - Add OCR quality checks and confidence filtering.
# - Preserve modality/source provenance in retrieval metadata.
# - Separate PHI-sensitive assets and apply governance controls.

# %% [markdown]
# ## Definition and Core Concepts
# - **Multimodal RAG**: retrieval-augmented generation that ingests non-text assets (images/tables) and text together.
# - **OCR Path**: converts visible biomedical text in figures/charts to retrievable evidence.
# - **Table-to-Text Path**: converts tabular clinical values into textual retrieval units.
# - **Provenance Metadata**: records modality and extraction backend for every chunk.

# %% [markdown]
# ## Why Multimodal RAG Was Developed
# Biomedical findings are often reported in:
# - Kaplan-Meier curves,
# - subgroup comparison charts,
# - biomarker tables.
#
# Text-only retrieval misses this evidence. Multimodal RAG was developed to close that gap.

# %% [markdown]
# ## What Traditional RAG Limitation It Solves
# Standard RAG usually indexes only narrative text. It underperforms when critical evidence exists only in visual/tabular form.
# Multimodal ingestion broadens retrievable evidence coverage.

# %% [markdown]
# ## Architecture Explanation
# - `Images -> OCR`: figure text extraction.
# - `Tables -> Parser`: structured value extraction.
# - `Corpus Merge`: both outputs become one multimodal text corpus.
# - `Chunk + Embed`: multimodal chunks indexed in Chroma.
# - `Retrieve + Generate + Judge`: same answer and quality pipeline as text RAG, now with multimodal context.

# %% [markdown]
# ## Workflow Explanation
# 1. Discover multimodal assets.
# 2. Extract OCR and table text.
# 3. Chunk and index multimodal evidence.
# 4. Retrieve for biomedical questions.
# 5. Evaluate retrieval/generation/RAG/judge metrics.

# %% [markdown]
# ## Component-by-Component Breakdown
# 1. **Asset Discovery**: scans `data/multimodal/images` and `data/multimodal/tables`.
# 2. **Extractor Layer**: CLI-first OCR and table text transformation.
# 3. **Indexer Layer**: Chroma multimodal collection.
# 4. **Evaluation Layer**: required metric families and judge outputs.

# %% [markdown]
# ## Advantages and Disadvantages
# **Advantages**
# - Captures evidence unavailable to text-only pipelines.
# - Preserves traceable multimodal provenance.
# - Reuses the same retrieval/evaluation scaffolding.
#
# **Disadvantages**
# - OCR noise can reduce precision.
# - Additional ingestion and storage complexity.
# - Requires modality-specific quality checks.

# %% [markdown]
# ## Comparison Against Other Implemented Variants
# - **Standard/Hybrid RAG**: strong for text, blind to non-text evidence.
# - **CRAG**: focuses on correction policy, not modality expansion.
# - **Multimodal RAG (this notebook)**: expands the evidence space itself.

# %% [markdown]
# ## Implementation Decisions in This Project
# - OCR backend: `ollama run glm-ocr` (primary) with optional API fallback.
# - Vision-specific strategy is implemented separately in NB10 to keep experiments isolated.
# - All outputs remain schema-complete placeholders until explicit execution.

# %% [markdown]
# ## Multimodal Architecture Diagram
#
# ```mermaid
# flowchart LR
#     A[Biomedical Images/Figures] --> B[CLI OCR via ollama run glm-ocr]
#     C[Biomedical Tables] --> D[Table-to-Text Parser]
#     B --> E[Multimodal Text Corpus]
#     D --> E
#     E --> F[Chunk + Embed<br/>qwen3-embedding:4b]
#     F --> G[Multimodal Vector Index]
#     G --> H[Multimodal Retrieval]
#     H --> I[Generation<br/>granite4.1:8b]
#     H --> J[Judge<br/>granite4.1:8b]
# ```

# %% [markdown]
# ## Workflow Diagram
#
# ```mermaid
# flowchart TD
#     A[Discover multimodal assets] --> B[CLI OCR and table parsing]
#     B --> C[Create multimodal documents]
#     C --> D[Chunk multimodal documents]
#     D --> E[Index to Chroma multimodal collection]
#     E --> F[Retrieve multimodal evidence]
#     F --> G[Generate biomedical answer]
#     G --> H[Evaluate retrieval/generation/rag metrics]
#     G --> I[Judge groundedness evaluation]
# ```

# %%
# Input: multimodal ingestion and evaluation modules.
# Output: initialized runtime for multimodal RAG pipeline.
# Logic: configure model routing and execution mode flags.
# Complexity: O(1).
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd().parent))

from src.config import settings
from src.data_pipeline import load_persisted_records
from src.evaluator import (
    GenerationExample,
    RetrievalExample,
    compute_generation_metrics,
    compute_rag_metrics,
    compute_retrieval_metrics,
)
from src.llm_judge import grade_groundedness, grade_retrieval_quality
from src.multimodal_rag import (
    build_multimodal_eval_queries,
    build_multimodal_documents,
    index_multimodal_chunks_to_chromadb,
    load_pmc_multimodal_manifest,
    multimodal_documents_to_chunks,
    multimodal_vector_search,
    persist_multimodal_manifest,
)
from src.utils import save_json

RUN_FULL_EVAL = os.getenv("RUN_FULL_EVAL", "false").strip().lower() == "true"
print(f"OCR model: {settings.multimodal_ocr_model}")
print(f"OCR CLI timeout: {settings.ocr_cli_timeout_seconds}s")
print(f"OCR CLI fallback enabled: {settings.ocr_cli_allow_fallback}")
print(f"Embedding model: {settings.embedding_model}")
print(f"Generator model: {settings.generator_model}")
print(f"Judge model: {settings.judge_model}")
print(f"RUN_FULL_EVAL={RUN_FULL_EVAL}")

# %% [markdown]
# ## Step 1: Discover Multimodal Biomedical Assets
#
# Expected asset locations (example):
# - images/charts/figures: `data/multimodal/images/`
# - tables: `data/multimodal/tables/`
#
# The notebook is robust to empty folders in implementation-only mode.

# %%
# Input: multimodal asset root directory.
# Output: discovered image and table asset paths.
# Logic: scan known multimodal subdirectories and collect supported file types.
# Complexity: O(number_of_files_in_asset_dirs).
multimodal_root = settings.multimodal_dir
image_dir = multimodal_root / "images"
table_dir = multimodal_root / "tables"

image_paths = []
if image_dir.exists():
    for suffix in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tiff", "*.webp"]:
        image_paths.extend(sorted(image_dir.glob(suffix)))

table_paths = []
if table_dir.exists():
    for suffix in ["*.csv", "*.tsv", "*.xlsx", "*.xls"]:
        table_paths.extend(sorted(table_dir.glob(suffix)))

print(f"Discovered image assets: {len(image_paths)}")
print(f"Discovered table assets: {len(table_paths)}")

# %% [markdown]
# ## Step 2: Build Multimodal Documents (OCR + Table Parsing)

# %%
# Input: discovered image/table asset paths.
# Output: multimodal text documents with modality metadata.
# Logic: run OCR on images and convert tables to textual evidence.
# Complexity: O(ocr_cost + table_parse_cost).
if RUN_FULL_EVAL:
    multimodal_docs = build_multimodal_documents(
        image_paths=image_paths,
        table_paths=table_paths,
        ocr_model=settings.multimodal_ocr_model,
        ocr_allow_fallback=settings.ocr_cli_allow_fallback,
        ocr_timeout_seconds=settings.ocr_cli_timeout_seconds,
    )
else:
    multimodal_docs = []

print(f"Multimodal documents built: {len(multimodal_docs)}")

# %% [markdown]
# ## Step 3: Chunk and Index Multimodal Evidence
#
# This creates a dedicated multimodal collection in ChromaDB.

# %%
# Input: multimodal documents.
# Output: multimodal chunks and optional indexed collection.
# Logic: split extracted text into retrieval units and index with embeddings.
# Complexity: O(chunking + embedding + indexing).
if RUN_FULL_EVAL and multimodal_docs:
    multimodal_chunks = multimodal_documents_to_chunks(multimodal_docs)
    multimodal_collection = index_multimodal_chunks_to_chromadb(
        multimodal_chunks,
        collection_name="medical_multimodal",
        batch_size=64,
    )
    persist_multimodal_manifest(multimodal_docs, multimodal_chunks)
else:
    multimodal_chunks = []
    multimodal_collection = None

print(f"Multimodal chunks: {len(multimodal_chunks)}")

# %% [markdown]
# ## Step 4: Multimodal Retrieval Demonstration
#
# We retrieve evidence from the multimodal index using biomedical queries.

# %%
# Input: biomedical query and multimodal collection.
# Output: top-k multimodal retrieval results.
# Logic: vector search over multimodal OCR/table chunks.
# Complexity: O(multimodal_dense_query).
sample_query = "What does the figure or table evidence suggest about diabetes-related biomarkers?"

if RUN_FULL_EVAL and multimodal_collection is not None:
    sample_rows = multimodal_vector_search(multimodal_collection, sample_query, top_k=6)
else:
    sample_rows = []

if sample_rows:
    for idx, row in enumerate(sample_rows, start=1):
        print(f"[{idx}] id={row['id']} score={row['score']:.4f} modality={row['metadata'].get('modality', '')}")
        print(row['text'][:240], "...\n")
else:
    print("Placeholder: multimodal retrieval outputs will appear after execution phase.")

# %% [markdown]
# ## Step 5: Evaluation Harness for Multimodal RAG
#
# Required retrieval metrics:
# - Precision@K, Recall@K, F1, MRR, NDCG
#
# Required generation metrics:
# - Exact Match, BLEU, ROUGE, METEOR, BERTScore
#
# Required RAG metrics:
# - Faithfulness, Context Precision, Context Recall, Answer Relevancy

# %%
# Input: multimodal docs/chunks and optional PMC asset manifest.
# Output: retrieval/generation/rag metric payloads.
# Logic: build multimodal-grounded eval queries, then compute required metrics.
# Complexity: O(num_queries * (retrieval + generation + metric_cost)).
records = load_persisted_records()
manifest_rows = load_pmc_multimodal_manifest()
multimodal_eval_queries = build_multimodal_eval_queries(
    docs=multimodal_docs,
    chunks=multimodal_chunks,
    manifest_rows=manifest_rows,
    max_queries=min(settings.eval_query_count, 80),
)

retrieval_examples: list[RetrievalExample] = []
generation_examples: list[GenerationExample] = []

if RUN_FULL_EVAL and multimodal_collection is not None and multimodal_eval_queries:
    subset = multimodal_eval_queries[: min(settings.generation_eval_count, len(multimodal_eval_queries))]
    for item in subset:
        rows = multimodal_vector_search(multimodal_collection, item.query, top_k=settings.top_k_retrieval)
        ids = [row["id"] for row in rows]
        retrieval_examples.append(
            RetrievalExample(retrieved_ids=ids, relevant_ids=item.relevant_chunk_ids)
        )

        context = "\n\n".join(f"[{i+1}] {row['text'][:900]}" for i, row in enumerate(rows[:8]))
        try:
            response = __import__("ollama").chat(
                model=settings.generator_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the biomedical question using multimodal evidence only. "
                            "State insufficiency if evidence is weak.\n\n"
                            f"Question: {item.query}\n\nEvidence:\n{context}"
                        ),
                    }
                ],
                options={"temperature": 0.2},
            )
            answer = response["message"]["content"]
        except Exception as exc:
            answer = (
                "Insufficient multimodal evidence available due to temporary generation "
                f"backend failure: {exc}"
            )

        generation_examples.append(
            GenerationExample(
                query=item.query,
                answer=answer,
                reference_answer=item.reference_answer,
                context_chunks=[row["text"] for row in rows[:8]],
            )
        )

    retrieval_metrics = compute_retrieval_metrics(retrieval_examples, k_values=[1, 3, 5, 8])
    generation_metrics = compute_generation_metrics(generation_examples, include_bertscore=True)
    rag_metrics = compute_rag_metrics(generation_examples)
else:
    retrieval_metrics = {
        "precision@1": None,
        "precision@3": None,
        "precision@5": None,
        "precision@8": None,
        "recall@1": None,
        "recall@3": None,
        "recall@5": None,
        "recall@8": None,
        "f1@1": None,
        "f1@3": None,
        "f1@5": None,
        "f1@8": None,
        "ndcg@1": None,
        "ndcg@3": None,
        "ndcg@5": None,
        "ndcg@8": None,
        "mrr": None,
        "placeholder_note": "Populate by setting RUN_FULL_EVAL=True and adding multimodal relevance labels.",
    }
    generation_metrics = {
        "exact_match": None,
        "bleu": None,
        "rouge1": None,
        "rouge2": None,
        "rougeL": None,
        "meteor": None,
        "bertscore_precision": None,
        "bertscore_recall": None,
        "bertscore_f1": None,
        "placeholder_note": "Populate by setting RUN_FULL_EVAL=True in explicit execution phase.",
    }
    rag_metrics = {
        "faithfulness": None,
        "context_precision": None,
        "context_recall": None,
        "answer_relevancy": None,
        "judge_groundedness": None,
        "judge_relevance": None,
        "judge_hallucination": None,
        "judge_completeness": None,
        "placeholder_note": "Populate by setting RUN_FULL_EVAL=True in explicit execution phase.",
    }

pd.DataFrame([retrieval_metrics])
pd.DataFrame([generation_metrics])
pd.DataFrame([rag_metrics])

# %% [markdown]
# ## Step 6: LLM-as-a-Judge (Multimodal)
#
# We explicitly evaluate multimodal evidence quality and answer grounding using `granite4.1:8b`.

# %%
# Input: sample multimodal retrieval rows and generated answer.
# Output: judge payloads.
# Logic: run retrieval-quality and groundedness graders on multimodal context.
# Complexity: O(judge_calls).
if RUN_FULL_EVAL and sample_rows:
    judge_retrieval_payload = grade_retrieval_quality(sample_query, sample_rows)
    judge_ground_payload = grade_groundedness(
        query=sample_query,
        answer=generation_examples[0].answer if generation_examples else "",
        context="\n\n".join(row["text"] for row in sample_rows[:8]),
    )
else:
    judge_retrieval_payload = {
        "retrieval_quality": None,
        "reason": "Placeholder until execution phase.",
        "missing_aspects": [],
    }
    judge_ground_payload = {
        "groundedness": None,
        "hallucination_risk": None,
        "relevance": None,
        "completeness": None,
        "reason": "Placeholder until execution phase.",
    }

judge_retrieval_payload, judge_ground_payload

# %% [markdown]
# ## Step 7: Persist Multimodal Artifacts
#
# This implementation phase writes placeholder-capable schemas.
# Real outputs are populated during explicit execution.

# %%
# Input: multimodal metric payloads and judge outputs.
# Output: saved multimodal metrics JSON and summary table.
# Logic: persist stable artifact contracts for downstream reporting.
# Complexity: O(payload_size).
artifact = {
    "mode": "placeholder" if not RUN_FULL_EVAL else "executed",
    "retrieval_metrics": retrieval_metrics,
    "generation_metrics": generation_metrics,
    "rag_metrics": rag_metrics,
    "judge_retrieval": judge_retrieval_payload,
    "judge_groundedness": judge_ground_payload,
    "asset_counts": {
        "images": len(image_paths),
        "tables": len(table_paths),
        "documents": len(multimodal_docs),
        "chunks": len(multimodal_chunks),
        "eval_queries": len(multimodal_eval_queries),
    },
    "notes": {
        "phase": "implementation_only_no_execution" if not RUN_FULL_EVAL else "executed",
        "ocr_model": settings.multimodal_ocr_model,
        "ocr_cli_timeout_seconds": settings.ocr_cli_timeout_seconds,
        "ocr_cli_allow_fallback": settings.ocr_cli_allow_fallback,
        "judge_model": settings.judge_model,
    },
}

save_json(artifact, settings.metrics_dir / "nb08_multimodal_rag_metrics.json")

pd.DataFrame(
    [
        {"category": "retrieval", **retrieval_metrics},
        {"category": "generation", **generation_metrics},
        {"category": "rag", **rag_metrics},
    ]
).to_csv(settings.tables_dir / "nb08_multimodal_rag_summary.csv", index=False)

print("Saved NB08 multimodal artifacts (placeholders unless RUN_FULL_EVAL=True).")

# %% [markdown]
# ## Post-Run Result Analysis Template (Populate After Execution)
# - Interpret retrieval gains/losses attributable to multimodal ingestion.
# - Interpret generation and RAG metric behavior with multimodal context.
# - Interpret judge scores for groundedness/hallucination under multimodal evidence.
# - Analyze OCR/table extraction error patterns and their impact on quality.
# - Summarize latency and operational tradeoffs vs text-only RAG.
