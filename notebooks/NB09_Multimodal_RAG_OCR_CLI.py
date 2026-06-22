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
# # NB09: Multimodal RAG with CLI OCR (`ollama run glm-ocr`)
#
# ## What
# This notebook builds a multimodal RAG pipeline where biomedical image evidence is extracted with a CLI-first OCR path:
# - primary OCR path: `ollama run glm-ocr`
# - fallback OCR path: Ollama Python API (`ollama.chat`) only if CLI call fails.
#
# ## Why
# - CLI-based OCR is easy to operationalize in shell-driven ETL jobs.
# - It keeps OCR behavior explicit and auditable from command history.
# - Fallback keeps pipeline resilience when CLI invocation fails unexpectedly.
#
# ## When
# Use this approach when:
# - research evidence is in chart annotations, legends, or figure panels,
# - your ingestion pipeline is shell-oriented,
# - you need deterministic OCR backend logging per asset.
#
# ## Tradeoffs
# - Adds subprocess management and timeout handling.
# - OCR quality still depends on image resolution and chart complexity.
# - Fallback introduces a second execution path that must be traced.
#
# ## Alternatives
# - **Text-only RAG**: lower complexity, but ignores figure/table evidence.
# - **Vision-model extraction directly**: richer semantics, usually higher cost/latency.
# - **External OCR service**: may improve quality, but adds data-governance and dependency overhead.
#
# ## Production Considerations
# - Persist OCR backend metadata (`ollama_run` vs fallback) for every asset.
# - Set per-asset OCR timeout and retry policy.
# - Add OCR quality QA checks before indexing.
# - Keep PHI-sensitive medical assets in approved storage boundaries.

# %% [markdown]
# ## Definition and Core Concepts
# - **CLI-first OCR Multimodal RAG**: a multimodal pipeline where OCR extraction is executed primarily through shell CLI calls.
# - **Primary backend**: `ollama run glm-ocr`.
# - **Fallback backend**: `ollama.chat` only when CLI fails or times out.
# - **Backend Provenance**: each extracted document records which OCR backend produced it.

# %% [markdown]
# ## Why This Technique Was Developed
# For many engineering teams, ETL is shell-centric. CLI-first OCR is transparent, scriptable, and easier to audit in operational runs.

# %% [markdown]
# ## What Traditional RAG Limitation It Solves
# Text-only RAG ignores figure/table content.
# API-only OCR may hide operational behavior inside application logs.
# CLI-first OCR gives explicit extraction commands and deterministic execution boundaries.

# %% [markdown]
# ## Architecture Explanation
# - `Image -> CLI OCR`: first attempt via `ollama run glm-ocr`.
# - `CLI Decision`: fallback only on failure/timeout.
# - `Table Parse`: tabular evidence converted to text.
# - `Index + Retrieve`: multimodal evidence indexed and retrieved like standard RAG.

# %% [markdown]
# ## Workflow Explanation
# 1. Discover image/table assets.
# 2. Execute CLI OCR with timeout policy.
# 3. Fallback to API only if necessary.
# 4. Persist provenance-rich multimodal docs.
# 5. Evaluate retrieval/generation/RAG/judge metrics.

# %% [markdown]
# ## Component-by-Component Breakdown
# 1. **CLI Command Builder**: deterministic command assembly for OCR calls.
# 2. **Execution Guard**: timeout and return-code handling.
# 3. **Fallback Path**: controlled API OCR recovery.
# 4. **Evaluation Layer**: same quality contracts as other notebooks.

# %% [markdown]
# ## Advantages and Disadvantages
# **Advantages**
# - Operationally explicit and auditable OCR path.
# - Easy shell automation for batch ingestion.
# - Controlled fallback behavior for resilience.
#
# **Disadvantages**
# - Subprocess orchestration complexity.
# - Requires careful timeout/retry policy tuning.
# - OCR quality remains image-dependent.

# %% [markdown]
# ## Comparison Against Other Implemented Variants
# - **NB08**: general multimodal framing.
# - **NB09 (this notebook)**: OCR CLI operational specialization.
# - **NB10**: vision-model specialization for non-text visual semantics.

# %% [markdown]
# ## Implementation Decisions in This Project
# - OCR model fixed to `glm-ocr` unless config override.
# - Fallback enabled by config and fully traceable.
# - Results saved with backend metadata so OCR route quality can be benchmarked later.

# %% [markdown]
# ## Architecture Diagram
#
# ```mermaid
# flowchart LR
#     A[Biomedical Images] --> B[CLI OCR<br/>ollama run glm-ocr]
#     B --> C{CLI Success?}
#     C -->|Yes| D[OCR Text]
#     C -->|No| E[Fallback OCR<br/>Ollama API]
#     E --> D
#     T[Biomedical Tables] --> F[Table-to-Text]
#     D --> G[Multimodal Text Corpus]
#     F --> G
#     G --> H[Chunk + Embed<br/>qwen3-embedding:4b]
#     H --> I[Chroma Multimodal Index]
#     I --> J[Retrieval]
#     J --> K[Generation<br/>granite4.1:8b]
#     J --> L[Judge<br/>granite4.1:8b]
# ```

# %% [markdown]
# ## Workflow Diagram
#
# ```mermaid
# flowchart TD
#     A[Discover image/table assets] --> B[OCR extraction with CLI-first policy]
#     B --> C[Build multimodal documents with provenance metadata]
#     C --> D[Chunk multimodal evidence]
#     D --> E[Index to Chroma collection: medical_multimodal_cli_ocr]
#     E --> F[Retrieve evidence for biomedical query]
#     F --> G[Generate answer with granite4.1:8b]
#     G --> H[Evaluate retrieval, generation, and RAG metrics]
#     G --> I[Judge groundedness and hallucination risk]
# ```

# %%
# Input: multimodal OCR and evaluation modules.
# Output: initialized runtime for CLI OCR multimodal pipeline.
# Logic: load modules and configure run mode for placeholder-safe implementation.
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
# ## Step 1: Discover Biomedical Multimodal Assets
#
# Expected directories:
# - images: `data/multimodal/images/`
# - tables: `data/multimodal/tables/`
#
# Empty directories are supported during implementation-only phase.

# %%
# Input: multimodal asset root path.
# Output: lists of discovered image and table files.
# Logic: scan directory patterns for supported image and table extensions.
# Complexity: O(number_of_files_in_multimodal_dirs).
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
# ## Step 2: Build Multimodal Documents with CLI OCR
#
# This step applies CLI-first OCR policy and records OCR backend provenance in metadata.

# %%
# Input: discovered image/table paths.
# Output: multimodal documents containing extracted OCR/table text.
# Logic: run CLI-first OCR ingestion with fallback policy and create provenance-rich docs.
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
# Chunks are persisted in a dedicated collection for CLI OCR experiments.

# %%
# Input: multimodal documents.
# Output: multimodal chunk list and optional Chroma collection.
# Logic: split documents into chunk units and index with embedding vectors.
# Complexity: O(chunking + embedding + indexing).
if RUN_FULL_EVAL and multimodal_docs:
    multimodal_chunks = multimodal_documents_to_chunks(multimodal_docs)
    multimodal_collection = index_multimodal_chunks_to_chromadb(
        multimodal_chunks,
        collection_name="medical_multimodal_cli_ocr",
        batch_size=64,
    )
    persist_multimodal_manifest(
        multimodal_docs,
        multimodal_chunks,
        out_path=settings.multimodal_dir / "multimodal_manifest_cli_ocr.json",
    )
else:
    multimodal_chunks = []
    multimodal_collection = None

print(f"Multimodal chunks: {len(multimodal_chunks)}")

# %% [markdown]
# ## Step 4: Retrieval Demonstration
#
# Run retrieval over multimodal OCR-derived evidence.

# %%
# Input: biomedical query and multimodal Chroma collection.
# Output: top-k retrieval rows from multimodal evidence index.
# Logic: vector search over indexed multimodal chunks.
# Complexity: O(multimodal_dense_query).
sample_query = "What figure or table evidence reports diabetes biomarker trends?"

if RUN_FULL_EVAL and multimodal_collection is not None:
    sample_rows = multimodal_vector_search(multimodal_collection, sample_query, top_k=6)
else:
    sample_rows = []

if sample_rows:
    for idx, row in enumerate(sample_rows, start=1):
        print(
            f"[{idx}] id={row['id']} score={row['score']:.4f} "
            f"modality={row['metadata'].get('modality', '')} "
            f"ocr_backend={row['metadata'].get('ocr_backend', '')}"
        )
        print(row["text"][:260], "...\n")
else:
    print("Placeholder: retrieval outputs will be populated during explicit execution phase.")

# %% [markdown]
# ## Step 5: Evaluation Harness
#
# Required retrieval metrics:
# - Precision@K
# - Recall@K
# - F1
# - MRR
# - NDCG
#
# Required generation metrics:
# - Exact Match
# - BLEU
# - ROUGE
# - METEOR
# - BERTScore
#
# Required RAG metrics:
# - Faithfulness
# - Context Precision
# - Context Recall
# - Answer Relevancy

# %%
# Input: multimodal docs/chunks and optional PMC asset manifest.
# Output: retrieval/generation/rag metric payload dictionaries.
# Logic: run multimodal retrieval + generation loops and aggregate required metrics.
# Complexity: O(num_queries * (retrieval + generation + scoring)).
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
        retrieved_ids = [row["id"] for row in rows]
        retrieval_examples.append(
            RetrievalExample(retrieved_ids=retrieved_ids, relevant_ids=item.relevant_chunk_ids)
        )

        context = "\n\n".join(f"[{i+1}] {row['text'][:900]}" for i, row in enumerate(rows[:8]))
        try:
            response = __import__("ollama").chat(
                model=settings.generator_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer the biomedical question using multimodal OCR evidence only. "
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
                "Insufficient OCR-derived evidence available due to temporary generation "
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
# ## Step 6: LLM-as-a-Judge
#
# We audit:
# - retrieval quality
# - groundedness
# - hallucination risk
# - completeness
# using `granite4.1:8b`.

# %%
# Input: sample retrieval rows and generated answer.
# Output: judge payload dictionaries for retrieval and grounding quality.
# Logic: run judge prompts over retrieved context and answer text.
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
# ## Step 7: Persist Artifacts
#
# In this phase, schemas are saved with placeholders.
# Real values are populated in explicit execution.

# %%
# Input: metrics payloads, judge payloads, and run metadata.
# Output: persisted JSON/CSV artifacts for reporting.
# Logic: save stable artifact contracts for downstream README/result sections.
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

save_json(artifact, settings.metrics_dir / "nb09_multimodal_ocr_cli_metrics.json")

pd.DataFrame(
    [
        {"category": "retrieval", **retrieval_metrics},
        {"category": "generation", **generation_metrics},
        {"category": "rag", **rag_metrics},
    ]
).to_csv(settings.tables_dir / "nb09_multimodal_ocr_cli_summary.csv", index=False)

print("Saved NB09 artifacts (placeholders unless RUN_FULL_EVAL=True).")

# %% [markdown]
# ## Post-Run Result Analysis Template (Populate After Execution)
# - Compare CLI OCR success-rate vs fallback-rate and explain failure causes.
# - Interpret retrieval/generation/RAG/judge metrics under CLI-first OCR evidence.
# - Analyze latency impact of subprocess OCR and timeout policy.
# - Explain observed biomedical answer quality gains/losses from OCR-derived context.
