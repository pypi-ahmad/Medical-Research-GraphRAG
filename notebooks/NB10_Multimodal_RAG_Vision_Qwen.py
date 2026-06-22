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
# # NB10: Multimodal RAG with Vision Model (`qwen3.5:4b`)
#
# ## What
# This notebook implements a second multimodal RAG pathway where image evidence is extracted by a vision-capable model (`qwen3.5:4b`) before retrieval and generation.
#
# ## Why
# - OCR captures visible text but may miss visual trend semantics.
# - A vision model can interpret chart structure, comparative patterns, and clinically relevant visual cues.
# - This complements OCR-first pipelines and enables side-by-side multimodal strategy benchmarking.
#
# ## When
# Use this approach when:
# - charts/figures contain trend-level evidence beyond literal text,
# - clinical interpretation depends on visual patterns,
# - you need stronger figure-grounded context expansion.
#
# ## Tradeoffs
# - Higher model complexity than OCR-only extraction.
# - Vision extraction can be sensitive to prompt design.
# - Potential latency increase relative to OCR-only ingestion.
#
# ## Alternatives
# - **OCR-only multimodal RAG**: simpler, but misses non-text visual semantics.
# - **Manual curation of figure captions**: precise but not scalable.
# - **Direct end-to-end VQA without retrieval**: less traceable and harder to audit for grounding.
#
# ## Production Considerations
# - Version vision prompts and outputs together with source assets.
# - Track hallucination risk separately for vision-derived context.
# - Preserve provenance metadata (`vision_model`, asset path, extraction timestamp).
# - Add quality gates for low-information or low-resolution images.

# %% [markdown]
# ## Definition and Core Concepts
# - **Vision-model multimodal RAG**: non-text image understanding converted into retrievable textual evidence.
# - **Vision extractor**: `qwen3.5:4b` interprets chart/diagram structure and trend semantics.
# - **Evidence transformation**: extracted insights become chunkable/indexable text.
# - **Provenance**: each chunk records vision model lineage for auditability.

# %% [markdown]
# ## Why This Technique Was Developed
# OCR captures visible text, but biomedical figures often carry critical non-text signals (trend slope, separation, trajectory).
# Vision extraction was developed to capture these signals in retrieval-ready form.

# %% [markdown]
# ## What Traditional RAG Limitation It Solves
# Text-only RAG and OCR-only RAG can miss non-verbatim visual semantics.
# Vision-model extraction reduces this blind spot for chart-heavy biomedical evidence.

# %% [markdown]
# ## Architecture Explanation
# - `Images -> Vision Model`: structured visual interpretation.
# - `Tables -> Parser`: tabular evidence path.
# - `Merged Evidence Corpus`: normalized textual evidence from both modalities.
# - `Chunk + Embed + Retrieve`: same retriever pipeline used for downstream answering and judging.

# %% [markdown]
# ## Workflow Explanation
# 1. Discover multimodal assets.
# 2. Extract image evidence with `qwen3.5:4b`.
# 3. Chunk and index in dedicated vision collection.
# 4. Retrieve for biomedical questions.
# 5. Evaluate retrieval/generation/RAG/judge outputs.

# %% [markdown]
# ## Component-by-Component Breakdown
# 1. **Vision Extraction Node**: converts image semantics to text evidence.
# 2. **Chunk/Index Node**: creates retrieval-ready multimodal chunks.
# 3. **Retrieval Node**: returns top-k vision-derived evidence.
# 4. **Evaluation Node**: produces consistent metric families for comparison.

# %% [markdown]
# ## Advantages and Disadvantages
# **Advantages**
# - Captures visual semantics beyond OCR text.
# - Better for graph/trend interpretation tasks.
# - Clean separation from OCR-specific experiments for benchmarking.
#
# **Disadvantages**
# - Prompt sensitivity and extraction variance.
# - Higher runtime complexity than OCR-only path.
# - Requires stronger hallucination monitoring.

# %% [markdown]
# ## Comparison Against Other Implemented Variants
# - **NB08**: generalized multimodal framing.
# - **NB09**: OCR operations focus.
# - **NB10 (this notebook)**: vision semantics focus with `qwen3.5:4b`.
# - **CRAG/Hybrid**: retrieval control and recall improvements on text side.

# %% [markdown]
# ## Implementation Decisions in This Project
# - Dedicated Chroma collection isolates vision experiments from OCR experiments.
# - Judge remains `granite4.1:8b` for cross-notebook consistency.
# - Placeholder-first persistence keeps execution and implementation phases separated.

# %% [markdown]
# ## Architecture Diagram
#
# ```mermaid
# flowchart LR
#     A[Biomedical Images] --> B[Vision Extraction<br/>qwen3.5:4b]
#     T[Biomedical Tables] --> C[Table-to-Text]
#     B --> D[Vision Evidence Corpus]
#     C --> D
#     D --> E[Chunk + Embed<br/>qwen3-embedding:4b]
#     E --> F[Chroma Index: medical_multimodal_qwen_vision]
#     F --> G[Multimodal Retrieval]
#     G --> H[Generation<br/>granite4.1:8b]
#     G --> I[Judge<br/>granite4.1:8b]
# ```

# %% [markdown]
# ## Workflow Diagram
#
# ```mermaid
# flowchart TD
#     A[Discover multimodal assets] --> B[Vision extraction for images]
#     B --> C[Build multimodal documents]
#     C --> D[Chunk and index in Chroma]
#     D --> E[Retrieve context for biomedical query]
#     E --> F[Generate answer]
#     F --> G[Evaluate retrieval/generation/RAG metrics]
#     F --> H[Judge evaluation]
# ```

# %%
# Input: vision multimodal and evaluation modules.
# Output: initialized runtime for Qwen vision multimodal pipeline.
# Logic: wire model configuration and import pipeline helpers.
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
    load_pmc_multimodal_manifest,
    persist_multimodal_manifest,
)
from src.multimodal_vision_rag import (
    build_vision_multimodal_documents,
    index_vision_chunks_to_chromadb,
    vision_documents_to_chunks,
    vision_multimodal_search,
)
from src.utils import save_json

RUN_FULL_EVAL = os.getenv("RUN_FULL_EVAL", "false").strip().lower() == "true"
print(f"Vision extraction model: {settings.multimodal_vision_model}")
print(f"Embedding model: {settings.embedding_model}")
print(f"Generator model: {settings.generator_model}")
print(f"Judge model: {settings.judge_model}")
print(f"RUN_FULL_EVAL={RUN_FULL_EVAL}")

# %% [markdown]
# ## Step 1: Discover Multimodal Assets
#
# Asset roots:
# - images: `data/multimodal/images/`
# - tables: `data/multimodal/tables/`

# %%
# Input: multimodal root directories.
# Output: discovered image/table asset paths.
# Logic: collect files with supported image/table suffix patterns.
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
# ## Step 2: Build Vision-Derived Multimodal Documents
#
# Images are interpreted by `qwen3.5:4b` to produce medically relevant textual evidence.

# %%
# Input: image/table asset paths.
# Output: multimodal documents with vision-derived text evidence.
# Logic: call vision extractor for images and table parser for tables.
# Complexity: O(vision_extraction + table_parse_cost).
if RUN_FULL_EVAL:
    multimodal_docs = build_vision_multimodal_documents(
        image_paths=image_paths,
        table_paths=table_paths,
        vision_model=settings.multimodal_vision_model,
    )
else:
    multimodal_docs = []

print(f"Vision multimodal documents built: {len(multimodal_docs)}")

# %% [markdown]
# ## Step 3: Chunk and Index Vision Evidence
#
# We index into a dedicated collection to keep comparison with OCR pipelines clean.

# %%
# Input: vision multimodal documents.
# Output: chunk list and optional collection handle.
# Logic: chunk multimodal text, embed, and persist in Chroma collection.
# Complexity: O(chunking + embedding + indexing).
if RUN_FULL_EVAL and multimodal_docs:
    multimodal_chunks = vision_documents_to_chunks(multimodal_docs)
    multimodal_collection = index_vision_chunks_to_chromadb(
        multimodal_chunks,
        collection_name="medical_multimodal_qwen_vision",
        batch_size=64,
    )
    persist_multimodal_manifest(
        multimodal_docs,
        multimodal_chunks,
        out_path=settings.multimodal_dir / "multimodal_manifest_qwen_vision.json",
    )
else:
    multimodal_chunks = []
    multimodal_collection = None

print(f"Vision multimodal chunks: {len(multimodal_chunks)}")

# %% [markdown]
# ## Step 4: Vision Retrieval Demonstration
#
# Retrieve relevant multimodal evidence from the vision-derived index.

# %%
# Input: biomedical query and vision multimodal collection.
# Output: top-k retrieved evidence rows.
# Logic: dense vector retrieval over vision-derived multimodal chunks.
# Complexity: O(multimodal_dense_query).
sample_query = "What visual evidence suggests progression patterns in diabetes biomarkers?"

if RUN_FULL_EVAL and multimodal_collection is not None:
    sample_rows = vision_multimodal_search(multimodal_collection, sample_query, top_k=6)
else:
    sample_rows = []

if sample_rows:
    for idx, row in enumerate(sample_rows, start=1):
        print(
            f"[{idx}] id={row['id']} score={row['score']:.4f} "
            f"modality={row['metadata'].get('modality', '')}"
        )
        print(row["text"][:260], "...\n")
else:
    print("Placeholder: vision retrieval outputs will be populated during explicit execution phase.")

# %% [markdown]
# ## Step 5: Evaluation Harness
#
# Retrieval metrics:
# - Precision@K
# - Recall@K
# - F1
# - MRR
# - NDCG
#
# Generation metrics:
# - Exact Match
# - BLEU
# - ROUGE
# - METEOR
# - BERTScore
#
# RAG metrics:
# - Faithfulness
# - Context Precision
# - Context Recall
# - Answer Relevancy

# %%
# Input: multimodal docs/chunks and optional PMC asset manifest.
# Output: retrieval/generation/rag metric payloads.
# Logic: run retrieval + generation loops and compute required metric families.
# Complexity: O(num_queries * (retrieval + generation + metrics)).
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
        rows = vision_multimodal_search(multimodal_collection, item.query, top_k=settings.top_k_retrieval)
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
                            "Answer the biomedical question using vision-derived multimodal evidence only. "
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
                "Insufficient vision-derived evidence available due to temporary generation "
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
# We evaluate retrieval and generation quality with `granite4.1:8b`.

# %%
# Input: sample retrieval rows and candidate answer.
# Output: judge payload dictionaries.
# Logic: run judge prompts for retrieval quality and answer groundedness.
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
# Placeholder contracts are persisted now; real values are filled during explicit execution.

# %%
# Input: metric payloads, judge outputs, and run metadata.
# Output: saved JSON and CSV artifact files.
# Logic: persist standardized outputs for downstream reporting and comparisons.
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
        "vision_model": settings.multimodal_vision_model,
        "judge_model": settings.judge_model,
    },
}

save_json(artifact, settings.metrics_dir / "nb10_multimodal_qwen_vision_metrics.json")

pd.DataFrame(
    [
        {"category": "retrieval", **retrieval_metrics},
        {"category": "generation", **generation_metrics},
        {"category": "rag", **rag_metrics},
    ]
).to_csv(settings.tables_dir / "nb10_multimodal_qwen_vision_summary.csv", index=False)

print("Saved NB10 artifacts (placeholders unless RUN_FULL_EVAL=True).")

# %% [markdown]
# ## Post-Run Result Analysis Template (Populate After Execution)
# - Compare vision-derived retrieval quality against OCR-derived retrieval.
# - Interpret generation/RAG/judge metric shifts under vision-derived context.
# - Analyze hallucination patterns specific to vision extraction.
# - Quantify latency/complexity tradeoffs of adding a vision model path.
# - Conclude where vision extraction is beneficial in biomedical QA pipelines.
