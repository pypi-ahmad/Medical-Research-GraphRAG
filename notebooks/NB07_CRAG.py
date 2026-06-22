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
# # NB07: Corrective RAG (CRAG) for Biomedical Questions
#
# ## What
# Corrective RAG (CRAG) is a quality-controlled retrieval-generation workflow that explicitly grades retrieval quality and applies corrective actions when retrieval is weak.
#
# ## Why
# Biomedical QA is high-risk for hallucination when retrieval quality is low. CRAG reduces this risk by introducing:
# - explicit retrieval grading,
# - query correction/retrieval retry,
# - fallback evidence acquisition,
# - post-generation grounding verification.
#
# ## When
# Use CRAG when:
# - retrieval quality is inconsistent across question types,
# - domain coverage is sparse or long-tail,
# - groundedness requirements are strict (medical safety).
#
# ## Tradeoffs
# - Better reliability than one-pass RAG.
# - Increased complexity and latency due to corrective loops.
# - Requires threshold tuning and route observability.
#
# ## Alternatives
# - **Single-pass RAG**: lower latency, less robust when retrieval fails.
# - **Agentic RAG without explicit corrective policy**: flexible, but can be harder to audit against deterministic correction criteria.
# - **Rerank-only pipelines**: helpful but may still fail when initial recall is weak.
#
# ## Production Considerations
# - Track route frequencies (`accept`, `correct`, `web_fallback`) as health signals.
# - Cap correction attempts to control cost.
# - Use strict model output schemas for graders.
# - Log complete state traces for replay and incident analysis.

# %% [markdown]
# ## Definition and Core Concepts
# - **CRAG (Corrective RAG)**: retrieval-quality-aware RAG with explicit corrective branches.
# - **Retrieval Grader**: a quality gate deciding whether to accept, correct, or fallback.
# - **Corrective Loop**: bounded retries that improve query/context before generation.
# - **Verification Gate**: post-answer groundedness check with optional corrective fallback.
#
# CRAG is a reliability overlay, not a replacement for baseline retrieval components.

# %% [markdown]
# ## Why CRAG Was Developed
# One-pass RAG pipelines often fail silently when retrieval quality is weak.
# CRAG introduces deterministic control points so failures become observable and recoverable.

# %% [markdown]
# ## What Traditional RAG Limitation It Solves
# Traditional RAG assumes retrieved context is sufficient. CRAG addresses:
# - low-quality retrieval acceptance,
# - lack of corrective query reformulation,
# - weak fallback behavior under sparse evidence,
# - insufficient post-answer grounding verification.

# %% [markdown]
# ## Architecture Explanation
# - `Hybrid Retriever`: initial candidate retrieval.
# - `Judge Retrieval Grader`: quality score + missing-aspect hints.
# - `Query Correction`: reformulate query using grader feedback.
# - `Web Fallback`: bounded external evidence path for unresolved gaps.
# - `Answer + Verify`: generate then verify groundedness before final response.

# %% [markdown]
# ## Workflow Explanation
# 1. Retrieve with hybrid search.
# 2. Grade retrieval quality.
# 3. Branch:
#    - accept context,
#    - retry with corrected query,
#    - fallback to web evidence.
# 4. Generate answer and verify grounding.
# 5. Finalize or perform one bounded corrective verify loop.

# %% [markdown]
# ## Component-by-Component Breakdown
# 1. **State Object (`CRAGState`)**: carries route metadata and trace.
# 2. **Routing Nodes**: retrieval, grading, correction, fallback, generation, verification.
# 3. **Safety Controls**: max correction count and verify-attempt bounds.
# 4. **Observability**: route traces and route summary tables.

# %% [markdown]
# ## Advantages and Disadvantages
# **Advantages**
# - More robust under weak retrieval.
# - Transparent route-level diagnostics.
# - Better guardrails for biomedical grounding.
#
# **Disadvantages**
# - Higher latency/cost from extra model calls.
# - More threshold tuning complexity.
# - Requires disciplined trace logging and monitoring.

# %% [markdown]
# ## Comparison Against Other Implemented Variants
# - **Standard RAG**: fastest path, weakest correction behavior.
# - **Hybrid RAG**: improves retrieval quality, but no explicit corrective controller.
# - **Agentic GraphRAG**: broader orchestration capabilities; CRAG is tighter reliability policy.
# - **CRAG (this notebook)**: explicit quality-driven correction and bounded fallback.

# %% [markdown]
# ## Implementation Decisions in This Project
# - Retrieval uses existing hybrid retriever from NB06.
# - Grading and groundedness use `granite4.1:8b`.
# - Query correction uses generator model with deterministic low-temperature rewrite prompt.
# - Web fallback is bounded and used only after explicit routing criteria.

# %% [markdown]
# ## CRAG Architecture Diagram
#
# ```mermaid
# flowchart TD
#     Q[Biomedical Query] --> R[Hybrid Retriever]
#     R --> G[Judge Retrieval Grader]
#     G -->|High quality| C[Context Build]
#     G -->|Low quality + retries left| QC[Query Correction]
#     QC --> R
#     G -->|Low quality + retries exhausted| WF[Web Fallback]
#     WF --> C
#     C --> A[Answer Generation<br/>granite4.1:8b]
#     A --> V[Judge Verification<br/>granite4.1:8b]
#     V -->|Grounded| F[Final Response]
#     V -->|Not grounded (bounded retry)| WF
# ```

# %% [markdown]
# ## Workflow Rationale
#
# CRAG differs from basic RAG because retrieval quality is a first-class signal controlling downstream behavior.
# This notebook implements CRAG as a standalone additive state machine without modifying existing GraphRAG/Agentic pipelines.

# %%
# Input: existing corpus artifacts and additive CRAG/hybrid modules.
# Output: initialized runtime for standalone CRAG pipeline.
# Logic: load chunks, sparse index, and chroma collection, then build CRAG graph.
# Complexity: O(number_of_chunks + index_build).
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd().parent))

from src.chroma_retriever import get_collection
from src.chunking import build_chunk_lookup, load_chunks
from src.config import settings
from src.crag_pipeline import CRAGResources, build_crag_workflow, crag_mermaid, run_crag_batch
from src.data_pipeline import build_extractive_eval_queries, load_persisted_records
from src.evaluator import (
    GenerationExample,
    RetrievalExample,
    compute_generation_metrics,
    compute_rag_metrics,
    compute_retrieval_metrics,
)
from src.hybrid_retriever import BiomedicalSparseIndex
from src.utils import save_json

RUN_FULL_EVAL = os.getenv("RUN_FULL_EVAL", "false").strip().lower() == "true"
print(f"CRAG acceptance threshold: {settings.crag_acceptance_threshold}")
print(f"CRAG max corrections: {settings.crag_max_corrections}")
print(f"Judge model: {settings.judge_model}")
print(f"RUN_FULL_EVAL={RUN_FULL_EVAL}")

# %% [markdown]
# ## Step 1: Load Corpus + Build Sparse Index

# %%
# Input: persisted MedMentions records/chunks.
# Output: eval query set and sparse index for hybrid retrieval.
# Logic: hydrate data and build lexical retriever needed by CRAG retrieval node.
# Complexity: O(total_tokens).
records = load_persisted_records()
chunks = load_chunks()
chunk_lookup = build_chunk_lookup(chunks)

try:
    collection = get_collection("medmentions_chroma_section_a")
except Exception:
    collection = get_collection("medmentions_chroma")

sparse_index = BiomedicalSparseIndex()
sparse_index.fit(chunks)

eval_queries = build_extractive_eval_queries(
    records=records,
    chunk_lookup=chunk_lookup,
    sample_size=settings.eval_query_count,
)

print(f"Records: {len(records):,}")
print(f"Chunks: {len(chunks):,}")
print(f"Eval queries: {len(eval_queries):,}")

# %% [markdown]
# ## Step 2: Build Standalone CRAG State Machine
#
# We keep this CRAG pipeline separate from existing agentic code to remain strictly additive.

# %%
# Input: chroma collection and sparse index resources.
# Output: compiled CRAG app.
# Logic: instantiate CRAG resource bundle and compile LangGraph workflow.
# Complexity: O(1) graph construction.
resources = CRAGResources(
    chroma_collection=collection,
    sparse_index=sparse_index,
)
app = build_crag_workflow(resources)
print("CRAG workflow compiled.")
print("\nCRAG Mermaid:\n")
print(crag_mermaid())

# %% [markdown]
# ## Step 3: Qualitative CRAG Route Demonstration
#
# We inspect route decisions and traces for representative biomedical queries.

# %%
# Input: demonstration biomedical queries.
# Output: CRAG final states including route traces.
# Logic: run batch CRAG inference and inspect corrective behaviors.
# Complexity: O(num_queries * CRAG_cost).
demo_queries = [
    "What evidence links diabetes with insulin resistance in this corpus?",
    "Summarize findings about hypertension risk factors.",
    "What do retrieved abstracts report about KRAS in pancreatic cancer?",
]

demo_states = run_crag_batch(app, demo_queries)

for idx, state in enumerate(demo_states, start=1):
    print(f"\n==== CRAG Demo {idx} ====")
    print("Query:", state.get("query", ""))
    print("Corrected query:", state.get("corrected_query", ""))
    print("Retrieval grade:", state.get("retrieval_grade", 0.0))
    print("Groundedness:", state.get("groundedness", 0.0))
    print("Route:", state.get("route", ""))
    print("Trace:", " -> ".join(state.get("trace", [])))
    print("Answer preview:", state.get("final_answer", "")[:420], "...")

# %% [markdown]
# ## Step 4: Route Analytics
#
# We summarize corrective behavior frequencies to understand reliability and cost tradeoffs.

# %%
# Input: CRAG run states.
# Output: route analytics table.
# Logic: aggregate route outcomes and corrective attempt counts.
# Complexity: O(num_states).
route_df = pd.DataFrame(
    [
        {
            "query": row.get("query", ""),
            "corrected_query": row.get("corrected_query", ""),
            "retrieval_grade": float(row.get("retrieval_grade", 0.0)),
            "groundedness": float(row.get("groundedness", 0.0)),
            "hallucination_risk": float(row.get("hallucination_risk", 0.0)),
            "completeness": float(row.get("completeness", 0.0)),
            "correction_attempts": int(row.get("correction_attempts", 0)),
            "verify_attempts": int(row.get("verify_attempts", 0)),
            "route": row.get("route", ""),
            "trace": " -> ".join(row.get("trace", [])),
        }
        for row in demo_states
    ]
)
route_df

# %% [markdown]
# ## Step 5: CRAG Evaluation Harness
#
# Required metric families are prepared here:
# - Retrieval metrics: Precision@K, Recall@K, F1, MRR, NDCG
# - Generation metrics: Exact Match, BLEU, ROUGE, METEOR, BERTScore
# - RAG metrics: Faithfulness, Context Precision, Context Recall, Answer Relevancy
# - LLM-as-Judge with `granite4.1:8b`

# %%
# Input: eval query set and CRAG app.
# Output: retrieval/generation/rag metric payloads.
# Logic: run CRAG on eval queries, derive retrieval/generation examples, and score metrics.
# Complexity: O(num_queries * CRAG_workflow_cost).
retrieval_examples: list[RetrievalExample] = []
generation_examples: list[GenerationExample] = []

if RUN_FULL_EVAL:
    states = run_crag_batch(app, [item.query for item in eval_queries])
    for item, state in zip(eval_queries, states):
        retrieved_ids = [row["id"] for row in state.get("retrieval_rows", [])]
        retrieval_examples.append(
            RetrievalExample(retrieved_ids=retrieved_ids, relevant_ids=item.supporting_chunk_ids)
        )

        generation_examples.append(
            GenerationExample(
                query=item.query,
                answer=state.get("final_answer", ""),
                reference_answer=item.reference_answer,
                context_chunks=[row.get("text", "") for row in state.get("retrieval_rows", [])[:8]],
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
        "placeholder_note": "Populate by setting RUN_FULL_EVAL=True in explicit execution phase.",
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
# ## Step 6: Persist CRAG Artifacts
#
# Artifacts are schema-complete now and can be populated with real values during the explicit execution phase.

# %%
# Input: route analytics and metric payloads.
# Output: persisted CRAG tables and JSON metrics payload.
# Logic: save stable CRAG report structure for downstream reporting.
# Complexity: O(payload_size).
route_df.to_csv(settings.tables_dir / "nb07_crag_route_summary.csv", index=False)

crag_payload = {
    "mode": "placeholder" if not RUN_FULL_EVAL else "executed",
    "workflow_mermaid": crag_mermaid(),
    "retrieval_metrics": retrieval_metrics,
    "generation_metrics": generation_metrics,
    "rag_metrics": rag_metrics,
    "route_summary_preview": route_df.head(20).to_dict(orient="records"),
    "notes": {
        "phase": "implementation_only_no_execution" if not RUN_FULL_EVAL else "executed",
        "judge_model": settings.judge_model,
    },
}
save_json(crag_payload, settings.metrics_dir / "nb07_crag_metrics.json")

print("Saved NB07 CRAG artifacts (placeholders unless RUN_FULL_EVAL=True).")

# %% [markdown]
# ## Post-Run Result Analysis Template (Populate After Execution)
# - Analyze route distribution (`accept`, `correct`, `web_fallback`) and what it implies about corpus coverage.
# - Interpret retrieval/generation/RAG metric changes vs non-CRAG baseline.
# - Interpret judge metrics (`granite4.1:8b`) for groundedness, relevance, hallucination, completeness.
# - Quantify latency/cost overhead from corrective loops and fallback paths.
# - Summarize where CRAG improved biomedical reliability and where it did not.
