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
# # NB06: Hybrid RAG for Biomedical QA (Dense + Sparse)
#
# ## What
# Hybrid RAG combines semantic dense retrieval with lexical sparse retrieval to improve biomedical evidence recall and precision.
#
# ## Why
# - Dense retrieval captures semantic similarity but can miss exact biomedical terms, drug names, abbreviations, and measurement strings.
# - Sparse retrieval captures lexical precision but can miss paraphrased evidence.
# - Combining both channels reduces failure modes common in medical literature QA.
#
# ## When
# Use Hybrid RAG when queries include mixed patterns:
# - domain abbreviations (`HTN`, `CKD`, `DM`)
# - exact biomarkers, dosages, gene names, ICD/UMLS tokens
# - semantic phrasing variation between question and evidence
#
# ## Tradeoffs
# - Higher engineering complexity than dense-only retrieval.
# - Additional tuning for dense/sparse fusion weights.
# - Potential latency increase due to dual retrieval channels.
#
# ## Alternatives
# - **Dense-only RAG**: simpler, but weaker lexical precision for strict biomedical identifiers.
# - **Sparse-only RAG**: strong literal matching, weaker semantic generalization.
# - **Cross-encoder rerank only**: improves ranking but still depends on candidate recall quality.
#
# ## Production Considerations
# - Track dense-vs-sparse contribution by query class.
# - Maintain biomedical abbreviation expansion and vocabulary drift monitoring.
# - Monitor retrieval latency and set backend-specific SLOs.
# - Keep fusion strategy configurable and versioned.

# %% [markdown]
# ## Definition and Core Concepts
# - **Dense retrieval**: semantic nearest-neighbor retrieval from embedding vectors.
# - **Sparse retrieval**: lexical retrieval based on exact/near-exact token overlap statistics.
# - **Fusion**: combine dense and sparse rankings (weighted score fusion or RRF) for a better overall candidate set.
#
# Hybrid RAG is not a replacement for GraphRAG. It is a retrieval improvement layer that can feed graph-aware context expansion.

# %% [markdown]
# ## Why Hybrid RAG Was Developed
# Biomedical questions contain mixed evidence signals:
# - semantic paraphrases (best handled by dense retrieval),
# - strict lexical strings such as genes, dosages, acronyms, UMLS-like tokens (best handled by sparse retrieval).
#
# Dense-only systems often miss exact terminology, while sparse-only systems miss semantic variants. Hybrid retrieval addresses this mismatch.

# %% [markdown]
# ## What Traditional RAG Limitation It Solves
# Standard single-channel retrieval can fail under:
# - term mismatch (`myocardial infarction` vs `heart attack`),
# - abbreviation-heavy clinical phrasing (`DM`, `CKD`, `HTN`),
# - numerically or lexically precise evidence requirements.
#
# Hybrid retrieval reduces these failures by preserving both semantic and lexical recall.

# %% [markdown]
# ## Architecture Explanation
# - `Query -> Dense Retriever`: captures semantic neighborhood.
# - `Query -> Sparse Retriever`: captures lexical exactness.
# - `Fusion`: combines evidence channels into one ranking surface.
# - `Top-K Context`: sent to generator and judge for answering and groundedness checks.

# %% [markdown]
# ## Workflow Explanation
# - Build sparse index from existing chunks.
# - Query both dense and sparse retrievers.
# - Fuse and rank candidates.
# - Evaluate retrieval metrics first, then generation/RAG metrics.
# - Run judge scoring for retrieval quality and groundedness.

# %% [markdown]
# ## Component-by-Component Breakdown
# 1. **Sparse Index (`BiomedicalSparseIndex`)**: BM25-style lexical retrieval with biomedical abbreviation expansion.
# 2. **Dense Index (Chroma)**: embedding-based semantic search.
# 3. **Fusion Function**: weighted score blending or optional RRF.
# 4. **Evaluation Layer**: retrieval, generation, RAG, and judge metrics in one contract.

# %% [markdown]
# ## Advantages and Disadvantages
# **Advantages**
# - Higher retrieval robustness for biomedical text.
# - Better recall on strict terms without dropping semantic coverage.
# - Works with existing vector stores without architecture replacement.
#
# **Disadvantages**
# - More knobs to tune (weights, candidate depth, fusion method).
# - Extra retrieval compute and latency.
# - Requires sparse index refresh discipline.

# %% [markdown]
# ## Comparison Against Other Implemented Variants
# - **Standard RAG**: simpler, but weaker lexical-semantic balance.
# - **CRAG**: adds correction control flow after retrieval quality checks.
# - **Agentic GraphRAG**: adds routing/tool orchestration and graph traversal.
# - **Hybrid RAG (this notebook)**: focuses specifically on improving candidate retrieval quality.

# %% [markdown]
# ## Implementation Decisions in This Project
# - Dense channel: Chroma + `qwen3-embedding:4b`.
# - Sparse channel: in-process BM25-style scorer with abbreviation expansion.
# - Fusion default: weighted score fusion using project-configured weights.
# - Evaluation: identical metric schema as the main project for apples-to-apples comparisons.

# %% [markdown]
# ## Architecture Diagram
#
# ```mermaid
# flowchart LR
#     Q[Biomedical Query] --> D[Dense Retriever<br/>qwen3-embedding:4b]
#     Q --> S[Sparse Retriever<br/>BM25-style lexical]
#     D --> F[Fusion Layer<br/>Weighted or RRF]
#     S --> F
#     F --> C[Top-K Context]
#     C --> G[Generator<br/>granite4.1:8b]
#     C --> J[Judge<br/>granite4.1:8b]
# ```

# %% [markdown]
# ## Workflow Diagram
#
# ```mermaid
# flowchart TD
#     A[Load MedMentions chunks + Chroma collection] --> B[Build sparse biomedical index]
#     B --> C[Run dense retrieval]
#     B --> D[Run sparse retrieval]
#     C --> E[Fuse rankings]
#     D --> E
#     E --> F[Evaluate retrieval metrics]
#     E --> G[Generate answers]
#     G --> H[Compute generation + RAG metrics]
#     G --> I[Judge evaluation]
# ```

# %%
# Input: project modules and persisted artifacts.
# Output: initialized runtime for hybrid biomedical retrieval experiments.
# Logic: load dependencies and configure notebook-level constants.
# Complexity: O(1).
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path.cwd().parent))

from src.chroma_retriever import get_collection
from src.chunking import build_chunk_lookup, load_chunks
from src.config import settings
from src.data_pipeline import build_extractive_eval_queries, load_persisted_records
from src.evaluator import (
    GenerationExample,
    RetrievalExample,
    compute_generation_metrics,
    compute_rag_metrics,
    compute_retrieval_metrics,
)
from src.llm_judge import grade_groundedness, grade_retrieval_quality
from src.hybrid_retriever import BiomedicalSparseIndex, hybrid_search
from src.utils import save_json

RUN_FULL_EVAL = os.getenv("RUN_FULL_EVAL", "false").strip().lower() == "true"
print(f"Embedding model: {settings.embedding_model}")
print(f"Generator model: {settings.generator_model}")
print(f"Judge model: {settings.judge_model}")
print(f"RUN_FULL_EVAL={RUN_FULL_EVAL}")

# %% [markdown]
# ## Step 1: Load Existing Corpus Artifacts
#
# We reuse the existing real MedMentions artifacts and do not rewrite upstream pipelines.

# %%
# Input: persisted records/chunks generated in prior notebooks.
# Output: in-memory records, chunk objects, and eval query definitions.
# Logic: hydrate canonical corpus artifacts for additive hybrid retrieval.
# Complexity: O(number_of_records + number_of_chunks).
records = load_persisted_records()
chunks = load_chunks()
chunk_lookup = build_chunk_lookup(chunks)

try:
    collection = get_collection("medmentions_chroma_section_a")
except Exception:
    collection = get_collection("medmentions_chroma")

eval_queries = build_extractive_eval_queries(
    records=records,
    chunk_lookup=chunk_lookup,
    sample_size=settings.eval_query_count,
)

print(f"Records: {len(records):,}")
print(f"Chunks: {len(chunks):,}")
print(f"Eval queries: {len(eval_queries):,}")

# %% [markdown]
# ## Step 2: Build Sparse Biomedical Index
#
# Sparse retrieval here uses BM25-style lexical ranking with biomedical abbreviation expansion.

# %%
# Input: chunk records.
# Output: sparse biomedical index.
# Logic: tokenize corpus, compute document frequencies, and store term-frequency maps.
# Complexity: O(total_tokens_in_corpus).
sparse_index = BiomedicalSparseIndex(k1=1.5, b=0.75)
sparse_index.fit(chunks)
print(f"Sparse index size: {sparse_index.size:,}")

# %% [markdown]
# ## Step 3: Hybrid Retrieval Function
#
# We combine dense and sparse channels with weighted score fusion.

# %%
# Input: natural-language biomedical query.
# Output: fused top-k retrieval rows.
# Logic: call dense and sparse retrievers then merge scores via configurable weights.
# Complexity: O(dense_query + sparse_query + fusion).
def retrieve_hybrid(query: str, top_k: int = 8) -> list[dict]:
    return hybrid_search(
        collection=collection,
        sparse_index=sparse_index,
        query=query,
        top_k=top_k,
        dense_weight=settings.hybrid_dense_weight,
        sparse_weight=settings.hybrid_sparse_weight,
        use_rrf=False,
    )


sample_query = "What evidence links diabetes mellitus with insulin resistance and glycemic control?"
sample_rows = retrieve_hybrid(sample_query, top_k=5)
for idx, row in enumerate(sample_rows, start=1):
    print(f"[{idx}] id={row['id']} score={row['score']:.4f} sources={row.get('sources', [])}")
    print(row['text'][:220], "...\n")

# %% [markdown]
# ## Step 4: Retrieval Metrics Harness
#
# Required metrics:
# - Precision@K
# - Recall@K
# - F1 Score
# - MRR
# - NDCG

# %%
# Input: eval query set with relevant chunk IDs.
# Output: retrieval metric dictionary.
# Logic: execute retrieval for each query and aggregate ranking metrics.
# Complexity: O(num_queries * retrieval_cost).
retrieval_examples: list[RetrievalExample] = []
retrieval_payload: list[dict] = []

if RUN_FULL_EVAL:
    for item in eval_queries:
        rows = retrieve_hybrid(item.query, top_k=settings.top_k_retrieval)
        ids = [row["id"] for row in rows]
        retrieval_examples.append(RetrievalExample(retrieved_ids=ids, relevant_ids=item.supporting_chunk_ids))
        retrieval_payload.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "retrieved_ids": ids,
                "relevant_ids": item.supporting_chunk_ids,
            }
        )

    retrieval_metrics = compute_retrieval_metrics(retrieval_examples, k_values=[1, 3, 5, 8])
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

pd.DataFrame([retrieval_metrics])

# %% [markdown]
# ## Step 5: Generation + RAG Metrics Harness
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
# Input: generation subset of eval queries.
# Output: generation and RAG metric dictionaries.
# Logic: generate answers from fused context then compute lexical/semantic/judge metrics.
# Complexity: O(num_generation_queries * (LLM_cost + metric_cost)).
generation_examples: list[GenerationExample] = []
generation_payload: list[dict] = []

if RUN_FULL_EVAL:
    subset = eval_queries[: min(settings.generation_eval_count, len(eval_queries))]
    for item in subset:
        rows = retrieve_hybrid(item.query, top_k=settings.top_k_retrieval)
        context = "\n\n".join(f"[{i+1}] {row['text'][:900]}" for i, row in enumerate(rows[:8]))

        try:
            response = __import__("ollama").chat(
                model=settings.generator_model,
                messages=[
                    {
                        "role": "user",
                        "content": (
                            "Answer this biomedical question using only provided evidence. "
                            "If insufficient evidence exists, say so explicitly.\n\n"
                            f"Question: {item.query}\n\nEvidence:\n{context}"
                        ),
                    }
                ],
                options={"temperature": 0.2},
            )
            answer = response["message"]["content"]
        except Exception as exc:
            answer = (
                "Insufficient evidence available due to a temporary generation backend "
                f"failure: {exc}"
            )
        generation_examples.append(
            GenerationExample(
                query=item.query,
                answer=answer,
                reference_answer=item.reference_answer,
                context_chunks=[row["text"] for row in rows[:8]],
            )
        )
        generation_payload.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "answer": answer,
                "reference_answer": item.reference_answer,
            }
        )

    generation_metrics = compute_generation_metrics(generation_examples, include_bertscore=True)
    rag_metrics = compute_rag_metrics(generation_examples)
else:
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

pd.DataFrame([generation_metrics])
pd.DataFrame([rag_metrics])

# %% [markdown]
# ## Step 6: LLM-as-a-Judge with `granite4.1:8b`
#
# We explicitly score:
# - retrieval quality
# - answer groundedness
# - hallucination risk
# - completeness

# %%
# Input: sample query, hybrid retrieval rows, and generated answer (or placeholder).
# Output: judge payload examples.
# Logic: call judge prompts through structured JSON judges.
# Complexity: O(judge_calls).
if RUN_FULL_EVAL and generation_payload:
    judge_retrieval_payload = grade_retrieval_quality(
        query=generation_payload[0]["query"],
        docs=sample_rows,
    )
    judge_ground_payload = grade_groundedness(
        query=generation_payload[0]["query"],
        answer=generation_payload[0]["answer"],
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
# ## Step 7: Persist Placeholder/Real Artifacts
#
# This implementation phase writes schema-complete payloads.
# During the execution phase, placeholders are replaced with real outputs.

# %%
# Input: retrieval/generation/rag/judge payloads.
# Output: saved JSON/CSV artifacts for README and downstream analysis.
# Logic: persist all results in stable schemas regardless of execution mode.
# Complexity: O(payload_size).
artifact = {
    "mode": "placeholder" if not RUN_FULL_EVAL else "executed",
    "retrieval_metrics": retrieval_metrics,
    "generation_metrics": generation_metrics,
    "rag_metrics": rag_metrics,
    "judge_retrieval": judge_retrieval_payload,
    "judge_groundedness": judge_ground_payload,
    "notes": {
        "phase": "implementation_only_no_execution" if not RUN_FULL_EVAL else "executed",
        "judge_model": settings.judge_model,
    },
}

save_json(artifact, settings.metrics_dir / "nb06_hybrid_rag_metrics.json")

pd.DataFrame(
    [
        {"category": "retrieval", **retrieval_metrics},
        {"category": "generation", **generation_metrics},
        {"category": "rag", **rag_metrics},
    ]
).to_csv(settings.tables_dir / "nb06_hybrid_rag_summary.csv", index=False)

print("Saved NB06 artifacts (placeholders unless RUN_FULL_EVAL=True).")

# %% [markdown]
# ## Post-Run Result Analysis Template (Populate After Execution)
# - Analyze actual retrieval deltas vs dense-only baseline: Precision@K, Recall@K, F1, MRR, NDCG.
# - Analyze generation deltas: Exact Match, BLEU, ROUGE, METEOR, BERTScore.
# - Analyze RAG metrics: Faithfulness, Context Precision, Context Recall, Answer Relevancy.
# - Analyze judge scores (`granite4.1:8b`): Groundedness, Relevance, Hallucination, Completeness.
# - Explain observed latency and complexity tradeoffs from dual retrieval channels.
# - Conclude when Hybrid RAG is worth enabling in this medical stack.
