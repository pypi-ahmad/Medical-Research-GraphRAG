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
# # NB05: End-to-End Evaluation
#
# ## What
# This notebook computes retrieval, generation, RAG, and LLM-judge metrics for the medical GraphRAG assistant.
#
# ## Why
# A production-grade RAG system needs layered evaluation: retrieval quality, generation quality, grounding quality, and holistic judge-based quality.
#
# ## When
# Use this notebook after indexing and agentic workflow setup are complete.
#
# ## Tradeoffs
# - Deterministic lexical metrics (EM, BLEU, ROUGE, METEOR) are stable but can under-reward semantically-correct paraphrases.
# - LLM-judge metrics capture nuanced quality but may introduce evaluator variance.
# - Full evaluation is compute-intensive; subset evaluation can speed iteration but may increase variance.
#
# ## Alternatives
# - **Why this metric mix**: combines deterministic, semantic, and judge-based perspectives.
# - **Why not accuracy-only evaluation**: single-point metrics hide retrieval and grounding failure modes.
# - **Why not judge-only evaluation**: judge-only scoring can be biased without deterministic anchors.
#
# ## Production Considerations
# - Track metrics over time to detect regressions and drift.
# - Keep evaluation datasets versioned and grounded in real source records.
# - Include failure-case slices (rare entities, long-tail biomedical terms).

# %%
# Input: persisted retrieval/graph artifacts plus agentic app and evaluator modules.
# Output: initialized end-to-end evaluation runtime.
# Logic: load state once and reuse for retrieval + generation + rag metric passes.
# Complexity: O(number_of_chunks + graph size).
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.append(str(Path.cwd().parent))

from src.agentic_rag import AgentResources, build_agentic_workflow, run_agentic_query
from src.chroma_retriever import entity_search, get_collection, reciprocal_rank_fusion, vector_search
from src.chunking import build_chunk_lookup, load_chunks
from src.config import settings
from src.data_pipeline import build_extractive_eval_queries, load_persisted_records
from src.evaluator import (
    GenerationExample,
    RetrievalExample,
    build_evaluation_bundle,
    compute_generation_metrics,
    compute_rag_metrics,
    compute_retrieval_metrics,
)
from src.graph_builder import concept_id_from_query, local_graph_expansion
from src.utils import save_json, save_plot, timed_block

records = load_persisted_records()
chunks = load_chunks()
chunk_lookup = build_chunk_lookup(chunks)
chroma_collection = get_collection("medmentions_chroma_section_a")

with (settings.graph_dir / "entity_graph.pkl").open("rb") as f:
    graph = pickle.load(f)
partition = json.loads((settings.graph_dir / "community_partition.json").read_text(encoding="utf-8"))
summaries = json.loads((settings.graph_dir / "community_summaries.json").read_text(encoding="utf-8"))

resources = AgentResources(
    chroma_collection=chroma_collection,
    chunks=chunks,
    graph=graph,
    partition=partition,
    summaries=summaries,
)
app = build_agentic_workflow(resources)

print(f"Records: {len(records):,}")
print(f"Chunks: {len(chunks):,}")
print("Judge model:", settings.judge_model)

# %% [markdown]
# ## Step 1: Build Grounded Evaluation Queries
#
# Query/reference pairs are extractive and derived from real MedMentions abstracts.

# %%
# Input: records and chunk lookup map.
# Output: grounded eval query list.
# Logic: deterministic extraction of query/reference evidence pairs.
# Complexity: O(records * entities).
eval_queries = build_extractive_eval_queries(
    records=records,
    chunk_lookup=chunk_lookup,
    sample_size=settings.eval_query_count,
)
print("Evaluation query count:", len(eval_queries))

# %% [markdown]
# ## Step 2: Retrieval-Only Pass
#
# Metrics computed:
# - Precision@K
# - Recall@K
# - F1@K
# - MRR
# - NDCG

# %%
# Input: eval query set and GraphRAG retrieval functions.
# Output: retrieval examples for metric computation.
# Logic: vector retrieval + graph-expanded entity retrieval + RRF fusion.
# Complexity: O(num_queries * retrieval_cost).
def retrieval_graphrag(query: str, top_k: int = 8):
    vector_rows = vector_search(chroma_collection, query, top_k=top_k * 2)
    query_concepts = concept_id_from_query(query, chunks)
    local_ctx = local_graph_expansion(graph, query_concepts, hops=settings.local_graph_hops)
    entity_rows = entity_search(chroma_collection, concept_ids=local_ctx.get("nodes", [])[:120], top_k=top_k * 2)
    return reciprocal_rank_fusion({"vector": vector_rows, "entity": entity_rows}, top_k=top_k)


retrieval_examples: list[RetrievalExample] = []
retrieval_payload: list[dict] = []

with timed_block("Run retrieval-only evaluation"):
    for item in eval_queries:
        rows = retrieval_graphrag(item.query, top_k=settings.top_k_retrieval)
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
pd.DataFrame([retrieval_metrics])

# %% [markdown]
# ## Step 3: Generation Pass
#
# Metrics computed:
# - Exact Match
# - BLEU
# - ROUGE
# - METEOR
# - BERTScore

# %%
# Input: generation subset from eval queries.
# Output: generation examples for lexical/semantic scoring.
# Logic: run full agentic pipeline and align outputs with grounded references.
# Complexity: O(num_generation_queries * workflow_cost).
generation_count = min(settings.generation_eval_count, len(eval_queries))
generation_subset = eval_queries[:generation_count]

generation_examples: list[GenerationExample] = []
run_states: list[dict] = []

with timed_block("Run generation evaluation subset"):
    for item in generation_subset:
        state = run_agentic_query(app, item.query)
        run_states.append(state)
        generation_examples.append(
            GenerationExample(
                query=item.query,
                answer=state.get("final_answer", ""),
                reference_answer=item.reference_answer,
                context_chunks=[doc["text"] for doc in state.get("retrieved_docs", [])[:8]],
            )
        )

generation_metrics = compute_generation_metrics(generation_examples, include_bertscore=True)
pd.DataFrame([generation_metrics])

# %% [markdown]
# ## Step 4: RAG + LLM Judge Metrics
#
# RAG metrics computed:
# - Faithfulness
# - Context Precision
# - Context Recall
# - Answer Relevancy
#
# Judge metrics computed using `granite4.1:8b`:
# - Groundedness
# - Relevance
# - Hallucination
# - Completeness

# %%
# Input: generation examples with contexts.
# Output: RAG and judge metric summary.
# Logic: run context-aware and judge-based scoring.
# Complexity: O(num_generation_examples * judge_calls).
rag_metrics = compute_rag_metrics(generation_examples)
pd.DataFrame([rag_metrics])

# %% [markdown]
# ## Step 5: Unified Evaluation Bundle
#
# We package all metric families into one structured payload for reporting and regression tracking.

# %%
# Input: retrieval and generation examples.
# Output: unified evaluation bundle dictionary.
# Logic: compute and consolidate metrics via shared evaluator entrypoint.
# Complexity: O(total_evaluation_calls).
bundle = build_evaluation_bundle(
    retrieval_examples,
    generation_examples,
    k_values=[1, 3, 5, 8],
    include_bertscore=True,
    metadata={
        "dataset": "bigbio/medmentions",
        "embedding_model": settings.embedding_model,
        "generator_model": settings.generator_model,
        "judge_model": settings.judge_model,
        "historical_outputs_kept": True,
        "execution_phase_required_for_validation": True,
    },
)

bundle_payload = bundle.to_dict()
bundle_payload.keys()

# %% [markdown]
# ## Step 6: Persist Tables and Figures

# %%
# Input: metric dictionaries and bundle payload.
# Output: metric tables, charts, and JSON artifact files.
# Logic: flatten metrics by category, save CSV/JSON, and visualize key metrics.
# Complexity: O(number_of_metrics).
summary_rows = []
for metric, value in retrieval_metrics.items():
    summary_rows.append({"category": "retrieval", "metric": metric, "value": value})
for metric, value in generation_metrics.items():
    summary_rows.append({"category": "generation", "metric": metric, "value": value})
for metric, value in rag_metrics.items():
    summary_rows.append({"category": "rag", "metric": metric, "value": value})

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(settings.tables_dir / "nb05_metric_summary.csv", index=False)

plt.figure(figsize=(12, 5))
retrieval_df = summary_df[summary_df["category"] == "retrieval"]
plt.bar(retrieval_df["metric"], retrieval_df["value"], color="#1f77b4")
plt.xticks(rotation=45, ha="right")
plt.title("NB05 Retrieval Metrics")
plt.ylabel("Score")
save_plot(settings.figures_dir / "nb05_retrieval_metrics.png")
plt.show()

plt.figure(figsize=(14, 6))
other_df = summary_df[summary_df["category"].isin(["generation", "rag"])]
plt.bar(other_df["metric"], other_df["value"], color="#2ca02c")
plt.xticks(rotation=45, ha="right")
plt.title("NB05 Generation + RAG Metrics")
plt.ylabel("Score")
save_plot(settings.figures_dir / "nb05_generation_rag_metrics.png")
plt.show()

save_json(bundle_payload, settings.metrics_dir / "nb05_evaluation_bundle.json")

# Save sample run states for qualitative inspection.
sample_rows = []
for state in run_states[:10]:
    sample_rows.append(
        {
            "query": state.get("query", ""),
            "route": state.get("route", ""),
            "retrieval_score": float(state.get("retrieval_score", 0.0)),
            "hallucination_score": float(state.get("hallucination_score", 0.0)),
            "trace": " -> ".join(state.get("trace", [])),
            "answer_preview": state.get("final_answer", "")[:320],
        }
    )

pd.DataFrame(sample_rows).to_csv(settings.tables_dir / "nb05_sample_agent_outputs.csv", index=False)

print("Saved NB05 evaluation tables, figures, and bundle artifacts.")

# %% [markdown]
# ## Recap
#
# This notebook provides complete evaluation coverage for:
# - Retrieval metrics
# - Generation metrics
# - RAG grounding metrics
# - LLM judge metrics
#
# Historical outputs are preserved; any newly added outputs should be validated in the explicit run-and-validate phase.
