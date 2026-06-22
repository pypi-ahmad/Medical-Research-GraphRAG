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
# # NB05: Section C Agentic GraphRAG + Full Evaluation
#
# ## What
# Build a LangGraph agentic GraphRAG workflow and run end-to-end evaluation.
#
# ## Required Workflow (implemented)
# Query -> Retrieval -> Retrieval Grader -> Graph Traversal -> Context Expansion -> Answer Generation -> Hallucination Detection -> Final Response.
#
# ## Why Agentic GraphRAG
# Fixed pipelines are brittle. Agentic flow adds dynamic control for retrieval quality and hallucination mitigation.
#
# ## LangGraph Choice
# - **Chosen**: LangGraph for explicit state-machine orchestration.
# - **Why not plain LangChain chains**: weaker conditional/branch control.
# - **Why not custom imperative loop only**: less transparent graph structure and tracing.

# %%
# Input: persisted retrieval + graph artifacts and agent/evaluator modules.
# Output: initialized runtime for agentic execution.
# Logic: load all resources once.
# Complexity: O(number_of_chunks + graph size).
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd().parent))

from src.agentic_rag import AgentResources, build_agentic_workflow, run_agentic_query, workflow_mermaid
from src.chroma_retriever import entity_search, get_collection, reciprocal_rank_fusion, vector_search
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
from src.graph_builder import concept_id_from_query, local_graph_expansion
from src.utils import save_json, save_plot, timed_block

records = load_persisted_records()
chunks = load_chunks()
chunk_lookup = build_chunk_lookup(chunks)

with (settings.graph_dir / "entity_graph.pkl").open("rb") as f:
    graph = pickle.load(f)
partition = json.loads((settings.graph_dir / "community_partition.json").read_text(encoding="utf-8"))
summaries = json.loads((settings.graph_dir / "community_summaries.json").read_text(encoding="utf-8"))

chroma_collection = get_collection("medmentions_chroma")

print(f"Records: {len(records):,}")
print(f"Chunks: {len(chunks):,}")
print(f"Graph nodes/edges: {graph.number_of_nodes():,}/{graph.number_of_edges():,}")

# %% [markdown]
# ## Step 1: Build Agentic LangGraph Workflow
#
# ### Node-by-node explanation
# - **Retrieval**: initial vector retrieval.
# - **Retrieval Grader**: LLM scores retrieval quality.
# - **Graph Traversal**: local graph expansion around query entities.
# - **Context Expansion**: merge vector, graph, and optional web evidence.
# - **Answer Generation**: produce grounded biomedical answer.
# - **Hallucination Detection**: LLM grounding audit with one retry.
# - **Final Response**: return finalized answer.

# %%
# Input: resources bundle.
# Output: compiled LangGraph app.
# Logic: instantiate graph workflow and compile once.
# Complexity: O(1) for graph construction metadata.
resources = AgentResources(
    chroma_collection=chroma_collection,
    chunks=chunks,
    graph=graph,
    partition=partition,
    summaries=summaries,
)

app = build_agentic_workflow(resources)
print("LangGraph workflow compiled.")
print("\nMermaid graph:\n")
print(workflow_mermaid())

# %% [markdown]
# ## Step 2: Qualitative Demo Runs (Real Outputs)
#
# We run representative biomedical queries and inspect route, trace, and final answer quality.

# %%
# Input: manual demo queries.
# Output: full agent states with traces and answers.
# Logic: invoke workflow per query.
# Complexity: O(number_of_queries * pipeline_cost).
demo_queries = [
    "What does this corpus report about diabetes and insulin resistance?",
    "What evidence in these abstracts discusses hypertension risk?",
    "How is pancreatic cancer described in relation to KRAS?",
]

demo_outputs = []
for query in demo_queries:
    with timed_block(f"Agent run: {query[:60]}"):
        state = run_agentic_query(app, query)
    demo_outputs.append(state)

for idx, state in enumerate(demo_outputs, start=1):
    print(f"\n==== Demo {idx} ====")
    print("Query:", state["query"])
    print("Route:", state.get("route", ""))
    print("Retrieval score:", state.get("retrieval_score", 0.0))
    print("Hallucination score:", state.get("hallucination_score", 0.0))
    print("Trace:", " -> ".join(state.get("trace", [])))
    print("Answer preview:", state.get("final_answer", "")[:450], "...")

# %% [markdown]
# ## Step 3: Build Evaluation Set
#
# We reuse extractive real references generated from MedMentions text; no synthetic references are used.

# %%
# Input: records + chunk lookup.
# Output: evaluation query objects.
# Logic: deterministic extractive query creation.
# Complexity: O(records * entities).
eval_queries = build_extractive_eval_queries(
    records=records,
    chunk_lookup=chunk_lookup,
    sample_size=settings.eval_query_count,
)

print(f"Evaluation query count: {len(eval_queries)}")

# %% [markdown]
# ## Step 4: Retrieval-Only Evaluation Pass (Fast, Deterministic)
#
# We compute retrieval metrics with the same GraphRAG retrieval logic but without generation calls.
# This keeps retrieval benchmarking fast and isolates retriever quality from generation latency.

# %%
# Input: evaluation query list.
# Output: retrieval payloads for ranking metrics.
# Logic: vector retrieval + graph-expanded entity channel + RRF fusion.
# Complexity: O(num_queries * retrieval_cost).
retrieval_payloads = []


def retrieval_only_graphrag(query: str, top_k: int = 8):
    vec = vector_search(chroma_collection, query, top_k=top_k * 2)
    concept_ids = concept_id_from_query(query, chunks)
    neighborhood = local_graph_expansion(graph, concept_ids, hops=settings.local_graph_hops)
    entity = entity_search(chroma_collection, concept_ids=neighborhood.get("nodes", [])[:100], top_k=top_k * 2)
    return reciprocal_rank_fusion({"vector": vec, "entity": entity}, top_k=top_k)


with timed_block("Run retrieval-only GraphRAG evaluation"):
    for item in eval_queries:
        results = retrieval_only_graphrag(item.query, top_k=settings.top_k_retrieval)
        retrieval_payloads.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "reference_answer": item.reference_answer,
                "supporting_chunk_ids": item.supporting_chunk_ids,
                "retrieved_ids": [row["id"] for row in results],
            }
        )

print("Collected retrieval payloads:", len(retrieval_payloads))

# %% [markdown]
# ## Step 5: Retrieval Metrics (Precision@K, Recall@K, F1, MRR, NDCG)

# %%
# Input: retrieval payloads with retrieved IDs and relevant IDs.
# Output: retrieval metrics dictionary.
# Logic: convert payloads into `RetrievalExample` objects.
# Complexity: O(num_queries * top_k).
retrieval_examples = [
    RetrievalExample(
        retrieved_ids=item["retrieved_ids"],
        relevant_ids=item["supporting_chunk_ids"],
    )
    for item in retrieval_payloads
]

retrieval_metrics = compute_retrieval_metrics(retrieval_examples, k_values=[1, 3, 5, 8])
retrieval_df = pd.DataFrame([retrieval_metrics])
retrieval_df

# %% [markdown]
# ## Step 6: Agentic Generation Pass + Generation Metrics (EM, BLEU, ROUGE, METEOR, BERTScore)
#
# We evaluate generation on a practical subset for runtime control while preserving real data references.

# %%
# Input: evaluation query subset.
# Output: generation metrics dictionary.
# Logic: run full agentic pipeline on subset, then evaluate against extractive references.
# Complexity: O(num_generation_examples * metric_cost).
generation_eval_count = min(settings.generation_eval_count, 15)
generation_subset = eval_queries[:generation_eval_count]

run_payloads = []
with timed_block("Run full agentic pipeline on generation subset"):
    for item in generation_subset:
        state = run_agentic_query(app, item.query)
        run_payloads.append(
            {
                "query_id": item.query_id,
                "query": item.query,
                "reference_answer": item.reference_answer,
                "supporting_chunk_ids": item.supporting_chunk_ids,
                "retrieved_ids": [doc["id"] for doc in state.get("retrieved_docs", [])],
                "answer": state.get("final_answer", ""),
                "context_chunks": [doc["text"] for doc in state.get("retrieved_docs", [])[:8]],
                "retrieval_score": state.get("retrieval_score", 0.0),
                "hallucination_score": state.get("hallucination_score", 0.0),
                "trace": state.get("trace", []),
                "route": state.get("route", ""),
            }
        )

print("Collected generation payloads:", len(run_payloads))

generation_examples = [
    GenerationExample(
        query=item["query"],
        answer=item["answer"],
        reference_answer=item["reference_answer"],
        context_chunks=item["context_chunks"],
    )
    for item in run_payloads
]

generation_metrics = compute_generation_metrics(generation_examples, include_bertscore=True)
generation_df = pd.DataFrame([generation_metrics])
generation_df

# %% [markdown]
# ## Step 7: RAG Metrics + LLM Judge Metrics
#
# Metrics:
# - Faithfulness
# - Context Precision
# - Context Recall
# - Answer Relevancy
# - Judge Groundedness
# - Judge Relevance
# - Judge Hallucination
# - Judge Completeness

# %%
# Input: generation examples.
# Output: rag/judge metrics dictionary.
# Logic: judge-driven evaluation over grounded examples.
# Complexity: O(num_generation_examples * judge_calls).
rag_metrics = compute_rag_metrics(generation_examples)
rag_df = pd.DataFrame([rag_metrics])
rag_df

# %% [markdown]
# ## Step 8: Consolidated Results Tables and Charts

# %%
# Input: metric dictionaries.
# Output: persisted CSV/JSON and visualization charts.
# Logic: combine into one summary frame and plot key indicators.
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

# Plot retrieval metrics.
retrieval_plot = summary_df[summary_df["category"] == "retrieval"]
plt.figure(figsize=(12, 5))
plt.bar(retrieval_plot["metric"], retrieval_plot["value"], color="#1f77b4")
plt.xticks(rotation=45, ha="right")
plt.title("NB05 Retrieval Metrics")
plt.ylabel("Score")
save_plot(settings.figures_dir / "nb05_retrieval_metrics.png")
plt.show()

# Plot generation + rag metrics.
other_plot = summary_df[summary_df["category"].isin(["generation", "rag"])]
plt.figure(figsize=(14, 6))
plt.bar(other_plot["metric"], other_plot["value"], color="#2ca02c")
plt.xticks(rotation=45, ha="right")
plt.title("NB05 Generation + RAG Metrics")
plt.ylabel("Score")
save_plot(settings.figures_dir / "nb05_generation_rag_metrics.png")
plt.show()

# %% [markdown]
# ## Step 9: Save Full Evaluation Artifact

# %%
# Input: metrics + sample outputs.
# Output: final JSON artifact used by README.
# Logic: persist all summary metrics and selected examples.
# Complexity: O(num_queries).
final_payload = {
    "retrieval_metrics": retrieval_metrics,
    "generation_metrics": generation_metrics,
    "rag_metrics": rag_metrics,
    "num_eval_queries": len(eval_queries),
    "num_generation_eval_queries": len(generation_examples),
    "sample_outputs": run_payloads[:8],
    "agent_workflow_mermaid": workflow_mermaid(),
}

save_json(final_payload, settings.metrics_dir / "nb05_final_evaluation.json")
print("Saved final evaluation artifact.")

# %% [markdown]
# ## Notebook Recap
#
# This notebook delivered Section C and end-to-end validation metrics:
# 1. LangGraph agentic GraphRAG with required node flow.
# 2. Web-search fallback path for poor retrieval quality.
# 3. Complete retrieval/generation/RAG/judge metric computation.
# 4. Persisted outputs for README reporting.
#
# You now have an end-to-end Medical Research Assistant pipeline built on real MedMentions data.
