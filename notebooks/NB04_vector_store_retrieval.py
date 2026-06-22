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
# # NB04: Section A + Section B Retrieval Systems
#
# ## Section A (ChromaDB GraphRAG)
# We build local persistent GraphRAG retrieval on ChromaDB.
#
# ## Section B (Pinecone GraphRAG)
# We mirror the workflow in Pinecone and compare against ChromaDB.
#
# ## Why compare these two
# - **ChromaDB** is local, low-ops, and cost-effective.
# - **Pinecone** is managed and scalable for production workloads.
#
# ## Why ChromaDB (chosen for local baseline)
# - Simple local persistence.
# - Zero managed-service billing.
# - Easy notebook iteration.
#
# ## Why not FAISS for this section
# - FAISS is excellent for local ANN speed, but does not provide a persistent metadata-rich DB interface by default.
#
# ## Why not Weaviate/Qdrant for this section
# - Both are strong alternatives, but add service/runtime complexity not necessary for this educational baseline.
#
# ## Production Considerations
# - Keep identical chunk schema across vector stores.
# - Benchmark quality and latency together, not in isolation.
# - Include cost and operational complexity in decision making.

# %%
# Input: persisted chunk, embedding, and graph artifacts.
# Output: initialized clients and loaded assets.
# Logic: hydrate all reusable artifacts and helper imports.
# Complexity: O(number_of_chunks + V + E) for loading.
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

from src.chroma_retriever import (
    entity_search,
    get_collection,
    index_chunks_to_chromadb,
    reciprocal_rank_fusion,
    vector_search,
)
from src.chunking import build_chunk_lookup, load_chunks
from src.config import settings
from src.data_pipeline import build_extractive_eval_queries, load_persisted_records
from src.embeddings import load_embedding_bundle
from src.evaluator import RetrievalExample, compute_retrieval_metrics
from src.graph_builder import concept_id_from_query, local_graph_expansion
from src.pinecone_retriever import (
    delete_index,
    index_chunks_to_pinecone,
    pinecone_cost_proxy,
    query_pinecone,
)
from src.utils import save_plot, timed_block

print("Using embedding model:", settings.embedding_model)
print("Retrieval top-k:", settings.top_k_retrieval)

# %% [markdown]
# ## Step 1: Load All Persisted Artifacts

# %%
# Input: persisted files from NB01-NB03.
# Output: in-memory artifacts for retrieval benchmarking.
# Logic: load records, chunks, embeddings, and graph.
# Complexity: O(N + chunks + vectors).
records = load_persisted_records()
chunks = load_chunks()
chunk_lookup = build_chunk_lookup(chunks)

embedding_bundle = load_embedding_bundle()
embeddings = embedding_bundle.matrix

with (settings.graph_dir / "entity_graph.pkl").open("rb") as f:
    graph = pickle.load(f)
partition = json.loads((settings.graph_dir / "community_partition.json").read_text(encoding="utf-8"))
summaries = json.loads((settings.graph_dir / "community_summaries.json").read_text(encoding="utf-8"))

print(f"Records: {len(records):,}")
print(f"Chunks: {len(chunks):,}")
print(f"Embedding shape: {embeddings.shape}")
print(f"Graph nodes/edges: {graph.number_of_nodes():,}/{graph.number_of_edges():,}")

# %% [markdown]
# ## Step 2: Build Section A ChromaDB GraphRAG Index
#
# ### Workflow
# 1. Index chunk embeddings.
# 2. Retrieve by vector similarity.
# 3. Expand with graph-derived entity neighborhood.
# 4. Fuse vector + entity channels with RRF.

# %%
# Input: chunks + embeddings.
# Output: persistent Chroma collection.
# Logic: create/replace collection and upsert chunk data.
# Complexity: O(number_of_chunks).
with timed_block("Index ChromaDB"):
    chroma_collection = index_chunks_to_chromadb(
        chunks=chunks,
        embeddings=embeddings,
        collection_name="medmentions_chroma",
        batch_size=128,
    )

print("Chroma indexing complete.")

# %% [markdown]
# ## Step 3: Build Section B Pinecone GraphRAG Index
#
# ### Workflow parity
# Same chunk embeddings and metadata schema are used to keep the comparison fair.

# %%
# Input: chunks + embeddings + Pinecone credentials.
# Output: Pinecone index for benchmark.
# Logic: create index and upsert vectors.
# Complexity: O(number_of_chunks).
pinecone_index_name = f"{settings.pinecone_index_prefix}-nb04"

with timed_block("Index Pinecone"):
    index_chunks_to_pinecone(
        chunks=chunks,
        embeddings=embeddings,
        index_name=pinecone_index_name,
        namespace="nb04",
        batch_size=100,
    )

print("Pinecone indexing complete.")

# %% [markdown]
# ## Step 4: Build Evaluation Query Set (Real Extractive References)
#
# We reuse real-record extractive query definitions so retrieval quality metrics are grounded in actual evidence.

# %%
# Input: normalized records + PMID->chunk lookup.
# Output: retrieval evaluation query list.
# Logic: extractive query generation with deterministic seed.
# Complexity: O(records * entities).
eval_queries = build_extractive_eval_queries(
    records=records,
    chunk_lookup=chunk_lookup,
    sample_size=settings.eval_query_count,
)

print(f"Evaluation queries: {len(eval_queries)}")

# %% [markdown]
# ## Step 5: Define Matched GraphRAG Retrieval Functions
#
# ### Chroma GraphRAG
# Vector + entity filtering channel fused with RRF.
#
# ### Pinecone GraphRAG
# Vector search + entity-aware reranking channel fused with RRF.

# %%
# Input: single user query.
# Output: ranked retrieval list with IDs, text, scores.
# Logic: matched multi-channel retrieval per backend.
# Complexity: roughly O(logN) ANN + rerank overhead.
def chroma_graphrag_search(query: str, top_k: int = 8):
    vector_results = vector_search(chroma_collection, query, top_k=top_k * 2)

    concept_ids = concept_id_from_query(query, chunks)
    neighborhood = local_graph_expansion(graph, concept_ids, hops=settings.local_graph_hops)
    expanded_concepts = neighborhood.get("nodes", [])[:80]

    entity_results = entity_search(chroma_collection, concept_ids=expanded_concepts, top_k=top_k * 2)

    fused = reciprocal_rank_fusion(
        {"vector": vector_results, "entity": entity_results},
        top_k=top_k,
    )
    return fused


def pinecone_graphrag_search(query: str, top_k: int = 8):
    vector_results = query_pinecone(
        query=query,
        index_name=pinecone_index_name,
        namespace="nb04",
        top_k=top_k * 3,
    )

    concept_ids = concept_id_from_query(query, chunks)
    neighborhood = local_graph_expansion(graph, concept_ids, hops=settings.local_graph_hops)
    expanded_concepts = set(neighborhood.get("nodes", [])[:120])

    entity_results = []
    for item in vector_results:
        concepts_str = item.get("metadata", {}).get("concept_ids", "")
        match_count = sum(1 for concept in expanded_concepts if concept and concept in concepts_str)
        if match_count <= 0:
            continue
        entity_results.append(
            {
                "id": item["id"],
                "text": item["text"],
                "metadata": item["metadata"],
                "score": float(match_count),
                "source": "entity",
            }
        )

    entity_results.sort(key=lambda x: x["score"], reverse=True)

    fused = reciprocal_rank_fusion(
        {"vector": vector_results[: top_k * 2], "entity": entity_results[: top_k * 2]},
        top_k=top_k,
    )
    return fused

# %% [markdown]
# ## Step 6: Latency + Retrieval Quality Benchmark
#
# ### Metrics
# - Precision@K
# - Recall@K
# - F1@K
# - MRR
# - NDCG

# %%
# Input: eval query list and backend search functions.
# Output: latency traces + retrieval metric dictionaries.
# Logic: run per-query retrieval and aggregate.
# Complexity: O(num_queries * retrieval_cost).
def benchmark_backend(name: str, search_fn):
    retrieval_examples = []
    latency_ms = []

    for item in eval_queries:
        start = time.perf_counter()
        results = search_fn(item.query, top_k=settings.top_k_retrieval)
        elapsed_ms = (time.perf_counter() - start) * 1000
        latency_ms.append(elapsed_ms)

        retrieved_ids = [r["id"] for r in results]
        retrieval_examples.append(
            RetrievalExample(
                retrieved_ids=retrieved_ids,
                relevant_ids=item.supporting_chunk_ids,
            )
        )

    retrieval_metrics = compute_retrieval_metrics(retrieval_examples, k_values=[1, 3, 5, 8])

    return {
        "name": name,
        "latency_ms": latency_ms,
        "metrics": retrieval_metrics,
    }

with timed_block("Benchmark Chroma GraphRAG"):
    chroma_benchmark = benchmark_backend("ChromaDB", chroma_graphrag_search)

with timed_block("Benchmark Pinecone GraphRAG"):
    pinecone_benchmark = benchmark_backend("Pinecone", pinecone_graphrag_search)

# %% [markdown]
# ## Step 7: Comparison Tables (Latency, Quality, Cost, Complexity, Scalability)

# %%
# Input: benchmark outputs.
# Output: comparison dataframe and saved CSV.
# Logic: aggregate p50/p95 latency and retrieval metrics.
# Complexity: O(num_queries).
def latency_summary(latencies: list[float]) -> dict[str, float]:
    values = np.array(latencies, dtype=float)
    return {
        "latency_p50_ms": float(np.percentile(values, 50)),
        "latency_p95_ms": float(np.percentile(values, 95)),
        "latency_p99_ms": float(np.percentile(values, 99)),
    }

chroma_latency = latency_summary(chroma_benchmark["latency_ms"])
pinecone_latency = latency_summary(pinecone_benchmark["latency_ms"])

chroma_cost_proxy = {
    "query_count": len(eval_queries),
    "upsert_count": len(chunks),
    "total_vectors": len(chunks),
    "pricing_note": "Local deployment: direct managed-service cost is approximately zero; infra cost depends on host machine.",
}

pinecone_cost = pinecone_cost_proxy(
    index_name=pinecone_index_name,
    namespace="nb04",
    query_count=len(eval_queries),
    upsert_count=len(chunks),
)

comparison = pd.DataFrame(
    [
        {
            "backend": "ChromaDB",
            **chroma_latency,
            **{k: float(v) for k, v in chroma_benchmark["metrics"].items()},
            "cost_driver": f"queries={chroma_cost_proxy['query_count']}, vectors={chroma_cost_proxy['total_vectors']}",
            "complexity_score_1_to_5": 2,
            "scalability_score_1_to_5": 3,
        },
        {
            "backend": "Pinecone",
            **pinecone_latency,
            **{k: float(v) for k, v in pinecone_benchmark["metrics"].items()},
            "cost_driver": f"queries={pinecone_cost['query_count']}, vectors={pinecone_cost['total_vectors']}",
            "complexity_score_1_to_5": 3,
            "scalability_score_1_to_5": 5,
        },
    ]
)

comparison.to_csv(settings.tables_dir / "nb04_chroma_vs_pinecone.csv", index=False)
comparison

# %% [markdown]
# ## Step 8: Comparison Charts

# %%
# Input: comparison table.
# Output: latency and quality comparison plots.
# Logic: plot key metrics side by side.
# Complexity: O(number_of_backends * number_of_metrics).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

latency_plot = comparison[["backend", "latency_p50_ms", "latency_p95_ms"]].set_index("backend")
latency_plot.plot(kind="bar", ax=axes[0], color=["#1f77b4", "#ff7f0e"])
axes[0].set_title("Latency Comparison")
axes[0].set_ylabel("Milliseconds")
axes[0].set_xlabel("Backend")
axes[0].legend(title="Percentile")

quality_plot = comparison[["backend", "precision@5", "recall@5", "f1@5", "mrr"]].set_index("backend")
quality_plot.plot(kind="bar", ax=axes[1], color=["#2ca02c", "#d62728", "#9467bd", "#8c564b"])
axes[1].set_title("Retrieval Quality Comparison")
axes[1].set_ylabel("Score")
axes[1].set_xlabel("Backend")
axes[1].legend(title="Metric")

save_plot(settings.figures_dir / "nb04_chroma_pinecone_comparison.png")
plt.show()

# Plot latency distributions.
plt.figure(figsize=(10, 5))
plt.hist(chroma_benchmark["latency_ms"], bins=25, alpha=0.6, label="ChromaDB", color="#1f77b4")
plt.hist(pinecone_benchmark["latency_ms"], bins=25, alpha=0.6, label="Pinecone", color="#ff7f0e")
plt.title("Per-query Latency Distribution")
plt.xlabel("Latency (ms)")
plt.ylabel("Query count")
plt.legend()
save_plot(settings.figures_dir / "nb04_latency_distribution.png")
plt.show()

# %% [markdown]
# ## Step 9: Retrieval Examples (Real Outputs)
#
# We print real retrieval outputs for both systems on the same query.

# %%
# Input: sample evaluation query.
# Output: side-by-side retrieval examples.
# Logic: run both GraphRAG search functions and inspect top results.
# Complexity: O(one query retrieval).
sample_query = eval_queries[0].query
print("Sample query:", sample_query)

sample_chroma = chroma_graphrag_search(sample_query, top_k=4)
sample_pinecone = pinecone_graphrag_search(sample_query, top_k=4)

print("\nChromaDB top results:")
for i, row in enumerate(sample_chroma, start=1):
    print(f"[{i}] id={row['id']} score={row['score']:.4f} sources={row.get('sources', [])}")
    print(row["text"][:220], "...\n")

print("\nPinecone top results:")
for i, row in enumerate(sample_pinecone, start=1):
    print(f"[{i}] id={row['id']} score={row['score']:.4f} sources={row.get('sources', [])}")
    print(row["text"][:220], "...\n")

# %% [markdown]
# ## Step 10: Persist Benchmark Artifacts and Cleanup Pinecone Index
#
# We save benchmark JSON for README and remove the Pinecone index to avoid ongoing cost.

# %%
# Input: benchmark outputs.
# Output: metrics JSON and cleaned Pinecone state.
# Logic: serialize benchmarks and delete temporary cloud index.
# Complexity: O(number_of_queries) serialization + O(1) cleanup.
benchmark_payload = {
    "chroma": {
        "metrics": chroma_benchmark["metrics"],
        "latency_summary": chroma_latency,
        "cost_proxy": chroma_cost_proxy,
    },
    "pinecone": {
        "metrics": pinecone_benchmark["metrics"],
        "latency_summary": pinecone_latency,
        "cost_proxy": pinecone_cost,
    },
}

(settings.metrics_dir / "nb04_retrieval_benchmark.json").write_text(
    json.dumps(benchmark_payload, indent=2, ensure_ascii=True),
    encoding="utf-8",
)

# Pinecone cleanup (required by project plan).
delete_index(pinecone_index_name)
print("Saved benchmark payload and deleted temporary Pinecone index.")

# %% [markdown]
# ## Notebook Recap
#
# You now have:
# 1. **Section A**: ChromaDB GraphRAG retrieval with local/global graph context.
# 2. **Section B**: Pinecone GraphRAG retrieval with matched workflow.
# 3. Side-by-side latency, quality, cost-proxy, complexity, and scalability comparisons.
# 4. Saved real retrieval outputs, tables, and charts for README reporting.
#
# Next: NB05 builds the full LangGraph agentic pipeline and runs end-to-end evaluation (retrieval + generation + RAG + LLM judge metrics).
