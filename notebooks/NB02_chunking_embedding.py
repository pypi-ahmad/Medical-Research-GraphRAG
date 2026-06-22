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
# # NB02: Chunking + Embeddings (From Documents to Retrieval Units)
#
# ## What
# Convert normalized MedMentions records into chunk-level retrieval units and embed them using `qwen3-embedding:4b` through Ollama.
#
# ## Why
# RAG does retrieval at chunk granularity, not full-document granularity. Good chunking and embedding quality are foundational for both vector search and GraphRAG.
#
# ## When
# Use this notebook whenever raw corpus updates or chunking settings change.
#
# ## Tradeoffs and Alternatives
# - **Recursive splitting (chosen)**: simple and robust; may still split nuanced context.
# - **Sentence-only splitting**: cleaner language boundaries, but weaker control over chunk size.
# - **Semantic chunking with LLM**: best coherence, much higher latency/cost.
#
# ## Model Choice
# - **Chosen**: `qwen3-embedding:4b` via Ollama.
# - **Why not smaller local models**: faster but lower semantic fidelity on biomedical terminology.
# - **Why not hosted APIs**: extra recurring cost and data-governance concerns.

# %%
# Input: notebook runtime and persisted records from NB01.
# Output: imports, deterministic plotting style, and path setup.
# Logic: one-time notebook initialization.
# Complexity: O(1).
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

sys.path.append(str(Path.cwd().parent))

from src.chunking import build_chunk_lookup, build_chunks, chunk_statistics, persist_chunks
from src.config import settings
from src.data_pipeline import load_persisted_records
from src.embeddings import EmbeddingBundle, embed_texts, persist_embedding_bundle
from src.utils import save_json, save_plot, timed_block

plt.rcParams["figure.figsize"] = (10, 6)

print(f"Embedding model: {settings.embedding_model}")
print(f"Chunk config: size={settings.chunk_size}, overlap={settings.chunk_overlap}")

# %% [markdown]
# ## Step 1: Load Normalized Records
#
# ### Why
# We operate on the normalized schema created in NB01 to avoid re-parsing raw dataset fields.

# %%
# Input: `data/processed/medmentions_records.json`.
# Output: list of typed `MedRecord` objects.
# Logic: deserialize persisted normalized records.
# Complexity: O(N).
records = load_persisted_records()
print(f"Loaded records: {len(records):,}")

# %% [markdown]
# ## Step 2: Build Entity-Aware Chunks
#
# ### Definitions
# - **Chunk**: retrieval unit sent to vector stores.
# - **Chunk overlap**: overlap between adjacent chunks to avoid boundary information loss.
#
# ### Production note
# Monitor chunk explosion. Smaller chunk sizes increase index size and latency.

# %%
# Input: normalized records + chunk settings.
# Output: list of `ChunkRecord` objects.
# Logic: recursive splitting + entity mention carryover.
# Complexity: O(total_characters + total_entities).
with timed_block("Build chunks"):
    chunks = build_chunks(
        records,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

print(f"Total chunks: {len(chunks):,}")

stats = chunk_statistics(chunks)
pd.DataFrame([stats])

# %% [markdown]
# ## Step 3: Visualize Chunk Distributions
#
# ### Why
# These plots verify that chunking settings are practical before expensive embedding and indexing.

# %%
# Input: chunk records.
# Output: saved distribution figure.
# Logic: histogram for chunk lengths and entity counts.
# Complexity: O(number_of_chunks).
chunk_lengths = np.array([len(chunk.text) for chunk in chunks])
chunk_entities = np.array([chunk.entity_count for chunk in chunks])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(chunk_lengths, bins=50, color="#1f77b4", alpha=0.85)
axes[0].set_title("Chunk Length Distribution")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")
axes[0].axvline(settings.chunk_size, color="red", linestyle="--", label="Target size")
axes[0].legend()

axes[1].hist(chunk_entities, bins=40, color="#2ca02c", alpha=0.85)
axes[1].set_title("Entities per Chunk")
axes[1].set_xlabel("Entity count")
axes[1].set_ylabel("Count")

save_plot(settings.figures_dir / "nb02_chunk_distributions.png")
plt.show()

# %% [markdown]
# ## Step 4: Embed Chunks with `qwen3-embedding:4b`
#
# ### What
# We embed chunk text into dense vectors for semantic retrieval.
#
# ### Why
# Dense embeddings capture semantic similarity beyond exact keyword overlap.
#
# ### Complexity
# Approximately `O(num_chunks * embedding_latency)` where embedding latency depends on local hardware.

# %%
# Input: chunk texts.
# Output: normalized embedding matrix.
# Logic: batch embedding through Ollama.
# Complexity: O(number_of_chunks).
chunk_texts = [chunk.text for chunk in chunks]

with timed_block("Embed chunks with Ollama"):
    matrix = embed_texts(chunk_texts, model=settings.embedding_model, batch_size=64, normalize=True)

print("Embedding matrix shape:", matrix.shape)
print("Embedding dimension:", matrix.shape[1] if matrix.size else 0)

# %% [markdown]
# ## Step 5: Embedding Sanity Check (PCA)
#
# ### Why
# A collapsed embedding space is a critical failure mode. PCA helps detect it quickly.

# %%
# Input: embedding matrix.
# Output: PCA projection plot.
# Logic: random sample then 2D PCA projection.
# Complexity: O(sample_size * embedding_dim^2) for PCA.
rng = np.random.default_rng(settings.random_seed)
sample_size = min(2000, matrix.shape[0])
indices = rng.choice(matrix.shape[0], size=sample_size, replace=False)
sampled = matrix[indices]

pca = PCA(n_components=2, random_state=settings.random_seed)
coords = pca.fit_transform(sampled)

plt.figure(figsize=(9, 7))
plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.4, c="#9467bd")
plt.title("PCA of Chunk Embeddings")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
save_plot(settings.figures_dir / "nb02_embedding_pca.png")
plt.show()

# %% [markdown]
# ## Step 6: Persist Chunk and Embedding Artifacts
#
# ### What gets saved
# - Chunk parquet table
# - Embedding matrix (`.npy`)
# - Aligned chunk ID mapping
# - PMID -> chunk IDs lookup for evaluation and graph linking

# %%
# Input: chunks + embedding matrix.
# Output: persistent artifacts in `data/processed`.
# Logic: save once, reuse across all later notebooks.
# Complexity: O(number_of_chunks).
chunk_path = persist_chunks(chunks)
bundle = EmbeddingBundle(chunk_ids=[chunk.chunk_id for chunk in chunks], matrix=matrix)
embedding_paths = persist_embedding_bundle(bundle)

chunk_lookup = build_chunk_lookup(chunks)
save_json(chunk_lookup, settings.processed_dir / "pmid_to_chunk_ids.json")

print("Saved chunk parquet:", chunk_path)
print("Saved embedding files:", embedding_paths)
print("Saved chunk lookup JSON")

# Persist summary table.
pd.DataFrame([stats]).to_csv(settings.tables_dir / "nb02_chunk_stats.csv", index=False)

# %% [markdown]
# ## Notebook Recap
#
# You now have:
# 1. Entity-aware chunk records.
# 2. Dense embeddings from `qwen3-embedding:4b`.
# 3. Persisted vector-ready artifacts for ChromaDB and Pinecone.
#
# Next: NB03 builds the NetworkX biomedical knowledge graph and community structure for GraphRAG local/global search.
