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
# # NB01: MedMentions Data Foundation (Zero-to-Hero)
#
# ## What
# We build the data foundation for a biomedical GraphRAG system using the real Hugging Face dataset `bigbio/medmentions`.
#
# ## Why
# GraphRAG quality is capped by data quality. If the ingest schema is noisy, every downstream stage (chunking, graph construction, retrieval, and generation) degrades.
#
# ## When
# Use this workflow when you need a biomedical corpus with real entity annotations (UMLS concept IDs) and want to avoid synthetic data.
#
# ## Tradeoffs
# - **All official splits vs train-only**: all splits maximize corpus coverage for GraphRAG, but reduce strict train/test separation for model training tasks.
# - **Keep raw schema vs normalize schema**: raw schema is faster to start, normalized schema is safer for reproducible pipelines.
#
# ## Alternatives
# - **NCBI Disease**: great quality, but narrow entity scope.
# - **BC5CDR**: useful for disease/chemical, but not broad enough for graph-based biomedical exploration.
# - **PubTator exports**: broad, but requires additional parsing and quality checks.
#
# ## Production Considerations
# - Persist normalized records with versioning.
# - Track exact data slice and random seed.
# - Keep schema stable so downstream jobs do not silently break.

# %%
# Input: notebook runtime and project source directory.
# Output: deterministic imports and plotting defaults.
# Logic: set up environment once at notebook start.
# Complexity: O(1).
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sys.path.append(str(Path.cwd().parent))

from src.config import settings
from src.data_pipeline import load_medmentions_records, persist_records
from src.utils import save_plot, timed_block

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

print(f"Project root: {settings.project_root}")
print(f"Processed directory: {settings.processed_dir}")

# %% [markdown]
# ## Step 1: Load Real MedMentions Records
#
# ### Definition
# A **record** is one PubMed document (PMID) with title, abstract, and manually curated entity mentions mapped to UMLS concept IDs.
#
# ### Why this step matters
# Every later artifact should be traceable to real source records. We do not use any synthetic augmentation in this project.

# %%
# Input: max_records configuration (5000 cap, real records only).
# Output: normalized record list with title, abstract, text, and entities.
# Logic: load official train/validation/test splits and normalize fields.
# Complexity: O(N) over records.
with timed_block("Load MedMentions"):
    records = load_medmentions_records(max_records=settings.max_records)

print(f"Total normalized records: {len(records):,}")

# %% [markdown]
# ## Step 2: Inspect Schema and Example Record
#
# ### What
# We inspect the normalized structure to make sure we know exactly what downstream code will consume.
#
# ### Why
# Debugging late-stage graph or retrieval failures is expensive. Early schema checks are cheap.

# %%
# Input: first normalized record.
# Output: readable schema preview.
# Logic: show key fields and example entity annotations.
# Complexity: O(1).
example = records[0]
print("PMID:", example.pmid)
print("Split:", example.split)
print("Title preview:", example.title[:140])
print("Abstract preview:", example.abstract[:300])
print("Entity count:", len(example.entities))
print("First 5 entities:")
for entity in example.entities[:5]:
    print(
        f"  text={entity.text!r} | concept_id={entity.concept_id} | semantic_types={entity.semantic_type_ids[:3]}"
    )

# %% [markdown]
# ## Step 3: Corpus-Level Diagnostics
#
# ### What
# We compute distribution statistics required for chunking and graph planning.
#
# ### Why
# - Text length distribution informs chunk size.
# - Entity density informs graph sparsity.
# - Split ratios explain where records come from.

# %%
# Input: normalized records list.
# Output: pandas stats table.
# Logic: aggregate split counts, char lengths, and entity counts.
# Complexity: O(N).
rows = []
for record in records:
    rows.append(
        {
            "pmid": record.pmid,
            "split": record.split,
            "text_chars": len(record.text),
            "title_chars": len(record.title),
            "abstract_chars": len(record.abstract),
            "entity_count": len(record.entities),
        }
    )

df = pd.DataFrame(rows)
summary = (
    df.groupby("split")
    .agg(
        records=("pmid", "count"),
        avg_text_chars=("text_chars", "mean"),
        median_text_chars=("text_chars", "median"),
        avg_entity_count=("entity_count", "mean"),
    )
    .reset_index()
)
summary

# %% [markdown]
# ## Step 4: Visualize Length and Entity Distributions
#
# ### Why these charts
# - Length histogram prevents accidentally choosing chunk sizes that over-fragment text.
# - Entity histogram estimates graph node/edge density and long-tail behavior.

# %%
# Input: dataframe with text and entity counts.
# Output: saved figures for README and later notebooks.
# Logic: build histograms and split-level bar chart.
# Complexity: O(N).
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(df["text_chars"], bins=40, color="#1f77b4", alpha=0.8)
axes[0].set_title("Document Length Distribution (characters)")
axes[0].set_xlabel("Characters")
axes[0].set_ylabel("Count")

axes[1].hist(df["entity_count"], bins=30, color="#2ca02c", alpha=0.8)
axes[1].set_title("Entity Count per Document")
axes[1].set_xlabel("Entities")
axes[1].set_ylabel("Count")

save_plot(settings.figures_dir / "nb01_length_entity_distributions.png")
plt.show()

split_counts = df["split"].value_counts().sort_index()
ax = split_counts.plot(kind="bar", color="#ff7f0e", title="Records per Official Split")
ax.set_xlabel("Split")
ax.set_ylabel("Records")
save_plot(settings.figures_dir / "nb01_split_counts.png")
plt.show()

# %% [markdown]
# ## Step 5: UMLS Concept and Semantic-Type Coverage
#
# ### Definition
# - **Concept ID (CUI)**: canonical UMLS identifier, for example `C0011849` for diabetes mellitus.
# - **Semantic Type ID (TUI)**: high-level biomedical category.
#
# ### Why
# GraphRAG can exploit concept IDs directly for entity-centric traversal.

# %%
# Input: entity annotations from all records.
# Output: top concept and semantic-type tables.
# Logic: flatten nested entities and count frequencies.
# Complexity: O(total_entities).
concept_counter = Counter()
semantic_counter = Counter()

for record in records:
    for entity in record.entities:
        concept_counter[entity.concept_id] += 1
        for semantic_id in entity.semantic_type_ids:
            semantic_counter[semantic_id] += 1

concept_df = pd.DataFrame(concept_counter.most_common(20), columns=["concept_id", "mentions"])
semantic_df = pd.DataFrame(semantic_counter.most_common(20), columns=["semantic_type_id", "mentions"])

print(f"Unique concept IDs: {len(concept_counter):,}")
print(f"Unique semantic type IDs: {len(semantic_counter):,}")
concept_df.head(10)

# %%
# Input: semantic type frequency table.
# Output: saved semantic-type chart.
# Logic: horizontal bar plot for readability.
# Complexity: O(K) for top-k semantic types.
ax = semantic_df.head(15).sort_values("mentions").plot(
    kind="barh",
    x="semantic_type_id",
    y="mentions",
    color="#9467bd",
    legend=False,
    title="Top Semantic Types in Selected MedMentions Slice",
)
ax.set_xlabel("Mentions")
ax.set_ylabel("Semantic Type ID")
save_plot(settings.figures_dir / "nb01_top_semantic_types.png")
plt.show()

# %% [markdown]
# ## Step 6: Persist Normalized Records for Downstream Notebooks
#
# ### What
# Save normalized records once so NB02+ never need to re-parse raw dataset structure.
#
# ### Why
# This removes repeated boilerplate, reduces error surface, and keeps notebook execution deterministic.

# %%
# Input: in-memory normalized records.
# Output: `data/processed/medmentions_records.json`.
# Logic: serialize dataclass objects to JSON.
# Complexity: O(N).
output_path = persist_records(records)
print("Saved normalized records:", output_path)

# Persist summary tables for README usage.
summary.to_csv(settings.tables_dir / "nb01_split_summary.csv", index=False)
concept_df.to_csv(settings.tables_dir / "nb01_top_concepts.csv", index=False)
semantic_df.to_csv(settings.tables_dir / "nb01_top_semantic_types.csv", index=False)
print("Saved NB01 tables to:", settings.tables_dir)

# %% [markdown]
# ## Notebook Recap
#
# You now have:
# 1. Real MedMentions records across all official splits.
# 2. Stable normalized schema ready for chunking.
# 3. Distribution diagnostics and saved figures.
# 4. Persisted JSON + CSV artifacts for reproducible downstream stages.
#
# Next: NB02 builds chunking + embeddings with `qwen3-embedding:4b` and persists vector-ready data.
