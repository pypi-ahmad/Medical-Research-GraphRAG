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
# # NB11: Selective Fine-Tuning for Medical RAG (Unsloth + PEFT + TRL)
#
# ## What Is This Technique?
# Selective fine-tuning is an **optional** optimization stage that adapts a base instruction model with domain-grounded supervision.
# In this project, we apply it only to the answer generation stage and keep retrieval architecture unchanged.
#
# ## Definition and Core Concepts
# - **Unsloth**: a memory- and speed-oriented training runtime for efficient LoRA/QLoRA-style adaptation.
# - **PEFT**: Parameter-Efficient Fine-Tuning; updates a small adapter layer (LoRA) instead of all model weights.
# - **TRL**: Transformer Reinforcement Learning library; here we use `SFTTrainer` for supervised fine-tuning (not RLHF).
#
# ## Why Was This Technique Developed?
# Full-model fine-tuning is expensive, slow, and operationally heavy. These tools were developed to make adaptation practical on constrained hardware and faster experimentation cycles.
#
# ## What Limitations of Traditional RAG Does It Solve?
# Traditional RAG can still produce weak answer style/structure even with good retrieval. Selective fine-tuning helps:
# - improve answer format consistency,
# - improve instruction adherence to biomedical grounding policies,
# - reduce generic or under-specified phrasing.
#
# ## Tradeoffs
# - Adds training lifecycle overhead (data curation, checkpointing, regression evaluation).
# - Increases operational burden for adapter versioning and deployment governance.
# - Can improve generation quality while leaving retrieval quality unchanged.
#
# ## Alternatives
# - Stronger prompt engineering with no fine-tuning.
# - Retrieval improvements (Hybrid/CRAG/Graph expansion) before model adaptation.
# - Full-model fine-tuning (higher cost, broader parameter updates).
#
# ## Production Considerations
# - Track adapter lineage (dataset snapshot, hyperparameters, base model version).
# - Use strict baseline-vs-finetuned regression gates before rollout.
# - Keep rollback path to baseline generator always available.
#
# ## Why This Is Optional in This Project
# Base GraphRAG/Hybrid/CRAG/Agentic paths already work. Fine-tuning is added only as an **incremental quality lever** if baseline evaluation indicates quality gaps worth the extra complexity.

# %% [markdown]
# ## Official Best-Practice References Used
# The implementation choices in this notebook follow current official documentation patterns:
# - Unsloth: `FastLanguageModel.from_pretrained` + `get_peft_model` adapter workflow.
# - PEFT: LoRA/QLoRA adapter strategy and target-module guidance.
# - TRL: `SFTTrainer` / `SFTConfig` supervised fine-tuning patterns.
#
# Reference links:
# - https://github.com/unslothai/unsloth
# - https://huggingface.co/docs/peft/index
# - https://huggingface.co/docs/trl/index

# %% [markdown]
# ## Architecture Diagram
#
# ```mermaid
# flowchart TD
#     A[Real MedMentions Queries + References] --> B[SFT Dataset Builder]
#     B --> C[JSONL Train/Eval]
#     C --> D[Unsloth Base Model Load]
#     D --> E[PEFT LoRA Adapter Injection]
#     E --> F[TRL SFTTrainer]
#     F --> G[Adapter Artifacts]
#     G --> H[Ollama Modelfile Template]
#     H --> I[Optional Finetuned Generator Variant]
#     I --> J[Re-run RAG Evaluation Suite]
# ```

# %% [markdown]
# ## Workflow Diagram
#
# ```mermaid
# flowchart LR
#     Q[Load Existing Real Artifacts] --> R[Build SFT Examples]
#     R --> S[Split Train/Eval]
#     S --> T[Persist JSONL]
#     T --> U{RUN_FULL_TRAIN?}
#     U -->|No| V[Write Placeholder Reports]
#     U -->|Yes| W[Train LoRA Adapter]
#     W --> X[Export Adapter + Modelfile]
#     X --> Y[Compare Baseline vs Finetuned Metrics]
#     V --> Z[Result Analysis Template]
#     Y --> Z
# ```

# %% [markdown]
# ## Architecture and Workflow Diagram Explanation
# - `A -> B -> C`: We reuse existing real biomedical artifacts and create SFT-ready rows without synthetic generation.
# - `D -> E -> F`: Unsloth handles efficient model loading, PEFT injects LoRA adapters, TRL runs supervised fine-tuning loops.
# - `G -> H`: Adapter outputs are prepared for Ollama import via a generated Modelfile template.
# - `I -> J`: After explicit training/execution, the exact same evaluation stack from NB05/NB06/NB07 is re-used for objective comparison.

# %% [markdown]
# ## Component-by-Component Breakdown
# 1. **Data Constructor**
# Turns real extractive query-reference pairs into prompt-completion and message-form rows.
# 2. **Trainer Factory**
# Builds Unsloth + PEFT model and TRL `SFTTrainer` objects only when dependencies are available.
# 3. **Artifact Export Layer**
# Saves adapters, tokenizer, metadata, and Ollama Modelfile template.
# 4. **Evaluation Comparison Layer**
# Stores delta placeholders for retrieval, generation, RAG, and judge metrics.

# %% [markdown]
# ## When Should You Use This in Real Systems?
# - Use when retrieval quality is already acceptable but answer quality is still weak.
# - Use when you can maintain a strict, high-quality, domain-specific supervision dataset.
# - Avoid when your issue is mostly retrieval recall/precision; fix retrieval first.
#
# ## Advantages
# - Much cheaper than full fine-tuning.
# - Keeps base model reusable while swapping adapters.
# - Can improve answer policy adherence with relatively small datasets.
#
# ## Disadvantages
# - Adds a training lifecycle and model governance burden.
# - Requires careful dataset curation and regression evaluation.
# - Adapter compatibility and deployment path must be managed.

# %% [markdown]
# ## Comparison Against Other Implemented Variants
#
# | Variant | Primary Goal | Best Use Case | Main Tradeoff |
# |---|---|---|---|
# | Standard RAG | Simple grounding | Fast baseline | Weak under lexical mismatch |
# | Hybrid RAG (NB06) | Better retrieval recall/precision | Mixed semantic + exact biomedical terms | More retrieval complexity |
# | CRAG (NB07) | Reliability via correction | High-risk hallucination environments | Higher latency |
# | Multimodal RAG (NB08-10) | Use figure/table evidence | Visual/tabular-heavy biomedical tasks | Ingestion complexity |
# | **Selective Fine-Tuning (NB11)** | Improve generation behavior | Retrieval is already good, output quality still lags | Training and lifecycle overhead |

# %% [markdown]
# ## Implementation Decisions in This Project
# - We keep this notebook strictly additive and optional.
# - We do not replace existing generator defaults globally.
# - We do not train by default in implementation-only mode (`RUN_FULL_TRAIN=False`).
# - We export schema-complete placeholders for all post-run result analysis.
# - We keep evaluation interface unchanged so baseline and finetuned outputs are directly comparable.

# %%
# Input: project modules for real-data SFT construction and optional fine-tuning.
# Output: initialized runtime and notebook-level execution flags.
# Logic: import additive fine-tuning helpers and keep training disabled by default.
# Complexity: O(1).
from __future__ import annotations

import os
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path.cwd().parent))

from src.chunking import build_chunk_lookup, load_chunks
from src.config import settings
from src.data_pipeline import build_extractive_eval_queries, load_persisted_records
from src.finetune_data import (
    build_biomedical_sft_examples,
    build_chunk_id_lookup,
    persist_sft_jsonl,
    train_eval_split_sft,
)
from src.finetune_unsloth import (
    LoRAHyperParams,
    SFTTrainConfig,
    create_peft_lora_model_fallback,
    create_sft_trainer,
    create_unsloth_lora_model,
    finetune_stack_status,
    load_sft_dataset_dict,
    persist_finetune_placeholder_report,
    save_adapter_bundle,
    write_ollama_modelfile_template,
)
from src.llm_judge import grade_groundedness
from src.evaluator import GenerationExample, compute_generation_metrics, compute_rag_metrics
from src.utils import save_json

RUN_FULL_TRAIN = os.getenv("RUN_FULL_TRAIN", "false").strip().lower() == "true"
print(f"Judge model for downstream evaluation: {settings.judge_model}")
print(f"Generator base model (HF): {settings.finetune_base_model_hf}")
print(f"RUN_FULL_TRAIN={RUN_FULL_TRAIN}")

# %% [markdown]
# ## Step 1: Validate Optional Training Stack Availability
#
# This step does not install anything; it only reports whether optional dependencies are available.

# %%
# Input: local python environment.
# Output: package availability table for unsloth/peft/trl stack.
# Logic: inspect import availability for optional training packages.
# Complexity: O(number_of_packages_checked).
stack = finetune_stack_status()
stack_df = pd.DataFrame(
    [{"package": name, "available": available} for name, available in stack.items()]
).sort_values("package")
stack_df

# %% [markdown]
# ## Step 2: Build Real Biomedical SFT Dataset
#
# We derive training supervision from real MedMentions extractive query/reference pairs.
# No synthetic rows are generated.

# %%
# Input: persisted MedMentions records/chunks and extractive eval queries.
# Output: train/eval SFT examples as structured Python objects.
# Logic: map each query to grounded supporting chunks and build prompt/completion rows.
# Complexity: O(number_of_queries * average_supporting_chunks).
records = load_persisted_records()
chunks = load_chunks()
pmid_chunk_lookup = build_chunk_lookup(chunks)
chunk_id_lookup = build_chunk_id_lookup(chunks)

eval_queries = build_extractive_eval_queries(
    records=records,
    chunk_lookup=pmid_chunk_lookup,
    sample_size=min(settings.finetune_max_train_examples + settings.finetune_max_eval_examples, 2800),
)

sft_examples = build_biomedical_sft_examples(
    eval_queries=eval_queries,
    chunk_lookup=chunk_id_lookup,
    max_examples=settings.finetune_max_train_examples + settings.finetune_max_eval_examples,
)

train_examples, eval_examples = train_eval_split_sft(
    sft_examples,
    eval_fraction=max(
        0.01,
        min(
            0.4,
            settings.finetune_max_eval_examples / max(1, len(sft_examples)),
        ),
    ),
    seed=settings.random_seed,
)

print(f"Records: {len(records):,}")
print(f"Chunks: {len(chunks):,}")
print(f"Eval queries: {len(eval_queries):,}")
print(f"SFT examples total: {len(sft_examples):,}")
print(f"Train examples: {len(train_examples):,}")
print(f"Eval examples: {len(eval_examples):,}")

# %% [markdown]
# ## Step 3: Persist SFT JSONL Artifacts
#
# TRL training uses these JSONL rows in the explicit training phase.

# %%
# Input: train/eval SFT example objects.
# Output: JSONL artifact paths and sample preview table.
# Logic: persist reproducible train/eval rows to disk for optional trainer consumption.
# Complexity: O(number_of_examples).
dataset_paths = persist_sft_jsonl(
    train_examples=train_examples,
    eval_examples=eval_examples,
)
dataset_paths

preview_rows = [
    {
        "example_id": row.example_id,
        "source_pmid": row.source_pmid,
        "query": row.query[:140],
        "completion": row.completion[:160],
        "supporting_chunks": len(row.supporting_chunk_ids),
    }
    for row in train_examples[:10]
]
pd.DataFrame(preview_rows)

# %% [markdown]
# ## Step 4: Configure Unsloth + PEFT + TRL
#
# We prepare trainer configuration objects, but keep training disabled unless explicitly requested.

# %%
# Input: project settings.
# Output: typed LoRA and training config objects.
# Logic: instantiate hyperparameter dataclasses for repeatable training plans.
# Complexity: O(1).
lora_cfg = LoRAHyperParams(
    r=settings.finetune_lora_rank,
    lora_alpha=settings.finetune_lora_alpha,
    lora_dropout=settings.finetune_lora_dropout,
)

train_cfg = SFTTrainConfig(
    base_model=settings.finetune_base_model_hf,
    max_seq_length=settings.finetune_max_seq_length,
    train_batch_size=settings.finetune_train_batch_size,
    gradient_accumulation_steps=settings.finetune_grad_accumulation,
    learning_rate=settings.finetune_learning_rate,
    max_steps=settings.finetune_max_steps,
    warmup_steps=settings.finetune_warmup_steps,
)

pd.DataFrame(
    [
        {
            "base_model": train_cfg.base_model,
            "max_seq_length": train_cfg.max_seq_length,
            "lora_rank": lora_cfg.r,
            "lora_alpha": lora_cfg.lora_alpha,
            "lora_dropout": lora_cfg.lora_dropout,
            "train_batch_size": train_cfg.train_batch_size,
            "grad_accumulation": train_cfg.gradient_accumulation_steps,
            "learning_rate": train_cfg.learning_rate,
            "max_steps": train_cfg.max_steps,
            "warmup_steps": train_cfg.warmup_steps,
        }
    ]
)

# %% [markdown]
# ## Step 5: Optional Training Block (Disabled by Default)
#
# This block is intentionally guarded:
# - it runs only when `RUN_FULL_TRAIN=True`,
# - it requires optional dependencies to be installed,
# - it preserves placeholder behavior otherwise.

# %%
# Input: SFT JSONL datasets and config objects.
# Output: adapter export metadata and Ollama Modelfile template path (or placeholders).
# Logic: instantiate Unsloth+PEFT model, build TRL trainer, optionally train and export adapter.
# Complexity: O(training_steps * model_forward_backward_cost) when enabled.
adapter_metadata: dict | None = None
trainer_log_summary: dict = {
    "loss": None,
    "eval_loss": None,
    "steps": None,
    "tokens_per_second": None,
    "status": "placeholder_not_executed",
}
modelfile_path: str | None = None
training_backend = "placeholder"
baseline_eval_bundle: dict[str, dict] = {}
finetuned_eval_bundle: dict[str, dict] = {}


def _generate_local(model, tokenizer, prompt: str, max_new_tokens: int = 180) -> str:
    import torch

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=train_cfg.max_seq_length)
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
        model = model.cuda()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    if decoded.startswith(prompt):
        answer = decoded[len(prompt):].strip()
    else:
        answer = decoded.strip()
    return answer or decoded.strip()


def _evaluate_local_model(model, tokenizer, rows) -> dict[str, dict]:
    """Evaluate local model answers against real eval references."""
    eval_examples_local: list[GenerationExample] = []
    judge_payloads: list[dict] = []
    latencies_ms: list[float] = []

    for row in rows:
        start = time.perf_counter()
        try:
            answer = _generate_local(model, tokenizer, row.prompt)
        except Exception as exc:
            answer = f"Generation failed during local NB11 evaluation: {exc}"
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(float(elapsed_ms))

        context_chunks = row.supporting_chunk_texts[:8]
        eval_examples_local.append(
            GenerationExample(
                query=row.query,
                answer=answer,
                reference_answer=row.completion,
                context_chunks=context_chunks,
            )
        )
        judge_payloads.append(
            grade_groundedness(
                query=row.query,
                answer=answer,
                context="\n\n".join(context_chunks),
            )
        )

    generation_metrics_local = compute_generation_metrics(eval_examples_local, include_bertscore=True)
    rag_metrics_local = compute_rag_metrics(eval_examples_local)

    def _avg(key: str) -> float:
        values = []
        for item in judge_payloads:
            try:
                values.append(float(item.get(key, 0.0)))
            except Exception:
                values.append(0.0)
        return float(np.mean(values)) if values else 0.0

    judge_metrics_local = {
        "groundedness": _avg("groundedness"),
        "relevance": _avg("relevance"),
        "hallucination": _avg("hallucination_risk"),
        "completeness": _avg("completeness"),
    }
    latency_local = {
        "p50_ms": float(np.percentile(latencies_ms, 50)) if latencies_ms else 0.0,
        "p95_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
    }
    return {
        "generation": generation_metrics_local,
        "rag": rag_metrics_local,
        "judge": judge_metrics_local,
        "latency": latency_local,
    }


if RUN_FULL_TRAIN:
    try:
        try:
            model, tokenizer = create_unsloth_lora_model(lora=lora_cfg, cfg=train_cfg)
            training_backend = "unsloth_peft_trl"
        except Exception as exc:
            print(f"Unsloth path unavailable, using transformers+PEFT fallback: {exc}")
            model, tokenizer = create_peft_lora_model_fallback(lora=lora_cfg, cfg=train_cfg)
            training_backend = "transformers_peft_trl_fallback"

        dataset_dict = load_sft_dataset_dict(
            train_jsonl=dataset_paths["train_jsonl"],
            eval_jsonl=dataset_paths["eval_jsonl"],
        )
        trainer = create_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset_dict,
            cfg=train_cfg,
            output_dir=settings.finetune_adapter_dir / settings.finetune_adapter_name,
        )

        eval_subset = eval_examples[: min(20, len(eval_examples))]
        baseline_eval_bundle = _evaluate_local_model(model, tokenizer, eval_subset)

        try:
            train_start = time.perf_counter()
            train_result = trainer.train()
            train_seconds = time.perf_counter() - train_start
        except Exception as train_exc:
            if training_backend != "unsloth_peft_trl":
                raise

            print(f"Unsloth training failed, retrying with PEFT fallback backend: {train_exc}")
            model, tokenizer = create_peft_lora_model_fallback(lora=lora_cfg, cfg=train_cfg)
            training_backend = "transformers_peft_trl_fallback_retry"
            trainer = create_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                dataset=dataset_dict,
                cfg=train_cfg,
                output_dir=settings.finetune_adapter_dir / settings.finetune_adapter_name,
            )
            baseline_eval_bundle = _evaluate_local_model(model, tokenizer, eval_subset)
            train_start = time.perf_counter()
            train_result = trainer.train()
            train_seconds = time.perf_counter() - train_start

        finetuned_eval_bundle = _evaluate_local_model(model, tokenizer, eval_subset)

        adapter_metadata = save_adapter_bundle(
            model=model,
            tokenizer=tokenizer,
            adapter_dir=settings.finetune_adapter_dir / settings.finetune_adapter_name,
            train_cfg=train_cfg,
            lora_cfg=lora_cfg,
        )
        modelfile = write_ollama_modelfile_template(
            base_model="granite4.1:8b",
            adapter_path=settings.finetune_adapter_dir / settings.finetune_adapter_name,
            output_path=settings.finetune_adapter_dir / settings.finetune_adapter_name / "Modelfile",
        )
        modelfile_path = str(modelfile)
        trainer_log_summary = {
            "loss": getattr(train_result, "training_loss", None),
            "eval_loss": None,
            "steps": train_cfg.max_steps,
            "tokens_per_second": None,
            "train_seconds": float(train_seconds),
            "backend": training_backend,
            "status": "executed",
        }
    except Exception as exc:
        training_backend = "failed"
        trainer_log_summary = {
            "loss": None,
            "eval_loss": None,
            "steps": None,
            "tokens_per_second": None,
            "backend": training_backend,
            "status": "failed",
            "error": str(exc),
        }
        adapter_metadata = {
            "adapter_dir": str(settings.finetune_adapter_dir / settings.finetune_adapter_name),
            "base_model": train_cfg.base_model,
            "note": "RUN_FULL_TRAIN=True but training failed; see trainer_log_summary.error.",
        }
        modelfile_path = str(settings.finetune_adapter_dir / settings.finetune_adapter_name / "Modelfile")
else:
    adapter_metadata = {
        "adapter_dir": str(settings.finetune_adapter_dir / settings.finetune_adapter_name),
        "base_model": train_cfg.base_model,
        "note": "Placeholder only. Set RUN_FULL_TRAIN=True for real adapter export.",
    }
    modelfile_path = str(settings.finetune_adapter_dir / settings.finetune_adapter_name / "Modelfile")

adapter_metadata

# %% [markdown]
# ## Step 6: Evaluation Plan for Finetuned vs Baseline
#
# Required post-run metrics remain exactly the same to ensure comparability:
# - Retrieval: Precision@K, Recall@K, F1 Score, MRR, NDCG
# - Generation: Exact Match, BLEU, ROUGE, METEOR, BERTScore
# - RAG: Faithfulness, Context Precision, Context Recall, Answer Relevancy
# - Judge (`granite4.1:8b`): Groundedness, Relevance, Hallucination, Completeness

# %%
# Input: baseline and finetuned local-evaluation bundles.
# Output: baseline-vs-finetuned comparison payload.
# Logic: compute per-metric deltas after real training, else persist placeholders.
# Complexity: O(number_of_metrics).
if RUN_FULL_TRAIN and baseline_eval_bundle and finetuned_eval_bundle:
    def _delta(after: dict, before: dict, key: str) -> float:
        return float(after.get(key, 0.0) - before.get(key, 0.0))

    comparison_payload = {
        "mode": "executed",
        "baseline": baseline_eval_bundle,
        "finetuned": finetuned_eval_bundle,
        "baseline_vs_finetuned": {
            "retrieval_metrics": {
                "precision@k_delta": 0.0,
                "recall@k_delta": 0.0,
                "f1_delta": 0.0,
                "mrr_delta": 0.0,
                "ndcg_delta": 0.0,
                "note": "Retrieval layer unchanged in selective fine-tuning run.",
            },
            "generation_metrics": {
                "exact_match_delta": _delta(finetuned_eval_bundle["generation"], baseline_eval_bundle["generation"], "exact_match"),
                "bleu_delta": _delta(finetuned_eval_bundle["generation"], baseline_eval_bundle["generation"], "bleu"),
                "rouge1_delta": _delta(finetuned_eval_bundle["generation"], baseline_eval_bundle["generation"], "rouge1"),
                "rouge2_delta": _delta(finetuned_eval_bundle["generation"], baseline_eval_bundle["generation"], "rouge2"),
                "rougeL_delta": _delta(finetuned_eval_bundle["generation"], baseline_eval_bundle["generation"], "rougeL"),
                "meteor_delta": _delta(finetuned_eval_bundle["generation"], baseline_eval_bundle["generation"], "meteor"),
                "bertscore_f1_delta": _delta(
                    finetuned_eval_bundle["generation"],
                    baseline_eval_bundle["generation"],
                    "bertscore_f1",
                ),
            },
            "rag_metrics": {
                "faithfulness_delta": _delta(finetuned_eval_bundle["rag"], baseline_eval_bundle["rag"], "faithfulness"),
                "context_precision_delta": _delta(
                    finetuned_eval_bundle["rag"], baseline_eval_bundle["rag"], "context_precision"
                ),
                "context_recall_delta": _delta(
                    finetuned_eval_bundle["rag"], baseline_eval_bundle["rag"], "context_recall"
                ),
                "answer_relevancy_delta": _delta(
                    finetuned_eval_bundle["rag"], baseline_eval_bundle["rag"], "answer_relevancy"
                ),
            },
            "judge_metrics": {
                "groundedness_delta": _delta(finetuned_eval_bundle["judge"], baseline_eval_bundle["judge"], "groundedness"),
                "relevance_delta": _delta(finetuned_eval_bundle["judge"], baseline_eval_bundle["judge"], "relevance"),
                "hallucination_delta": _delta(
                    finetuned_eval_bundle["judge"], baseline_eval_bundle["judge"], "hallucination"
                ),
                "completeness_delta": _delta(finetuned_eval_bundle["judge"], baseline_eval_bundle["judge"], "completeness"),
            },
            "latency": {
                "p50_ms_delta": _delta(finetuned_eval_bundle["latency"], baseline_eval_bundle["latency"], "p50_ms"),
                "p95_ms_delta": _delta(finetuned_eval_bundle["latency"], baseline_eval_bundle["latency"], "p95_ms"),
            },
        },
        "notes": {
            "judge_model": settings.judge_model,
            "execution_phase": "executed",
            "training_backend": training_backend,
        },
    }
else:
    comparison_payload = {
        "mode": "placeholder",
        "baseline_vs_finetuned": {
            "retrieval_metrics": {
                "precision@k_delta": None,
                "recall@k_delta": None,
                "f1_delta": None,
                "mrr_delta": None,
                "ndcg_delta": None,
            },
            "generation_metrics": {
                "exact_match_delta": None,
                "bleu_delta": None,
                "rouge1_delta": None,
                "rouge2_delta": None,
                "rougeL_delta": None,
                "meteor_delta": None,
                "bertscore_f1_delta": None,
            },
            "rag_metrics": {
                "faithfulness_delta": None,
                "context_precision_delta": None,
                "context_recall_delta": None,
                "answer_relevancy_delta": None,
            },
            "judge_metrics": {
                "groundedness_delta": None,
                "relevance_delta": None,
                "hallucination_delta": None,
                "completeness_delta": None,
            },
            "latency": {
                "p50_ms_delta": None,
                "p95_ms_delta": None,
            },
        },
        "notes": {
            "judge_model": settings.judge_model,
            "execution_phase": "implementation_only_no_training",
        },
    }
if comparison_payload["mode"] == "executed":
    pd.DataFrame(
        [
            {"section": "retrieval", **comparison_payload["baseline_vs_finetuned"]["retrieval_metrics"]},
            {"section": "generation", **comparison_payload["baseline_vs_finetuned"]["generation_metrics"]},
            {"section": "rag", **comparison_payload["baseline_vs_finetuned"]["rag_metrics"]},
            {"section": "judge", **comparison_payload["baseline_vs_finetuned"]["judge_metrics"]},
            {"section": "latency", **comparison_payload["baseline_vs_finetuned"]["latency"]},
        ]
    )
else:
    pd.DataFrame(
        [
            {"section": "retrieval", "placeholder": "precision@k_delta / recall@k_delta / f1_delta / mrr_delta / ndcg_delta"},
            {"section": "generation", "placeholder": "exact_match_delta / bleu_delta / rouge_delta / meteor_delta / bertscore_delta"},
            {"section": "rag", "placeholder": "faithfulness_delta / context_precision_delta / context_recall_delta / answer_relevancy_delta"},
            {"section": "judge", "placeholder": "groundedness_delta / relevance_delta / hallucination_delta / completeness_delta"},
        ]
    )

# %% [markdown]
# ## Step 7: Persist NB11 Artifacts
#
# We save complete schemas now so post-run execution can populate real measured values without changing report structure.

# %%
# Input: dataset artifacts, training placeholders, and comparison payload.
# Output: persisted JSON metrics/report files and a compact CSV summary.
# Logic: write stable contracts for future execution and README result integration.
# Complexity: O(payload_size).
if RUN_FULL_TRAIN and comparison_payload["mode"] == "executed":
    placeholder_report = settings.finetune_reports_dir / "nb11_finetune_report.json"
    save_json(
        {
            "mode": "executed",
            "stack_status": stack,
            "dataset": {
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
            },
            "training_metrics": trainer_log_summary,
            "post_run_quality": comparison_payload["baseline_vs_finetuned"],
        },
        placeholder_report,
    )
else:
    placeholder_report = persist_finetune_placeholder_report(
        stack=stack,
        train_examples=len(train_examples),
        eval_examples=len(eval_examples),
    )

nb11_payload = {
    "mode": "placeholder" if not RUN_FULL_TRAIN else "executed",
    "stack": stack,
    "dataset": {
        "train_examples": len(train_examples),
        "eval_examples": len(eval_examples),
        "train_jsonl": str(dataset_paths["train_jsonl"]),
        "eval_jsonl": str(dataset_paths["eval_jsonl"]),
    },
    "lora_config": {
        "rank": lora_cfg.r,
        "alpha": lora_cfg.lora_alpha,
        "dropout": lora_cfg.lora_dropout,
        "target_modules": list(lora_cfg.target_modules),
    },
    "training_config": {
        "base_model": train_cfg.base_model,
        "max_seq_length": train_cfg.max_seq_length,
        "train_batch_size": train_cfg.train_batch_size,
        "gradient_accumulation_steps": train_cfg.gradient_accumulation_steps,
        "learning_rate": train_cfg.learning_rate,
        "max_steps": train_cfg.max_steps,
        "warmup_steps": train_cfg.warmup_steps,
    },
    "trainer_log_summary": trainer_log_summary,
    "adapter_metadata": adapter_metadata,
    "ollama_modelfile_path": modelfile_path,
    "comparison_payload": comparison_payload,
    "placeholder_report": str(placeholder_report),
}
save_json(nb11_payload, settings.metrics_dir / "nb11_selective_finetune_metrics.json")

pd.DataFrame(
    [
        {"item": "stack_ready", "value": all(stack.values())},
        {"item": "train_examples", "value": len(train_examples)},
        {"item": "eval_examples", "value": len(eval_examples)},
        {"item": "run_full_train", "value": RUN_FULL_TRAIN},
        {"item": "adapter_path", "value": adapter_metadata.get("adapter_dir", "") if adapter_metadata else ""},
    ]
).to_csv(settings.tables_dir / "nb11_selective_finetune_summary.csv", index=False)

print("Saved NB11 selective fine-tuning artifacts (placeholders unless RUN_FULL_TRAIN=True).")

# %% [markdown]
# ## Post-Run Result Analysis Template (Populate After Explicit Execution)
#
# ### 1. Actual Outputs and Artifacts
# - Add training logs, adapter export proof, and Ollama model creation logs.
#
# ### 2. Metric Interpretation
# - Interpret actual Retrieval metrics (Precision@K, Recall@K, F1 Score, MRR, NDCG).
# - Interpret actual Generation metrics (Exact Match, BLEU, ROUGE, METEOR, BERTScore).
# - Interpret actual RAG metrics (Faithfulness, Context Precision, Context Recall, Answer Relevancy).
# - Interpret actual Judge metrics (Groundedness, Relevance, Hallucination, Completeness).
#
# ### 3. Latency and Efficiency Analysis
# - Compare baseline vs finetuned P50/P95 latency and cost implications.
#
# ### 4. What Changed Because of Unsloth, PEFT, and TRL?
# - Unsloth: report memory/speed changes observed during training.
# - PEFT: report adapter size and deployment flexibility vs full-model tuning.
# - TRL: report training stability and reproducibility behavior.
#
# ### 5. Lessons Learned and Practical Takeaways
# - Document failure modes, guardrails, and tuning recommendations.
#
# ### 6. Final Conclusion
# Summarize whether selective fine-tuning was worth the extra complexity for this medical RAG system based on real measured results.
