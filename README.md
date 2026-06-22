# Medical Research GraphRAG

Biomedical AI engineering project for medically grounded question answering using real MedMentions data and multiple RAG variants.

## Overview
This repository implements and evaluates:
- GraphRAG (Chroma baseline + Pinecone comparison)
- Agentic GraphRAG (LangGraph state machine)
- Hybrid RAG (dense + sparse biomedical retrieval)
- Corrective RAG (CRAG)
- Multimodal RAG (OCR CLI and vision extraction)
- Unified evaluation (retrieval, generation, RAG, LLM judge)
- Optional selective fine-tuning track (Unsloth + PEFT + TRL)

## Ground-Truth Policy
All documentation in this repository is grounded in:
1. Local source code under `src/` and notebook scripts under `notebooks/*.py`.
2. Real execution artifacts under `outputs/`.
3. Real run logs under `outputs/logs/` and state under `outputs/run_state/`.

No synthetic metrics or placeholder results are presented as completed outcomes.

## Real Data + Model Stack
- Dataset: `bigbio/medmentions` (real records only; no synthetic records)
- Effective corpus in latest run: 4,392 records
  - `train=2,635`, `validation=878`, `test=879`
- Embeddings: `qwen3-embedding:4b` (Ollama)
- Generator: `granite4.1:8b` (Ollama)
- Judge: `granite4.1:8b` (Ollama)
- OCR (multimodal): `glm-ocr` via `ollama run`
- Vision (multimodal): `qwen3.5:4b`

## Latest Real Run Provenance (June 22, 2026)
Primary sources:
- Run state: `outputs/run_state/full_real_pipeline.state`
- Full run log: `outputs/logs/full_real_pipeline_20260622_110834.log`
- Per-notebook logs: `outputs/logs/NB*_20260622_110834.log`

State summary (`full_real_pipeline.state`):
- Preflight + model steps: done
- NB01 through NB11: done
- `pytest_q`: done

## Exact Workflow Map

| Stage | Notebook | Primary implementation modules | Main artifacts |
|---|---|---|---|
| Data foundation | `NB01_Data_Exploration` | `src/data_pipeline.py` | `outputs/tables/nb01_*`, `outputs/figures/nb01_*` |
| Chroma GraphRAG | `NB02_Chroma_GraphRAG` | `src/chroma_retriever.py`, `src/graph_builder.py`, `src/chunking.py`, `src/embeddings.py` | `outputs/tables/nb02_*`, `outputs/figures/nb02_*` |
| Pinecone GraphRAG | `NB03_Pinecone_GraphRAG` | `src/pinecone_retriever.py` + shared graph stack | `outputs/metrics/nb03_retrieval_benchmark.json`, `outputs/tables/nb03_chroma_vs_pinecone.csv` |
| Agentic GraphRAG | `NB04_Agentic_GraphRAG` | `src/agentic_rag.py` | `outputs/metrics/nb04_agentic_demo.json`, `outputs/tables/nb04_agentic_route_summary.csv` |
| Unified evaluation | `NB05_Evaluation` | `src/evaluator.py`, `src/llm_judge.py` | `outputs/metrics/nb05_evaluation_bundle.json`, `outputs/tables/nb05_metric_summary.csv` |
| Hybrid RAG | `NB06_Hybrid_RAG` | `src/hybrid_retriever.py` | `outputs/metrics/nb06_hybrid_rag_metrics.json` |
| CRAG | `NB07_CRAG` | `src/crag_pipeline.py` | `outputs/metrics/nb07_crag_metrics.json`, `outputs/tables/nb07_crag_route_summary.csv` |
| Multimodal baseline | `NB08_Multimodal_RAG` | `src/multimodal_rag.py`, `src/multimodal_assets_pmc.py` | `outputs/metrics/nb08_multimodal_rag_metrics.json` |
| Multimodal CLI OCR | `NB09_Multimodal_RAG_OCR_CLI` | `src/multimodal_rag.py` | `outputs/metrics/nb09_multimodal_ocr_cli_metrics.json` |
| Multimodal vision | `NB10_Multimodal_RAG_Vision_Qwen` | `src/multimodal_vision_rag.py` | `outputs/metrics/nb10_multimodal_qwen_vision_metrics.json` |
| Optional fine-tune track | `NB11_Selective_Finetuning_Unsloth_PEFT_TRL` | `src/finetune_data.py`, `src/finetune_unsloth.py` | `outputs/metrics/nb11_selective_finetune_metrics.json`, `outputs/finetune/` |

## Key Results Snapshot (Latest Artifacts)

| Area | Key values |
|---|---|
| NB03 Chroma latency | p50 `5973ms`, p95 `6673ms`, p99 `6773ms` |
| NB03 Pinecone latency | p50 `9658ms`, p95 `11889ms`, p99 `14087ms` |
| NB05 baseline retrieval (`k=8`) | precision `0.0417`, recall `0.3000`, MRR `0.2162`, NDCG `0.2324` |
| NB06 hybrid retrieval (`k=8`) | precision `0.0625`, recall `0.4500`, MRR `0.3931`, NDCG `0.4101` |
| NB07 CRAG retrieval (`k=8`) | precision `0.0781`, recall `0.3750`, MRR `0.3125`, NDCG `0.3314` |
| NB08 multimodal retrieval (`k=8`) | precision `0.1250`, recall `1.0000`, MRR `1.0000`, NDCG `1.0000` |
| NB09 multimodal CLI retrieval (`k=8`) | precision `0.1250`, recall `1.0000`, MRR `1.0000`, NDCG `1.0000` |
| NB10 multimodal vision retrieval (`k=8`) | precision `0.1667`, recall `1.0000`, MRR `1.0000`, NDCG `1.0000` |
| NB11 latest trainer status | `failed` with `'LlamaAttention' object has no attribute 'apply_qkv'` |

## Environment and Execution

### Setup
```bash
cd /home/ahmad/AI/Medical-Research-GraphRAG

if [ ! -d .venv ]; then
  uv python install 3.12.10
  uv venv --python 3.12.10 .venv
fi

source .venv/bin/activate
uv sync --extra dev --extra finetune
```

### Required Ollama models
```bash
ollama pull qwen3-embedding:4b
ollama pull granite4.1:8b
ollama pull glm-ocr
ollama pull qwen3.5:4b
```

### Notebook execution scripts
- Canonical baseline: `bash scripts/execute_notebooks.sh`
- Additive notebooks: `bash scripts/execute_additional_notebooks.sh`
- Strict resumable run: `bash scripts/run_full_real_pipeline_strict.sh`

## Documentation Index
- Main handbook: `docs/handbook.md`
- Evidence ledger: `docs/evidence_ledger.md`
- Tutorial chapters: `docs/tutorials/*.md`
- Combined publication markdown: `docs/documentation.md`
- PDF: `docs/documentation.pdf`

## Repository Size Policy
To keep default GitHub clones lightweight, large generated runtime artifacts are intentionally not versioned:
- `chroma_db/`
- `graphs/relation_edges.json`
- `graphs/entity_graph.pkl`
- `data/processed/chunk_embeddings.npy`
- `data/processed/medmentions_records.json`
- `outputs/metrics/nb04_agentic_demo.json`

Regenerate these locally by running:
```bash
bash scripts/run_full_real_pipeline_strict.sh
```

## Current Limitations (Evidence-Based)
1. Latest NB11 metrics artifact reports training backend failure; baseline-vs-finetuned deltas are null in `comparison_payload`.
2. `BERTScore` fields are `0.0` in current metric artifacts.
3. Multimodal assets are real open biomedical/health datasets (OWID-based chart/table pipeline), not radiology/pathology imaging datasets.

## Safety Note
This is an engineering research system, not a clinical decision support product. Human clinical review and governance are required for medical deployment.
