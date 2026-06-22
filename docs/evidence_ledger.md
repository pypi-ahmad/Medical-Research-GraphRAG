# Evidence Ledger (Source of Truth)

This file records the exact local artifacts used as ground truth for the documentation set.

## Provenance Policy
Claims are sourced from:
1. Repository code (`src/`, `notebooks/*.py`, `scripts/`)
2. Run state and logs (`outputs/run_state/`, `outputs/logs/`)
3. Metrics/tables/figures (`outputs/metrics/`, `outputs/tables/`, `outputs/figures/`)

## Latest Full Run (Detected)
- State file: `outputs/run_state/full_real_pipeline.state`
- Latest full run log: `outputs/logs/full_real_pipeline_20260622_110834.log`
- Log window: `2026-06-22 11:08:34` to `2026-06-22 12:08:49`

### Run-state keys marked done
- preflight/model fetch steps
- NB01 through NB11 notebook steps
- pytest gate (`pytest_q`)

## Primary Metric Artifacts (latest timestamps)
- `outputs/metrics/nb03_retrieval_benchmark.json` (Jun 22, 10:03)
- `outputs/metrics/nb05_evaluation_bundle.json` (Jun 22, 10:20)
- `outputs/metrics/nb06_hybrid_rag_metrics.json` (Jun 22, 10:27)
- `outputs/metrics/nb07_crag_metrics.json` (Jun 22, 11:28)
- `outputs/metrics/nb08_multimodal_rag_metrics.json` (Jun 22, 11:37)
- `outputs/metrics/nb09_multimodal_ocr_cli_metrics.json` (Jun 22, 11:42)
- `outputs/metrics/nb10_multimodal_qwen_vision_metrics.json` (Jun 22, 11:52)
- `outputs/metrics/nb11_selective_finetune_metrics.json` (Jun 22, 12:06)

## Key Numeric Facts Referenced in Docs

### NB03 backend benchmark
- Chroma latency p50/p95/p99: `5973.14 / 6672.91 / 6773.33 ms`
- Pinecone latency p50/p95/p99: `9657.66 / 11888.92 / 14086.93 ms`

### NB05 baseline evaluation
- Retrieval (`k=8`): precision `0.0417`, recall `0.3000`, MRR `0.2162`, NDCG `0.2324`
- RAG: faithfulness `0.9458`, answer_relevancy `0.9833`

### NB06 hybrid
- Retrieval (`k=8`): precision `0.0625`, recall `0.4500`, MRR `0.3931`, NDCG `0.4101`

### NB07 CRAG
- Retrieval (`k=8`): precision `0.0781`, recall `0.3750`, MRR `0.3125`, NDCG `0.3314`

### NB08/NB09/NB10 multimodal
- NB08 (`k=8`): precision `0.1250`, recall `1.0000`, MRR `1.0000`, NDCG `1.0000`
- NB09 (`k=8`): precision `0.1250`, recall `1.0000`, MRR `1.0000`, NDCG `1.0000`
- NB10 (`k=8`): precision `0.1667`, recall `1.0000`, MRR `1.0000`, NDCG `1.0000`

### NB11 (latest metrics artifact)
- `trainer_log_summary.backend = failed`
- `trainer_log_summary.error = 'LlamaAttention' object has no attribute 'apply_qkv'`
- `comparison_payload.mode = placeholder`

## Known Artifact Tension (Documented Neutrality)
- `outputs/metrics/nb11_selective_finetune_metrics.json` reports a failed training backend in latest run.
- `outputs/finetune/adapters/medresearch-lora/` contains existing adapter files from an earlier timestamp.
- Documentation therefore treats NB11 as: implemented pipeline present, latest run comparison deltas not available.

## Script-Level Evidence
- Strict run orchestrator: `scripts/run_full_real_pipeline_strict.sh`
- Notebook order in script includes NB01 -> NB11 and a final `pytest -q` step.
- Resume semantics are driven by `outputs/run_state/full_real_pipeline.state`.

## Documentation Scope Guard
Only documentation files are modified during this documentation pass:
- `README.md`
- `docs/*.md`
- `docs/documentation.pdf`
- optional docs-only rendering intermediates under `docs/`
