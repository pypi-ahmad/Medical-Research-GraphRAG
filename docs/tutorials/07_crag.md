# 07. Corrective RAG (CRAG)

## What is this technique?
CRAG is a quality-gated RAG controller that:
1. grades retrieval quality,
2. corrects the query when quality is low,
3. falls back to web evidence when retries are exhausted,
4. verifies groundedness before finalizing.

## Definition and core concepts
- **Retrieval grader**: judge-based quality score.
- **Corrective loop**: bounded query rewrite + re-retrieval attempts.
- **Fallback path**: external evidence retrieval.
- **Verification gate**: groundedness check after generation.

## Why was this developed?
In biomedical QA, weak retrieval can silently propagate into hallucinated outputs. CRAG makes retrieval quality a first-class control signal.

## What limitation of traditional RAG does it solve?
Traditional RAG often has no explicit correction policy when retrieval is poor. CRAG introduces deterministic routing and bounded recovery behavior.

## CRAG workflow diagram

```mermaid
flowchart TD
    A[Query] --> B[Hybrid retrieval]
    B --> C[Retrieval grader]
    C -->|Accept| D[Context expansion]
    C -->|Correct| E[Query rewrite]
    E --> B
    C -->|Fallback| F[Web fallback]
    F --> D
    D --> G[Answer generation]
    G --> H[Grounding verification]
    H -->|Grounded| I[Finalize]
    H -->|Not grounded (bounded)| F
```

## How it appears in code
`src/crag_pipeline.py`:
- State and resources: `CRAGState`, `CRAGResources` (20-48)
- Query rewrite: `_rewrite_query_with_llm` (72-100)
- Workflow assembly: `build_crag_workflow` (102-267)
- Run helpers: `run_crag_query`, `run_crag_batch` (269-298)

Dependencies:
- Hybrid retrieval from `src/hybrid_retriever.py`
- Judge grading from `src/llm_judge.py`

Notebook:
- `notebooks/NB07_CRAG.py`

## Component-by-component breakdown
1. Retrieve via hybrid dense+sparse channel.
2. Grade retrieval quality (`retrieval_quality` + missing aspects).
3. Route to accept/correct/fallback.
4. Build context and generate answer.
5. Verify groundedness; optionally one fallback retry.
6. Persist route traces.

## Real outputs
- Metrics: `outputs/metrics/nb07_crag_metrics.json`
- Route table: `outputs/tables/nb07_crag_route_summary.csv`

Latest key values:
- Retrieval (`k=8`): precision `0.0781`, recall `0.3750`, MRR `0.3125`, NDCG `0.3314`
- RAG: faithfulness `0.8938`, answer_relevancy `0.9813`

Route examples from `nb07_crag_route_summary.csv`:
- Diabetes query: direct finalize route, no corrections.
- KRAS pancreatic query: 2 query corrections + web fallback + verify retry before finalize.

## Why CRAG over simpler alternatives?
- Better reliability than one-pass RAG.
- More auditable than ad-hoc retries.

## When should this be used?
- High-risk domains needing explicit fallback/verification behavior.
- Environments where retrieval quality varies significantly by query type.

## Advantages
- Transparent and bounded correction policy.
- Strong traceability for postmortem/debug.

## Disadvantages
- Increased latency and model-call cost.
- Requires threshold tuning (`crag_acceptance_threshold`, `crag_max_corrections`).

## Comparison against other implemented variants
- Hybrid improves retrieval quality.
- CRAG improves reliability when retrieval is still weak.
- Agentic GraphRAG offers broader orchestration; CRAG is stricter corrective policy.

## Production considerations
- Monitor route frequencies (`accept`, `correct`, `web_fallback`).
- Keep correction and verification limits bounded.
- Restrict fallback sources for medical safety requirements.

## Conclusion
CRAG provides a reliability control plane that is explicit, measurable, and suitable for medically grounded QA pipelines.
