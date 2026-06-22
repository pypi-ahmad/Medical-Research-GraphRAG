"""Comprehensive evaluation suite for GraphRAG and Agentic RAG."""

from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import re
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import ollama
from loguru import logger
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer

from src.config import settings
from src.utils import timer


@dataclass(slots=True)
class RetrievalExample:
    """One retrieval evaluation sample with ranked output and known relevant IDs."""

    retrieved_ids: list[str]
    relevant_ids: list[str]


@dataclass(slots=True)
class GenerationExample:
    """One generation evaluation sample with context and reference answer."""

    query: str
    answer: str
    reference_answer: str
    context_chunks: list[str]


@dataclass(slots=True)
class EvaluationBundle:
    """Unified evaluation payload for notebook and README reporting."""

    retrieval_metrics: dict[str, float]
    generation_metrics: dict[str, float]
    rag_metrics: dict[str, float]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert bundle to a plain dictionary."""
        return asdict(self)


def _normalize_text(text: str) -> str:
    """Normalize text for exact-match style metrics."""
    lowered = text.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered.strip()


def precision_at_k(relevant_flags: list[int], k: int) -> float:
    """Compute precision at rank k."""
    if k <= 0:
        return 0.0
    top = relevant_flags[:k]
    if not top:
        return 0.0
    return float(sum(top) / len(top))


def recall_at_k(relevant_flags: list[int], total_relevant: int, k: int) -> float:
    """Compute recall at rank k."""
    if k <= 0 or total_relevant <= 0:
        return 0.0
    return float(sum(relevant_flags[:k]) / total_relevant)


def f1_score(precision: float, recall: float) -> float:
    """Compute F1 from precision and recall."""
    if precision + recall == 0:
        return 0.0
    return float((2 * precision * recall) / (precision + recall))


def reciprocal_rank(relevant_flags: list[int]) -> float:
    """Compute reciprocal rank for one query."""
    for rank, flag in enumerate(relevant_flags, start=1):
        if flag:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(relevant_flags: list[int], k: int) -> float:
    """Compute NDCG@k for binary relevance."""
    values = relevant_flags[:k]
    if not values:
        return 0.0

    dcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(values))
    ideal = sorted(values, reverse=True)
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal))
    if idcg == 0:
        return 0.0
    return float(dcg / idcg)


@timer
def compute_retrieval_metrics(
    examples: list[RetrievalExample],
    k_values: list[int] | None = None,
) -> dict[str, float]:
    """Compute retrieval metrics over multiple examples."""
    if not examples:
        return {}

    k_values = k_values or [1, 3, 5, 8, 10]

    metrics: dict[str, float] = {}
    for k in k_values:
        p_scores, r_scores, f_scores, n_scores = [], [], [], []
        for ex in examples:
            relevant_set = set(ex.relevant_ids)
            flags = [1 if rid in relevant_set else 0 for rid in ex.retrieved_ids]
            p = precision_at_k(flags, k)
            r = recall_at_k(flags, total_relevant=max(1, len(relevant_set)), k=k)
            p_scores.append(p)
            r_scores.append(r)
            f_scores.append(f1_score(p, r))
            n_scores.append(ndcg_at_k(flags, k))

        metrics[f"precision@{k}"] = float(np.mean(p_scores))
        metrics[f"recall@{k}"] = float(np.mean(r_scores))
        metrics[f"f1@{k}"] = float(np.mean(f_scores))
        metrics[f"ndcg@{k}"] = float(np.mean(n_scores))

    rr_scores = []
    for ex in examples:
        relevant_set = set(ex.relevant_ids)
        flags = [1 if rid in relevant_set else 0 for rid in ex.retrieved_ids]
        rr_scores.append(reciprocal_rank(flags))

    metrics["mrr"] = float(np.mean(rr_scores))
    return metrics


def exact_match(reference: str, candidate: str) -> float:
    """Compute exact match after normalization."""
    return float(_normalize_text(reference) == _normalize_text(candidate))


def bleu(reference: str, candidate: str) -> float:
    """Compute sentence-level BLEU with smoothing."""
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if not ref_tokens or not cand_tokens:
        return 0.0

    smoothing = SmoothingFunction().method1
    return float(
        sentence_bleu(
            [ref_tokens],
            cand_tokens,
            smoothing_function=smoothing,
            weights=(0.25, 0.25, 0.25, 0.25),
        )
    )


def rouge(reference: str, candidate: str) -> dict[str, float]:
    """Compute ROUGE-1/2/L F1 scores."""
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference, candidate)
    return {
        "rouge1": float(scores["rouge1"].fmeasure),
        "rouge2": float(scores["rouge2"].fmeasure),
        "rougeL": float(scores["rougeL"].fmeasure),
    }


def meteor(reference: str, candidate: str) -> float:
    """Compute METEOR metric."""
    try:
        return float(meteor_score([reference.split()], candidate.split()))
    except Exception:
        return 0.0


@timer
def bertscore_batch(
    references: list[str],
    candidates: list[str],
    model_type: str = "prajjwal1/bert-tiny",
) -> dict[str, float]:
    """Compute BERTScore precision/recall/F1 averages."""
    try:
        from bert_score import score as bert_score_fn

        p, r, f1 = bert_score_fn(candidates, references, model_type=model_type, verbose=False)
        return {
            "bertscore_precision": float(np.mean(p.cpu().numpy())),
            "bertscore_recall": float(np.mean(r.cpu().numpy())),
            "bertscore_f1": float(np.mean(f1.cpu().numpy())),
        }
    except Exception as exc:
        logger.warning("BERTScore failed: {}", exc)
        return {
            "bertscore_precision": 0.0,
            "bertscore_recall": 0.0,
            "bertscore_f1": 0.0,
        }


@timer
def compute_generation_metrics(
    examples: list[GenerationExample],
    include_bertscore: bool = True,
) -> dict[str, float]:
    """Compute EM, BLEU, ROUGE, METEOR, and optional BERTScore."""
    if not examples:
        return {}

    em_scores, bleu_scores, meteor_scores = [], [], []
    rouge1_scores, rouge2_scores, rougeL_scores = [], [], []

    references, candidates = [], []

    for ex in examples:
        references.append(ex.reference_answer)
        candidates.append(ex.answer)

        em_scores.append(exact_match(ex.reference_answer, ex.answer))
        bleu_scores.append(bleu(ex.reference_answer, ex.answer))
        meteor_scores.append(meteor(ex.reference_answer, ex.answer))

        rouge_scores = rouge(ex.reference_answer, ex.answer)
        rouge1_scores.append(rouge_scores["rouge1"])
        rouge2_scores.append(rouge_scores["rouge2"])
        rougeL_scores.append(rouge_scores["rougeL"])

    metrics = {
        "exact_match": float(np.mean(em_scores)),
        "bleu": float(np.mean(bleu_scores)),
        "rouge1": float(np.mean(rouge1_scores)),
        "rouge2": float(np.mean(rouge2_scores)),
        "rougeL": float(np.mean(rougeL_scores)),
        "meteor": float(np.mean(meteor_scores)),
    }

    if include_bertscore:
        metrics.update(bertscore_batch(references, candidates))

    return metrics


def _judge_json(prompt: str, model: str | None = None, temperature: float = 0.0) -> dict[str, Any]:
    """Run an Ollama judge prompt and parse JSON output robustly."""
    model_name = model or settings.judge_model

    try:
        def _worker(host: str | None, queue: mp.Queue) -> None:
            try:
                client = ollama.Client(host=host)
                response = client.chat(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    format="json",
                    options={"temperature": temperature},
                )
                if hasattr(response, "model_dump"):
                    response = response.model_dump()
                queue.put({"ok": True, "content": response["message"]["content"]})
            except Exception as exc:  # noqa: BLE001
                queue.put({"ok": False, "error": str(exc)})

        queue: mp.Queue = mp.Queue(maxsize=1)
        proc = mp.Process(target=_worker, args=(os.getenv("OLLAMA_HOST"), queue), daemon=True)
        proc.start()
        proc.join(90.0)
        if proc.is_alive():
            proc.terminate()
            proc.join(timeout=2)
            logger.warning("Judge call timeout after 90s")
            return {}
        if queue.empty():
            logger.warning("Judge process exited without payload")
            return {}
        result = queue.get()
        if not result.get("ok", False):
            logger.warning("Judge worker failed: {}", result.get("error", "unknown"))
            return {}
        content = str(result.get("content", ""))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Judge call failed: {}", exc)
        return {}

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}


def faithfulness(query: str, answer: str, context_chunks: list[str]) -> float:
    """Score faithfulness by checking support of answer claims in retrieved context."""
    context = "\n\n".join(context_chunks[:8])
    prompt = f"""
You are auditing RAG faithfulness.
Question: {query}
Answer: {answer}
Context:
{context}

Return JSON only:
{{"faithfulness": <float between 0 and 1>, "reason": "short reason"}}
"""
    payload = _judge_json(prompt, temperature=0.0)
    score = payload.get("faithfulness", 0.0)
    try:
        return float(min(max(score, 0.0), 1.0))
    except Exception:
        return 0.0


def context_precision(query: str, context_chunks: list[str]) -> float:
    """Estimate context precision: fraction of chunks relevant to query."""
    if not context_chunks:
        return 0.0

    prompt = f"""
You are grading retrieval relevance.
Question: {query}
Retrieved Chunks:
{json.dumps(context_chunks[:8], ensure_ascii=True)}

Return JSON only:
{{"relevance_labels": [0 or 1 for each chunk in order]}}
"""
    payload = _judge_json(prompt, temperature=0.0)
    labels = payload.get("relevance_labels", [])
    if not labels:
        return 0.0

    labels = [1 if int(v) >= 1 else 0 for v in labels[: len(context_chunks[:8])]]
    return float(np.mean(labels)) if labels else 0.0


def context_recall(query: str, reference_answer: str, context_chunks: list[str]) -> float:
    """Estimate context recall: does retrieved context cover reference evidence?"""
    context = "\n\n".join(context_chunks[:8])
    prompt = f"""
You are grading whether retrieved context covers the gold reference evidence.
Question: {query}
Reference Answer: {reference_answer}
Context:
{context}

Return JSON only:
{{"context_recall": <float between 0 and 1>}}
"""
    payload = _judge_json(prompt, temperature=0.0)
    score = payload.get("context_recall", 0.0)
    try:
        return float(min(max(score, 0.0), 1.0))
    except Exception:
        return 0.0


def answer_relevancy(query: str, answer: str) -> float:
    """Estimate answer relevancy using judge model."""
    prompt = f"""
Question: {query}
Answer: {answer}
Return JSON only: {{"answer_relevancy": <float between 0 and 1>}}
"""
    payload = _judge_json(prompt, temperature=0.0)
    score = payload.get("answer_relevancy", 0.0)
    try:
        return float(min(max(score, 0.0), 1.0))
    except Exception:
        return 0.0


def llm_judge_axes(query: str, answer: str, context_chunks: list[str]) -> dict[str, float]:
    """Judge groundedness, relevance, hallucination, and completeness (1-5 scale)."""
    context = "\n\n".join(context_chunks[:8])
    prompt = f"""
You are an expert biomedical QA evaluator.
Question: {query}
Answer: {answer}
Context:
{context}

Score on a 1-5 scale and return JSON only:
{{
  "groundedness": <1-5>,
  "relevance": <1-5>,
  "hallucination": <1-5 where 5 means very low hallucination>,
  "completeness": <1-5>
}}
"""
    payload = _judge_json(prompt, temperature=0.0)

    def clamp(v: Any) -> float:
        try:
            return float(min(max(float(v), 1.0), 5.0))
        except Exception:
            return 1.0

    return {
        "groundedness": clamp(payload.get("groundedness", 1)),
        "relevance": clamp(payload.get("relevance", 1)),
        "hallucination": clamp(payload.get("hallucination", 1)),
        "completeness": clamp(payload.get("completeness", 1)),
    }


@timer
def compute_rag_metrics(examples: list[GenerationExample]) -> dict[str, float]:
    """Compute RAG-specific metrics over generation examples."""
    if not examples:
        return {}

    faithfulness_scores = []
    ctx_precision_scores = []
    ctx_recall_scores = []
    answer_rel_scores = []

    groundedness, relevance, hallucination, completeness = [], [], [], []

    for ex in examples:
        faithfulness_scores.append(faithfulness(ex.query, ex.answer, ex.context_chunks))
        ctx_precision_scores.append(context_precision(ex.query, ex.context_chunks))
        ctx_recall_scores.append(context_recall(ex.query, ex.reference_answer, ex.context_chunks))
        answer_rel_scores.append(answer_relevancy(ex.query, ex.answer))

        judge = llm_judge_axes(ex.query, ex.answer, ex.context_chunks)
        groundedness.append(judge["groundedness"])
        relevance.append(judge["relevance"])
        hallucination.append(judge["hallucination"])
        completeness.append(judge["completeness"])

    return {
        "faithfulness": float(np.mean(faithfulness_scores)),
        "context_precision": float(np.mean(ctx_precision_scores)),
        "context_recall": float(np.mean(ctx_recall_scores)),
        "answer_relevancy": float(np.mean(answer_rel_scores)),
        "judge_groundedness": float(np.mean(groundedness)),
        "judge_relevance": float(np.mean(relevance)),
        "judge_hallucination": float(np.mean(hallucination)),
        "judge_completeness": float(np.mean(completeness)),
    }


def build_evaluation_bundle(
    retrieval_examples: list[RetrievalExample],
    generation_examples: list[GenerationExample],
    *,
    k_values: list[int] | None = None,
    include_bertscore: bool = True,
    metadata: dict[str, Any] | None = None,
) -> EvaluationBundle:
    """Build one structured evaluation payload across all metric families."""
    retrieval_metrics = compute_retrieval_metrics(retrieval_examples, k_values=k_values)
    generation_metrics = compute_generation_metrics(
        generation_examples,
        include_bertscore=include_bertscore,
    )
    rag_metrics = compute_rag_metrics(generation_examples)

    final_metadata = dict(metadata or {})
    final_metadata.setdefault("retrieval_example_count", len(retrieval_examples))
    final_metadata.setdefault("generation_example_count", len(generation_examples))
    final_metadata.setdefault("judge_model", settings.judge_model)
    final_metadata.setdefault("bertscore_enabled", include_bertscore)

    return EvaluationBundle(
        retrieval_metrics=retrieval_metrics,
        generation_metrics=generation_metrics,
        rag_metrics=rag_metrics,
        metadata=final_metadata,
    )
