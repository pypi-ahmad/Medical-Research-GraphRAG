"""Corrective RAG (CRAG) pipeline with explicit quality-gated routing."""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import Any, TypedDict

import ollama
from duckduckgo_search import DDGS
from langgraph.graph import END, StateGraph
from loguru import logger

from src.config import settings
from src.llm_judge import grade_groundedness, grade_retrieval_quality
from src.hybrid_retriever import BiomedicalSparseIndex, hybrid_search


class CRAGState(TypedDict, total=False):
    """Mutable CRAG state shared across workflow nodes."""

    query: str
    corrected_query: str
    retrieval_rows: list[dict[str, Any]]
    retrieval_grade: float
    retrieval_reason: str
    missing_aspects: list[str]
    correction_attempts: int
    web_rows: list[dict[str, Any]]
    route: str
    expanded_context: str
    answer: str
    groundedness: float
    hallucination_risk: float
    completeness: float
    verify_attempts: int
    final_answer: str
    trace: list[str]


@dataclass(slots=True)
class CRAGResources:
    """External resources required by standalone CRAG workflow."""

    chroma_collection: Any
    sparse_index: BiomedicalSparseIndex


def _append_trace(state: CRAGState, label: str) -> None:
    state.setdefault("trace", []).append(label)


def _safe_web_search(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Run web fallback search with resilience to transient failures."""
    rows: list[dict[str, str]] = []
    try:
        with DDGS(timeout=12) as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                rows.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("body", ""),
                        "url": item.get("href", ""),
                    }
                )
    except Exception as exc:
        logger.warning("CRAG web fallback failed: {}", exc)
    return rows


def _rewrite_query_with_llm(
    query: str,
    missing_aspects: list[str],
) -> str:
    """Rewrite the query using missing-aspect guidance from retrieval grading."""
    gaps = "\n".join(f"- {item}" for item in missing_aspects[:8]) if missing_aspects else "- none"
    prompt = f"""
You are improving a biomedical information retrieval query.
Original Query: {query}
Missing Aspects:
{gaps}

Return one improved biomedical query only (plain text, no markdown).
"""
    try:
        timeout_seconds = float(os.getenv("OLLAMA_GENERATION_TIMEOUT_SECONDS", "90"))
        host = os.getenv("OLLAMA_HOST")
        client = ollama.Client(host=host, timeout=timeout_seconds)
        response = client.chat(
            model=settings.generator_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.0},
        )
        candidate = response["message"]["content"].strip()
    except Exception as exc:
        logger.warning("CRAG query rewrite failed; using original query. Error: {}", exc)
        candidate = query
    return candidate or query


def build_crag_workflow(resources: CRAGResources):
    """Build standalone CRAG state machine with corrective routing."""

    def retrieve_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "retrieve_hybrid")
        query = state.get("corrected_query") or state["query"]
        rows = hybrid_search(
            collection=resources.chroma_collection,
            sparse_index=resources.sparse_index,
            query=query,
            top_k=settings.top_k_retrieval,
            use_rrf=False,
        )
        state["retrieval_rows"] = rows
        return state

    def grade_retrieval_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "grade_retrieval")
        query = state.get("corrected_query") or state["query"]
        payload = grade_retrieval_quality(query, state.get("retrieval_rows", []))
        state["retrieval_grade"] = float(payload.get("retrieval_quality", 0.0))
        state["retrieval_reason"] = str(payload.get("reason", ""))
        state["missing_aspects"] = list(payload.get("missing_aspects", []))

        attempts = int(state.get("correction_attempts", 0))
        if state["retrieval_grade"] >= settings.crag_acceptance_threshold:
            state["route"] = "accept"
        elif attempts < settings.crag_max_corrections:
            state["route"] = "correct"
        else:
            state["route"] = "web_fallback"
        return state

    def query_correction_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "query_correction")
        attempts = int(state.get("correction_attempts", 0)) + 1
        state["correction_attempts"] = attempts
        state["corrected_query"] = _rewrite_query_with_llm(
            query=state.get("corrected_query") or state["query"],
            missing_aspects=state.get("missing_aspects", []),
        )
        return state

    def web_fallback_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "web_fallback")
        query = state.get("corrected_query") or state["query"]
        state["web_rows"] = _safe_web_search(query, max_results=5)
        return state

    def context_expansion_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "context_expansion")
        parts: list[str] = []

        rows = state.get("retrieval_rows", [])
        if rows:
            parts.append("### Hybrid Retrieval Evidence")
            for idx, row in enumerate(rows[:8], start=1):
                src = ",".join(row.get("sources", [])) if "sources" in row else row.get("source", "")
                parts.append(f"[{idx}] ({src}) {row.get('text', '')[:900]}")

        web_rows = state.get("web_rows", [])
        if web_rows:
            parts.append("### Corrective Web Evidence")
            for row in web_rows:
                parts.append(f"- {row.get('title', '')}: {row.get('snippet', '')} ({row.get('url', '')})")

        state["expanded_context"] = "\n\n".join(parts)
        return state

    def answer_generation_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "answer_generation")
        question = state.get("corrected_query") or state["query"]
        prompt = f"""
You are a biomedical research assistant.
Question: {question}

Evidence:
{state.get("expanded_context", "")}

Rules:
1) Use only the provided evidence.
2) If evidence is insufficient, say what is missing.
3) Cite evidence snippets as [1], [2], ... where possible.
"""
        try:
            timeout_seconds = float(os.getenv("OLLAMA_GENERATION_TIMEOUT_SECONDS", "120"))
            host = os.getenv("OLLAMA_HOST")
            client = ollama.Client(host=host, timeout=timeout_seconds)
            response = client.chat(
                model=settings.generator_model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": 0.2},
            )
            state["answer"] = response["message"]["content"]
        except Exception as exc:
            logger.warning("CRAG answer generation failed: {}", exc)
            state["answer"] = (
                "Insufficient evidence due to temporary generation backend failure. "
                "Please retry."
            )
        return state

    def verify_answer_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "verify_answer")
        question = state.get("corrected_query") or state["query"]
        payload = grade_groundedness(
            query=question,
            answer=state.get("answer", ""),
            context=state.get("expanded_context", ""),
        )
        state["groundedness"] = float(payload.get("groundedness", 0.0))
        state["hallucination_risk"] = float(payload.get("hallucination_risk", 0.0))
        state["completeness"] = float(payload.get("completeness", 0.0))

        verify_attempts = int(state.get("verify_attempts", 0))
        if state["groundedness"] >= settings.hallucination_threshold or verify_attempts >= 1:
            state["route"] = "finalize"
        else:
            state["verify_attempts"] = verify_attempts + 1
            state["route"] = "web_fallback"
        return state

    def finalize_node(state: CRAGState) -> CRAGState:
        _append_trace(state, "finalize")
        state["final_answer"] = state.get("answer", "")
        return state

    workflow = StateGraph(CRAGState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_retrieval", grade_retrieval_node)
    workflow.add_node("query_correction", query_correction_node)
    workflow.add_node("web_fallback", web_fallback_node)
    workflow.add_node("context_expansion", context_expansion_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("verify_answer", verify_answer_node)
    workflow.add_node("finalize", finalize_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_retrieval")

    workflow.add_conditional_edges(
        "grade_retrieval",
        lambda state: state.get("route", "accept"),
        {
            "accept": "context_expansion",
            "correct": "query_correction",
            "web_fallback": "web_fallback",
        },
    )
    workflow.add_edge("query_correction", "retrieve")
    workflow.add_edge("web_fallback", "context_expansion")
    workflow.add_edge("context_expansion", "answer_generation")
    workflow.add_edge("answer_generation", "verify_answer")

    workflow.add_conditional_edges(
        "verify_answer",
        lambda state: state.get("route", "finalize"),
        {
            "finalize": "finalize",
            "web_fallback": "web_fallback",
        },
    )

    workflow.add_edge("finalize", END)
    return workflow.compile()


def run_crag_query(app, query: str) -> dict[str, Any]:
    """Execute one query through the CRAG workflow."""
    state: CRAGState = {
        "query": query,
        "corrected_query": query,
        "retrieval_rows": [],
        "retrieval_grade": 0.0,
        "retrieval_reason": "",
        "missing_aspects": [],
        "correction_attempts": 0,
        "web_rows": [],
        "route": "",
        "expanded_context": "",
        "answer": "",
        "groundedness": 0.0,
        "hallucination_risk": 0.0,
        "completeness": 0.0,
        "verify_attempts": 0,
        "final_answer": "",
        "trace": [],
    }
    run_id = f"crag-{uuid.uuid4().hex[:12]}"
    result = app.invoke(state, config={"configurable": {"thread_id": run_id}})
    return dict(result)


def run_crag_batch(app, queries: list[str]) -> list[dict[str, Any]]:
    """Execute multiple CRAG queries and collect final states."""
    return [run_crag_query(app, query) for query in queries]


def crag_mermaid() -> str:
    """Return CRAG workflow diagram."""
    return """
flowchart TD
    A[Query] --> B[Hybrid Retrieval]
    B --> C[Retrieval Grader]
    C -->|High quality| D[Context Expansion]
    C -->|Low quality, retries left| E[Query Correction]
    E --> B
    C -->|Low quality, retries exhausted| F[Web Fallback]
    F --> D
    D --> G[Answer Generation]
    G --> H[Judge Verification]
    H -->|Grounded| I[Final Response]
    H -->|Ungrounded, one retry| F
""".strip()
