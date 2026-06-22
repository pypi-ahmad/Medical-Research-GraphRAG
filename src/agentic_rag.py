"""LangGraph-based Agentic GraphRAG workflow implementation."""

from __future__ import annotations

import concurrent.futures as cf
import json
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal, TypedDict

import ollama
from duckduckgo_search import DDGS
from langgraph.graph import END, StateGraph
from loguru import logger

from src.chroma_retriever import reciprocal_rank_fusion, vector_search
from src.chunking import ChunkRecord
from src.config import settings
from src.graph_builder import (
    community_summary_text,
    concept_id_from_query,
    local_graph_expansion,
    rank_communities_for_query,
)


class AgentState(TypedDict, total=False):
    """Mutable state shared across LangGraph nodes."""

    query: str
    retrieved_docs: list[dict[str, Any]]
    retrieval_score: float
    extracted_concept_ids: list[str]
    graph_traversal: dict[str, Any]
    selected_communities: list[dict[str, Any]]
    web_results: list[dict[str, str]]
    expanded_context: str
    answer_draft: str
    hallucination_score: float
    retries: int
    final_answer: str
    route: str
    trace: list[str]


@dataclass(slots=True)
class AgentResources:
    """External resources required by the agent workflow."""

    chroma_collection: Any
    chunks: list[ChunkRecord]
    graph: Any
    partition: dict[str, int]
    summaries: list[dict[str, Any]]


def _append_trace(state: AgentState, msg: str) -> None:
    """Append deterministic trace message to pipeline state."""
    state.setdefault("trace", []).append(msg)


def _chat_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    options: dict[str, Any] | None = None,
    response_format: str | None = None,
    timeout_seconds: float = 120.0,
) -> dict[str, Any]:
    """Execute Ollama chat with hard timeout and host routing."""
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "options": options or {},
    }
    if response_format is not None:
        payload["format"] = response_format

    host = os.getenv("OLLAMA_HOST")

    def _run_request() -> dict[str, Any]:
        client = ollama.Client(host=host)
        response = client.chat(**payload)
        if hasattr(response, "model_dump"):
            response = response.model_dump()
        return response

    with cf.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_request)
        try:
            return future.result(timeout=timeout_seconds)
        except cf.TimeoutError as exc:
            future.cancel()
            raise TimeoutError(f"Ollama chat timed out after {timeout_seconds}s") from exc


def _judge_retrieval(query: str, docs: list[dict[str, Any]]) -> float:
    """Score retrieval quality in [0, 1] using LLM judge."""
    context = "\n\n".join(doc["text"][:500] for doc in docs[:6])
    prompt = f"""
You are grading retrieval quality for biomedical QA.
Question: {query}
Retrieved context:
{context}
Return JSON only: {{"retrieval_quality": <float between 0 and 1>}}
"""

    try:
        response = _chat_completion(
            model=settings.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format="json",
            options={"temperature": 0.0},
            timeout_seconds=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Retrieval judge call failed: {}", exc)
        return 0.0

    try:
        payload = json.loads(response["message"]["content"])
        score = float(payload.get("retrieval_quality", 0.0))
    except Exception:
        score = 0.0

    return float(min(max(score, 0.0), 1.0))


def _hallucination_score(query: str, answer: str, context: str) -> float:
    """Score answer grounding in [0, 1], where higher is safer."""
    prompt = f"""
Question: {query}
Answer: {answer}
Context:
{context[:5000]}

Rate groundedness in context and return JSON only:
{{"groundedness": <float between 0 and 1>}}
"""

    try:
        response = _chat_completion(
            model=settings.judge_model,
            messages=[{"role": "user", "content": prompt}],
            response_format="json",
            options={"temperature": 0.0},
            timeout_seconds=30.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Hallucination judge call failed: {}", exc)
        return 0.0

    try:
        payload = json.loads(response["message"]["content"])
        score = float(payload.get("groundedness", 0.0))
    except Exception:
        score = 0.0

    return float(min(max(score, 0.0), 1.0))


def _web_search(query: str, max_results: int = 3) -> list[dict[str, str]]:
    """Fallback web search for low-quality retrieval scenarios."""
    results: list[dict[str, str]] = []
    try:
        # Keep web fallback bounded to avoid long-tail hangs during notebook execution.
        with DDGS(timeout=12) as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(
                    {
                        "title": item.get("title", ""),
                        "snippet": item.get("body", ""),
                        "url": item.get("href", ""),
                    }
                )
    except Exception as exc:
        logger.warning("Web fallback search failed: {}", exc)
    return results


def build_agentic_workflow(resources: AgentResources):
    """Build and compile the LangGraph workflow with explicit node semantics."""

    def retrieval_node(state: AgentState) -> AgentState:
        _append_trace(state, "retrieval")
        docs = vector_search(resources.chroma_collection, state["query"], top_k=settings.top_k_retrieval)
        state["retrieved_docs"] = docs
        state["extracted_concept_ids"] = concept_id_from_query(state["query"], resources.chunks)
        return state

    def retrieval_grader_node(state: AgentState) -> AgentState:
        _append_trace(state, "retrieval_grader")
        score = _judge_retrieval(state["query"], state.get("retrieved_docs", []))
        state["retrieval_score"] = score
        state["route"] = "web_fallback" if score < settings.retrieval_grade_threshold else "graph_traversal"
        return state

    def web_search_node(state: AgentState) -> AgentState:
        _append_trace(state, "web_search")
        state["web_results"] = _web_search(state["query"], max_results=3)
        return state

    def graph_traversal_node(state: AgentState) -> AgentState:
        _append_trace(state, "graph_traversal")
        concept_ids = state.get("extracted_concept_ids", [])
        traversal = local_graph_expansion(resources.graph, concept_ids, hops=settings.local_graph_hops)
        selected = rank_communities_for_query(concept_ids, resources.partition, resources.summaries, top_k=3)

        state["graph_traversal"] = traversal
        state["selected_communities"] = selected
        return state

    def context_expansion_node(state: AgentState) -> AgentState:
        _append_trace(state, "context_expansion")

        sections: list[str] = []

        docs = state.get("retrieved_docs", [])
        if docs:
            sections.append("### Retrieved Chunks")
            for idx, item in enumerate(docs[:6], start=1):
                sections.append(f"[{idx}] {item['text'][:900]}")

        traversal = state.get("graph_traversal", {})
        if traversal.get("nodes"):
            sections.append("### Graph Traversal Nodes")
            sections.append(", ".join(traversal["nodes"][:60]))

        communities = state.get("selected_communities", [])
        if communities:
            sections.append("### Community Summaries")
            for community in communities:
                sections.append(community_summary_text(community))

        web_results = state.get("web_results", [])
        if web_results:
            sections.append("### Web Fallback Evidence")
            for item in web_results:
                sections.append(f"- {item['title']}: {item['snippet'][:320]} ({item['url']})")

        state["expanded_context"] = "\n\n".join(sections)
        return state

    def answer_generation_node(state: AgentState) -> AgentState:
        _append_trace(state, "answer_generation")

        system_prompt = (
            "You are a biomedical research assistant. Answer only from provided evidence. "
            "If evidence is insufficient, explicitly say what is missing. "
            "Cite numbered evidence snippets [1], [2], ... when possible."
        )
        user_prompt = f"""
Question: {state['query']}

Evidence:
{state.get('expanded_context', '')}

Write a concise, factual biomedical answer with citations.
"""
        try:
            response = _chat_completion(
                model=settings.generator_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                options={"temperature": 0.2},
                timeout_seconds=60.0,
            )
            state["answer_draft"] = response["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Answer generation call failed: {}", exc)
            state["answer_draft"] = (
                "The system could not complete generation within the configured timeout. "
                "Please retry with a narrower biomedical question."
            )
        return state

    def hallucination_detection_node(state: AgentState) -> AgentState:
        _append_trace(state, "hallucination_detection")

        score = _hallucination_score(
            state["query"],
            state.get("answer_draft", ""),
            state.get("expanded_context", ""),
        )
        state["hallucination_score"] = score

        retries = state.get("retries", 0)
        if score < settings.hallucination_threshold and retries < 1:
            state["retries"] = retries + 1
        else:
            state["retries"] = retries
        return state

    def final_response_node(state: AgentState) -> AgentState:
        _append_trace(state, "final_response")
        state["final_answer"] = state.get("answer_draft", "")
        return state

    workflow = StateGraph(AgentState)

    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("retrieval_grader", retrieval_grader_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("graph_traversal", graph_traversal_node)
    workflow.add_node("context_expansion", context_expansion_node)
    workflow.add_node("answer_generation", answer_generation_node)
    workflow.add_node("hallucination_detection", hallucination_detection_node)
    workflow.add_node("final_response", final_response_node)

    workflow.set_entry_point("retrieval")
    workflow.add_edge("retrieval", "retrieval_grader")

    workflow.add_conditional_edges(
        "retrieval_grader",
        lambda state: state.get("route", "graph_traversal"),
        {
            "web_fallback": "web_search",
            "graph_traversal": "graph_traversal",
        },
    )

    workflow.add_edge("web_search", "graph_traversal")
    workflow.add_edge("graph_traversal", "context_expansion")
    workflow.add_edge("context_expansion", "answer_generation")
    workflow.add_edge("answer_generation", "hallucination_detection")

    workflow.add_conditional_edges(
        "hallucination_detection",
        lambda state: "answer_generation"
        if state.get("hallucination_score", 0.0) < settings.hallucination_threshold
        and state.get("retries", 0) > 0
        else "final_response",
        {
            "answer_generation": "answer_generation",
            "final_response": "final_response",
        },
    )

    workflow.add_edge("final_response", END)

    return workflow.compile()


def run_agentic_query(app, query: str) -> dict[str, Any]:
    """Run a single query through the compiled agentic workflow."""
    thread_id = f"thread-{uuid.uuid4().hex[:12]}"
    state: AgentState = {
        "query": query,
        "retrieved_docs": [],
        "retrieval_score": 0.0,
        "extracted_concept_ids": [],
        "graph_traversal": {},
        "selected_communities": [],
        "web_results": [],
        "expanded_context": "",
        "answer_draft": "",
        "hallucination_score": 0.0,
        "retries": 0,
        "final_answer": "",
        "route": "",
        "trace": [],
    }

    result = app.invoke(state, config={"configurable": {"thread_id": thread_id}})
    return dict(result)


def run_agentic_batch(app, queries: list[str]) -> list[dict[str, Any]]:
    """Run multiple queries sequentially and collect final states."""
    outputs = []
    for query in queries:
        outputs.append(run_agentic_query(app, query))
    return outputs


def workflow_mermaid() -> str:
    """Return Mermaid graph showing required agentic GraphRAG flow."""
    return """
flowchart TD
    A[Query] --> B[Retrieval]
    B --> C[Retrieval Grader]
    C -->|Poor| D[Web Search Fallback]
    C -->|Good| E[Graph Traversal]
    D --> E
    E --> F[Context Expansion]
    F --> G[Answer Generation]
    G --> H[Hallucination Detection]
    H -->|Low Grounding, retry once| G
    H --> I[Final Response]
""".strip()
