"""LLM-as-a-judge helpers powered by the project judge model."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

import ollama
from loguru import logger

from src.config import settings


def _extract_json(content: str) -> dict[str, Any]:
    """Parse JSON payloads robustly from judge output."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", content, flags=re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}


def judge_json(
    prompt: str,
    model: str | None = None,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """Run a judge prompt and return structured JSON."""
    retries = max(1, int(os.getenv("OLLAMA_JUDGE_RETRIES", "3")))
    timeout_seconds = float(os.getenv("OLLAMA_JUDGE_TIMEOUT_SECONDS", "90"))
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            client = ollama.Client(host=host, timeout=timeout_seconds)
            response = client.chat(
                model=model or settings.judge_model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": temperature},
            )
            payload = _extract_json(response["message"]["content"])
            if payload:
                return payload
            logger.warning(
                "Judge returned empty/non-JSON payload on attempt {}/{}",
                attempt,
                retries,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Judge call attempt {}/{} failed: {}",
                attempt,
                retries,
                exc,
            )
        if attempt < retries:
            time.sleep(min(2 * attempt, 6))

    logger.error("Judge failed after {} attempts. Returning empty payload. Last error: {}", retries, last_exc)
    return {}


def grade_retrieval_quality(query: str, docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Grade retrieval quality with a scalar score and rationale."""
    context = "\n\n".join(doc.get("text", "")[:600] for doc in docs[:8])
    prompt = f"""
You are a biomedical retrieval quality auditor.
Question: {query}
Retrieved Context:
{context}

Return strict JSON:
{{
  "retrieval_quality": <float 0 to 1>,
  "reason": "<short rationale>",
  "missing_aspects": ["<optional gaps>"]
}}
"""
    payload = judge_json(prompt, temperature=0.0)
    score = payload.get("retrieval_quality", 0.0)
    try:
        payload["retrieval_quality"] = float(min(max(float(score), 0.0), 1.0))
    except Exception:
        payload["retrieval_quality"] = 0.0
    payload.setdefault("reason", "")
    payload.setdefault("missing_aspects", [])
    return payload


def grade_groundedness(query: str, answer: str, context: str) -> dict[str, Any]:
    """Grade biomedical groundedness and hallucination risk."""
    prompt = f"""
You are a biomedical answer-grounding auditor.
Question: {query}
Answer: {answer}
Context:
{context[:8000]}

Return strict JSON:
{{
  "groundedness": <float 0 to 1>,
  "hallucination_risk": <float 0 to 1 where 1 is highest risk>,
  "relevance": <float 0 to 1>,
  "completeness": <float 0 to 1>,
  "reason": "<short rationale>"
}}
"""
    payload = judge_json(prompt, temperature=0.0)
    for key in ["groundedness", "hallucination_risk", "relevance", "completeness"]:
        value = payload.get(key, 0.0)
        try:
            payload[key] = float(min(max(float(value), 0.0), 1.0))
        except Exception:
            payload[key] = 0.0
    payload.setdefault("reason", "")
    return payload
