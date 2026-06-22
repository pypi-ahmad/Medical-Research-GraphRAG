"""Backward-compatible shim for legacy guardian judge imports.

The project-standard judge model is now `settings.judge_model` (granite4.1:8b).
This module keeps old imports functional while delegating to `src.llm_judge`.
"""

from __future__ import annotations

from src.llm_judge import _extract_json, grade_groundedness, grade_retrieval_quality, judge_json


def guardian_json(prompt: str, model: str | None = None, temperature: float = 0.0):
    """Compatibility wrapper for older call sites."""
    return judge_json(prompt=prompt, model=model, temperature=temperature)


__all__ = [
    "_extract_json",
    "guardian_json",
    "grade_retrieval_quality",
    "grade_groundedness",
]
