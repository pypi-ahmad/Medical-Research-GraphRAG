"""Tests for judge compatibility shim and score normalization."""

from __future__ import annotations

from src.guardian_judge import _extract_json, guardian_json
from src.llm_judge import grade_retrieval_quality


def test_extract_json_embedded_payload() -> None:
    raw = "Some text before {\"retrieval_quality\": 0.8, \"reason\": \"ok\"} after"
    parsed = _extract_json(raw)
    assert parsed["retrieval_quality"] == 0.8
    assert parsed["reason"] == "ok"


def test_guardian_json_delegates_to_judge(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.guardian_judge.judge_json",
        lambda prompt, model=None, temperature=0.0: {"ok": True, "prompt": prompt},
    )
    payload = guardian_json("hello")
    assert payload["ok"] is True
    assert payload["prompt"] == "hello"


def test_grade_retrieval_quality_clamps(monkeypatch) -> None:
    monkeypatch.setattr(
        "src.llm_judge.judge_json",
        lambda prompt, model=None, temperature=0.0: {
            "retrieval_quality": 9,
            "reason": "too high in raw payload",
            "missing_aspects": [],
        },
    )
    payload = grade_retrieval_quality("q", [{"text": "doc"}])
    assert payload["retrieval_quality"] == 1.0
