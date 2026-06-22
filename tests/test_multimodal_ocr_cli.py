"""Tests for CLI-first multimodal OCR behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from src.multimodal_rag import _glm_ocr_cli_command, extract_text_with_glm_ocr_with_backend


def test_glm_ocr_cli_command_shape(tmp_path: Path) -> None:
    image_path = tmp_path / "figure.png"
    command = _glm_ocr_cli_command(
        image_path=image_path,
        query="extract text",
        model="glm-ocr",
    )
    assert command[0:3] == ["ollama", "run", "glm-ocr"]
    assert command[3] == str(image_path)
    assert command[4] == "extract text"


def test_extract_text_with_glm_ocr_with_backend_cli_success(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"

    monkeypatch.setattr(
        "src.multimodal_rag.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="OCR output", stderr=""),
    )

    text, backend = extract_text_with_glm_ocr_with_backend(
        image_path=image_path,
        model="glm-ocr",
        allow_fallback=False,
        timeout_seconds=30,
    )
    assert text == "OCR output"
    assert backend == "ollama_run"


def test_extract_text_with_glm_ocr_with_backend_fallback(monkeypatch, tmp_path: Path) -> None:
    image_path = tmp_path / "chart.png"

    def _raise(*args, **kwargs):
        raise RuntimeError("cli unavailable")

    monkeypatch.setattr("src.multimodal_rag.subprocess.run", _raise)
    monkeypatch.setattr(
        "src.multimodal_rag.ollama.chat",
        lambda **kwargs: {"message": {"content": "fallback ocr text"}},
    )

    text, backend = extract_text_with_glm_ocr_with_backend(
        image_path=image_path,
        model="glm-ocr",
        allow_fallback=True,
        timeout_seconds=30,
    )
    assert text == "fallback ocr text"
    assert backend == "ollama_chat_fallback"
