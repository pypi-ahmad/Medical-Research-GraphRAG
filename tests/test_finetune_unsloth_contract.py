"""Contract tests for optional Unsloth/PEFT/TRL helper utilities."""

from __future__ import annotations

import json
from pathlib import Path

from src.finetune_unsloth import (
    finetune_stack_status,
    persist_finetune_placeholder_report,
    write_ollama_modelfile_template,
)


def test_finetune_stack_status_keys() -> None:
    payload = finetune_stack_status()
    for key in ["unsloth", "peft", "trl", "datasets", "transformers", "accelerate"]:
        assert key in payload


def test_write_ollama_modelfile_template(tmp_path: Path) -> None:
    path = write_ollama_modelfile_template(
        base_model="granite4.1:8b",
        adapter_path=tmp_path / "adapter",
        output_path=tmp_path / "Modelfile",
        system_prompt="Ground biomedical claims in evidence.",
    )
    text = path.read_text(encoding="utf-8")
    assert "FROM granite4.1:8b" in text
    assert "ADAPTER" in text
    assert "SYSTEM" in text


def test_persist_finetune_placeholder_report(tmp_path: Path) -> None:
    report = persist_finetune_placeholder_report(
        stack={"unsloth": False, "peft": True, "trl": True},
        train_examples=100,
        eval_examples=20,
        out_path=tmp_path / "report.json",
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["mode"] == "placeholder"
    assert payload["dataset"]["train_examples"] == 100
    assert payload["dataset"]["eval_examples"] == 20
