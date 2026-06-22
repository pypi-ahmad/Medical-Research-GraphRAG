"""Contract checks for additive NB06-NB11 tutorial notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

ADDITIONAL_NOTEBOOKS = [
    "NB06_Hybrid_RAG.ipynb",
    "NB07_CRAG.ipynb",
    "NB08_Multimodal_RAG.ipynb",
    "NB09_Multimodal_RAG_OCR_CLI.ipynb",
    "NB10_Multimodal_RAG_Vision_Qwen.ipynb",
    "NB11_Selective_Finetuning_Unsloth_PEFT_TRL.ipynb",
]


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown(nb: dict) -> str:
    rows: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            rows.extend(cell.get("source", []))
    return "\n".join(rows)


def _code(nb: dict) -> str:
    rows: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            rows.extend(cell.get("source", []))
    return "\n".join(rows)


def test_additional_notebooks_exist() -> None:
    for name in ADDITIONAL_NOTEBOOKS:
        assert (NOTEBOOK_DIR / name).exists(), f"Missing notebook: {name}"


def test_additional_notebooks_have_tutorial_sections() -> None:
    required_markers = [
        "## What",
        "## Why",
        "## Tradeoffs",
        "## Alternatives",
        "## Production Considerations",
        "## Definition and Core Concepts",
        "## Component-by-Component Breakdown",
        "## Comparison Against",
        "## Implementation Decisions",
        "## Post-Run Result Analysis Template",
    ]
    for name in ADDITIONAL_NOTEBOOKS:
        md = _markdown(_load(NOTEBOOK_DIR / name))
        for marker in required_markers:
            assert marker in md, f"{name} missing marker: {marker}"


def test_additional_notebooks_have_metric_terms() -> None:
    required_terms = [
        "Precision@K",
        "Recall@K",
        "MRR",
        "NDCG",
        "BLEU",
        "ROUGE",
        "METEOR",
        "BERTScore",
        "Faithfulness",
        "Context Precision",
        "Context Recall",
        "Answer Relevancy",
    ]
    for name in ADDITIONAL_NOTEBOOKS:
        md = _markdown(_load(NOTEBOOK_DIR / name))
        for term in required_terms:
            assert term in md, f"{name} missing metric term: {term}"


def test_additional_notebooks_have_cell_contract_comments() -> None:
    markers = ["# Input:", "# Output:", "# Logic:", "# Complexity:"]
    for name in ADDITIONAL_NOTEBOOKS:
        code = _code(_load(NOTEBOOK_DIR / name))
        for marker in markers:
            assert marker in code, f"{name} missing code marker: {marker}"
