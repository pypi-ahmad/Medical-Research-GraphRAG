"""Contract checks for canonical tutorial notebooks."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK_DIR = PROJECT_ROOT / "notebooks"

CANONICAL_NOTEBOOKS = [
    "NB01_Data_Exploration.ipynb",
    "NB02_Chroma_GraphRAG.ipynb",
    "NB03_Pinecone_GraphRAG.ipynb",
    "NB04_Agentic_GraphRAG.ipynb",
    "NB05_Evaluation.ipynb",
]


def _load_notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_text(nb: dict) -> str:
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "markdown":
            parts.extend(cell.get("source", []))
    return "\n".join(parts)


def _code_text(nb: dict) -> str:
    parts: list[str] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            parts.extend(cell.get("source", []))
    return "\n".join(parts)


def test_canonical_notebooks_exist() -> None:
    for name in CANONICAL_NOTEBOOKS:
        assert (NOTEBOOK_DIR / name).exists(), f"Missing canonical notebook: {name}"


def test_notebooks_include_zero_to_hero_sections() -> None:
    required_markers = [
        "## What",
        "## Why",
        "Tradeoffs",
        "Alternatives",
        "Production Considerations",
    ]
    for name in CANONICAL_NOTEBOOKS:
        text = _markdown_text(_load_notebook(NOTEBOOK_DIR / name))
        for marker in required_markers:
            assert marker in text, f"{name} missing section marker: {marker}"


def test_notebooks_include_cell_contract_comments() -> None:
    comment_markers = ["# Input:", "# Output:", "# Logic:", "# Complexity:"]
    for name in CANONICAL_NOTEBOOKS:
        text = _code_text(_load_notebook(NOTEBOOK_DIR / name))
        for marker in comment_markers:
            assert marker in text, f"{name} missing code comment marker: {marker}"
