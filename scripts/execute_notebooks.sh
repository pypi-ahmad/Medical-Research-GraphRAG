#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  uv python install 3.12.10
  uv venv --python 3.12.10 .venv
  source .venv/bin/activate
  uv sync
else
  source .venv/bin/activate
fi

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export RUN_FULL_EVAL="${RUN_FULL_EVAL:-false}"
export RUN_FULL_TRAIN="${RUN_FULL_TRAIN:-false}"

run_nb() {
  local nb="$1"
  local out="${nb%.ipynb}.executed.ipynb"
  echo "Executing $nb"
  jupyter nbconvert --to notebook --execute "$nb" --output "$(basename "$out")" --output-dir notebooks
}

run_nb notebooks/NB01_Data_Exploration.ipynb
run_nb notebooks/NB02_Chroma_GraphRAG.ipynb
run_nb notebooks/NB03_Pinecone_GraphRAG.ipynb
run_nb notebooks/NB04_Agentic_GraphRAG.ipynb
run_nb notebooks/NB05_Evaluation.ipynb

echo "Notebook execution complete."
