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

run_nb notebooks/NB06_Hybrid_RAG.ipynb
run_nb notebooks/NB07_CRAG.ipynb
run_nb notebooks/NB08_Multimodal_RAG.ipynb
run_nb notebooks/NB09_Multimodal_RAG_OCR_CLI.ipynb
run_nb notebooks/NB10_Multimodal_RAG_Vision_Qwen.ipynb
run_nb notebooks/NB11_Selective_Finetuning_Unsloth_PEFT_TRL.ipynb

echo "Additional notebook execution complete."
