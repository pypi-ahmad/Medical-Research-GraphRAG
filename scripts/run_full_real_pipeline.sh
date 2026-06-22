#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [ ! -d .venv ]; then
  uv python install 3.12.10
  uv venv --python 3.12.10 .venv
fi

source .venv/bin/activate
uv sync --extra dev --extra finetune

if [ -z "${PINECONE_API_KEY:-}" ]; then
  echo "PINECONE_API_KEY is required for real NB03 Pinecone execution." >&2
  exit 1
fi

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export RUN_FULL_EVAL="true"
export RUN_FULL_TRAIN="true"

if ! ollama list >/tmp/ollama_list_pre.txt 2>&1; then
  echo "Starting ollama server..."
  nohup ollama serve >/tmp/ollama_serve.log 2>&1 &
  sleep 5
fi

ensure_model() {
  local model="$1"
  if ! ollama list | awk 'NR>1 {print $1}' | grep -Fx "$model" >/dev/null 2>&1; then
    echo "Pulling $model ..."
    ollama pull "$model"
  fi
}

ensure_model "qwen3-embedding:4b"
ensure_model "granite4.1:8b"
ensure_model "glm-ocr"
ensure_model "qwen3.5:4b"

python scripts/fetch_pmc_multimodal_assets.py --max-images 5 --max-tables 3

run_nb() {
  local nb="$1"
  local out="${nb%.ipynb}.executed.ipynb"
  echo "Executing $nb"
  jupyter nbconvert \
    --to notebook \
    --execute "$nb" \
    --ExecutePreprocessor.timeout=-1 \
    --output "$(basename "$out")" \
    --output-dir notebooks
}

NOTEBOOKS=(
  "notebooks/NB01_Data_Exploration.ipynb"
  "notebooks/NB02_Chroma_GraphRAG.ipynb"
  "notebooks/NB03_Pinecone_GraphRAG.ipynb"
  "notebooks/NB04_Agentic_GraphRAG.ipynb"
  "notebooks/NB05_Evaluation.ipynb"
  "notebooks/NB06_Hybrid_RAG.ipynb"
  "notebooks/NB07_CRAG.ipynb"
  "notebooks/NB08_Multimodal_RAG.ipynb"
  "notebooks/NB09_Multimodal_RAG_OCR_CLI.ipynb"
  "notebooks/NB10_Multimodal_RAG_Vision_Qwen.ipynb"
  "notebooks/NB11_Selective_Finetuning_Unsloth_PEFT_TRL.ipynb"
)

for nb in "${NOTEBOOKS[@]}"; do
  run_nb "$nb"
done

pytest

python - <<'PY'
from src.config import settings
from src.pinecone_retriever import delete_index

index_name = f"{settings.pinecone_index_prefix}-section-b"
try:
    delete_index(index_name)
    print(f"Deleted Pinecone index: {index_name}")
except Exception as exc:
    print(f"Pinecone cleanup warning: {exc}")
PY

echo "Full real pipeline complete."
