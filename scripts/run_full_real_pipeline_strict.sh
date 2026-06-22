#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

LOG_DIR="$ROOT_DIR/outputs/logs"
STATE_DIR="$ROOT_DIR/outputs/run_state"
mkdir -p "$LOG_DIR" "$STATE_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_LOG="$LOG_DIR/full_real_pipeline_${STAMP}.log"
STATE_FILE="$STATE_DIR/full_real_pipeline.state"

RESUME=true
FORCE=false
MAX_RETRIES="${MAX_RETRIES:-2}"
NOTEBOOK_TIMEOUT="${NOTEBOOK_TIMEOUT:--1}"

if [[ "${1:-}" == "--force" ]]; then
  FORCE=true
  RESUME=false
fi

exec > >(tee -a "$RUN_LOG") 2>&1

log() {
  printf "[%s] %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

set_state() {
  local key="$1"
  local value="$2"
  if [[ -f "$STATE_FILE" ]] && grep -q "^${key}=" "$STATE_FILE"; then
    sed -i "s|^${key}=.*|${key}=${value}|g" "$STATE_FILE"
  else
    printf "%s=%s\n" "$key" "$value" >>"$STATE_FILE"
  fi
}

get_state() {
  local key="$1"
  if [[ ! -f "$STATE_FILE" ]]; then
    echo ""
    return
  fi
  awk -F= -v k="$key" '$1==k {print $2}' "$STATE_FILE" | tail -n 1
}

run_once_with_retry() {
  local step_key="$1"
  local cmd="$2"
  local attempts=0

  if [[ "$FORCE" != "true" && "$RESUME" == "true" ]] && [[ "$(get_state "$step_key")" == "done" ]]; then
    log "Skipping already-completed step: $step_key"
    return 0
  fi

  until (( attempts > MAX_RETRIES )); do
    attempts=$((attempts + 1))
    log "Running step [$step_key], attempt ${attempts}/$((MAX_RETRIES + 1))"
    if eval "$cmd"; then
      set_state "$step_key" "done"
      log "Step succeeded: $step_key"
      return 0
    fi
    log "Step failed: $step_key"
    if (( attempts > MAX_RETRIES )); then
      log "Step exceeded retry budget: $step_key"
      return 1
    fi
    sleep 5
  done
}

ensure_venv() {
  if [[ ! -d .venv ]]; then
    log "Creating .venv with uv Python 3.12.10"
    uv python install 3.12.10
    uv venv --python 3.12.10 .venv
  fi
}

ensure_ollama_service() {
  if ollama list >/dev/null 2>&1; then
    log "Ollama service is already reachable."
    return
  fi

  log "Starting Ollama service..."
  nohup ollama serve >>"$LOG_DIR/ollama_serve_${STAMP}.log" 2>&1 &
  sleep 5

  local tries=0
  until ollama list >/dev/null 2>&1; do
    tries=$((tries + 1))
    if (( tries > 12 )); then
      log "Ollama service did not become ready in time."
      return 1
    fi
    sleep 2
  done
  log "Ollama service is ready."
}

ensure_model() {
  local model="$1"
  if ollama list | awk 'NR>1 {print $1}' | grep -Fx "$model" >/dev/null 2>&1; then
    log "Model present: $model"
    return 0
  fi
  log "Pulling missing model: $model"
  ollama pull "$model"
}

run_notebook() {
  local nb="$1"
  local stem
  stem="$(basename "$nb" .ipynb)"
  local out="notebooks/${stem}.executed.ipynb"
  local key="nb_${stem}"
  local nb_log="$LOG_DIR/${stem}_${STAMP}.log"

  if [[ "$FORCE" != "true" && "$RESUME" == "true" ]] && [[ "$(get_state "$key")" == "done" ]]; then
    log "Skipping notebook (state=done): $nb"
    return 0
  fi

  if [[ "$FORCE" != "true" && "$RESUME" == "true" ]] && [[ -f "$out" ]] && [[ "$out" -nt "$nb" ]]; then
    log "Notebook output is already newer than source, marking done: $nb"
    set_state "$key" "done"
    return 0
  fi

  local attempts=0
  until (( attempts > MAX_RETRIES )); do
    attempts=$((attempts + 1))
    log "Executing notebook [$nb], attempt ${attempts}/$((MAX_RETRIES + 1))"
    if jupyter nbconvert \
      --to notebook \
      --execute "$nb" \
      --ExecutePreprocessor.timeout="$NOTEBOOK_TIMEOUT" \
      --output "$(basename "$out")" \
      --output-dir notebooks \
      >"$nb_log" 2>&1; then
      set_state "$key" "done"
      log "Notebook succeeded: $nb"
      return 0
    fi
    log "Notebook failed: $nb (see $nb_log)"
    if (( attempts > MAX_RETRIES )); then
      return 1
    fi
    sleep 6
  done
}

main() {
  log "Strict full pipeline run started."
  log "Run log: $RUN_LOG"
  log "State file: $STATE_FILE"

  ensure_venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
  uv sync --extra dev --extra finetune

  export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
  export RUN_FULL_EVAL="${RUN_FULL_EVAL:-true}"
  export RUN_FULL_TRAIN="${RUN_FULL_TRAIN:-true}"
  export CLEANUP_PINECONE_INDEX="${CLEANUP_PINECONE_INDEX:-false}"

  if [[ -z "${PINECONE_API_KEY:-}" ]]; then
    log "PINECONE_API_KEY is required for Pinecone notebook execution."
    return 1
  fi

  run_once_with_retry "preflight_ollama" "ensure_ollama_service"
  run_once_with_retry "model_qwen3_embedding_4b" "ensure_model 'qwen3-embedding:4b'"
  run_once_with_retry "model_granite4_1_8b" "ensure_model 'granite4.1:8b'"
  run_once_with_retry "model_glm_ocr" "ensure_model 'glm-ocr'"
  run_once_with_retry "model_qwen3_5_4b" "ensure_model 'qwen3.5:4b'"
  run_once_with_retry "fetch_pmc_assets" "python scripts/fetch_pmc_multimodal_assets.py --max-images 5 --max-tables 3"

  local notebooks=(
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

  for nb in "${notebooks[@]}"; do
    run_notebook "$nb"
  done

  run_once_with_retry "pytest_q" "pytest -q"

  if [[ "$CLEANUP_PINECONE_INDEX" == "true" ]]; then
    run_once_with_retry "pinecone_cleanup" "python - <<'PY'
from src.config import settings
from src.pinecone_retriever import delete_index

index_name = f\"{settings.pinecone_index_prefix}-section-b\"
try:
    delete_index(index_name)
    print(f\"Deleted Pinecone index: {index_name}\")
except Exception as exc:
    print(f\"Pinecone cleanup warning: {exc}\")
PY"
  else
    log "Skipping Pinecone index cleanup (CLEANUP_PINECONE_INDEX=false)."
  fi

  log "Strict full pipeline run completed successfully."
}

main "$@"
