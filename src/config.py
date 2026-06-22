"""Project-wide typed configuration for the Medical Research GraphRAG system."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProjectSettings(BaseSettings):
    """Runtime settings loaded from environment variables and sensible defaults."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Models
    embedding_model: str = Field(default="qwen3-embedding:4b")
    generator_model: str = Field(default="granite4.1:8b")
    judge_model: str = Field(default="granite4.1:8b")
    # Backward-compat alias retained for older notebook references.
    guardian_judge_model: str = Field(default="granite4.1:8b")
    multimodal_ocr_model: str = Field(default="glm-ocr")
    multimodal_vision_model: str = Field(default="qwen3.5:4b")
    ocr_cli_allow_fallback: bool = Field(default=True)
    ocr_cli_timeout_seconds: int = Field(default=120)
    # Optional fine-tuning stack (Unsloth + PEFT + TRL), used only in NB11.
    finetune_base_model_hf: str = Field(default="unsloth/llama-3.2-3b-instruct")
    finetune_adapter_name: str = Field(default="medresearch-lora")
    finetune_max_train_examples: int = Field(default=2500)
    finetune_max_eval_examples: int = Field(default=300)
    finetune_max_seq_length: int = Field(default=2048)
    finetune_lora_rank: int = Field(default=16)
    finetune_lora_alpha: int = Field(default=32)
    finetune_lora_dropout: float = Field(default=0.0)
    finetune_train_batch_size: int = Field(default=2)
    finetune_grad_accumulation: int = Field(default=8)
    finetune_learning_rate: float = Field(default=2e-4)
    finetune_max_steps: int = Field(default=120)
    finetune_warmup_steps: int = Field(default=10)
    finetune_export_gguf: bool = Field(default=False)

    # Data controls
    random_seed: int = Field(default=42)
    max_records: int = Field(default=5000)
    chunk_size: int = Field(default=2000)
    chunk_overlap: int = Field(default=200)

    # Retrieval controls
    top_k_retrieval: int = Field(default=8)
    local_graph_hops: int = Field(default=2)
    retrieval_grade_threshold: float = Field(default=0.45)
    hallucination_threshold: float = Field(default=0.70)
    hybrid_dense_weight: float = Field(default=0.6)
    hybrid_sparse_weight: float = Field(default=0.4)
    crag_max_corrections: int = Field(default=2)
    crag_acceptance_threshold: float = Field(default=0.55)

    # Evaluation controls
    eval_query_count: int = Field(default=100)
    generation_eval_count: int = Field(default=40)

    # Pinecone controls
    pinecone_api_key: str | None = Field(default=None)
    pinecone_cloud: str = Field(default="aws")
    pinecone_region: str = Field(default="us-east-1")
    pinecone_index_prefix: str = Field(default="medmentions-graphrag")

    # Paths
    project_root: Path = Field(default=Path(__file__).resolve().parent.parent)

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def graph_dir(self) -> Path:
        return self.project_root / "graphs"

    @property
    def chroma_dir(self) -> Path:
        return self.project_root / "chroma_db"

    @property
    def outputs_dir(self) -> Path:
        return self.project_root / "outputs"

    @property
    def figures_dir(self) -> Path:
        return self.outputs_dir / "figures"

    @property
    def tables_dir(self) -> Path:
        return self.outputs_dir / "tables"

    @property
    def metrics_dir(self) -> Path:
        return self.outputs_dir / "metrics"

    @property
    def eval_dir(self) -> Path:
        return self.project_root / "evaluations"

    @property
    def multimodal_dir(self) -> Path:
        return self.data_dir / "multimodal"

    @property
    def finetune_dir(self) -> Path:
        return self.project_root / "outputs" / "finetune"

    @property
    def finetune_dataset_dir(self) -> Path:
        return self.finetune_dir / "datasets"

    @property
    def finetune_adapter_dir(self) -> Path:
        return self.finetune_dir / "adapters"

    @property
    def finetune_reports_dir(self) -> Path:
        return self.finetune_dir / "reports"

    def ensure_dirs(self) -> None:
        """Create all required project directories if they do not already exist."""
        for path in [
            self.data_dir,
            self.processed_dir,
            self.graph_dir,
            self.chroma_dir,
            self.outputs_dir,
            self.figures_dir,
            self.tables_dir,
            self.metrics_dir,
            self.eval_dir,
            self.multimodal_dir,
            self.finetune_dir,
            self.finetune_dataset_dir,
            self.finetune_adapter_dir,
            self.finetune_reports_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = ProjectSettings()
settings.ensure_dirs()
