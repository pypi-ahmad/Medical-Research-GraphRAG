"""Optional Unsloth + PEFT + TRL training helpers for NB11.

These utilities are intentionally optional and are not used by the base
GraphRAG / Agentic RAG pipeline unless NB11 is explicitly executed.
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from loguru import logger

from src.config import settings
from src.utils import save_json


def finetune_stack_status() -> dict[str, bool]:
    """Return package availability for the optional fine-tuning stack."""
    packages = ["unsloth", "peft", "trl", "datasets", "transformers", "accelerate"]
    return {name: importlib.util.find_spec(name) is not None for name in packages}


def require_finetune_stack(require_unsloth: bool = True) -> None:
    """Raise a clear error if optional fine-tuning deps are missing."""
    status = finetune_stack_status()
    required = ["peft", "trl", "datasets", "transformers", "accelerate"]
    if require_unsloth:
        required.append("unsloth")
    missing = [name for name in required if not status.get(name, False)]
    if missing:
        joined = ", ".join(sorted(missing))
        raise RuntimeError(
            "Optional fine-tuning dependencies are missing: "
            f"{joined}. Install with `uv sync --extra finetune` before running NB11 training."
        )


@dataclass(slots=True)
class LoRAHyperParams:
    """LoRA adapter configuration used with PEFT via Unsloth."""

    r: int = settings.finetune_lora_rank
    lora_alpha: int = settings.finetune_lora_alpha
    lora_dropout: float = settings.finetune_lora_dropout
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


@dataclass(slots=True)
class SFTTrainConfig:
    """Training hyperparameters for TRL SFTTrainer."""

    base_model: str = settings.finetune_base_model_hf
    max_seq_length: int = settings.finetune_max_seq_length
    load_in_4bit: bool = True
    train_batch_size: int = settings.finetune_train_batch_size
    gradient_accumulation_steps: int = settings.finetune_grad_accumulation
    learning_rate: float = settings.finetune_learning_rate
    max_steps: int = settings.finetune_max_steps
    warmup_steps: int = settings.finetune_warmup_steps
    logging_steps: int = 5
    save_steps: int = 25
    eval_steps: int = 25
    seed: int = settings.random_seed


def _resolve_target_modules_for_model(model: Any, configured: tuple[str, ...]) -> list[str]:
    """Resolve LoRA target modules present in the loaded model."""
    available = {name.split(".")[-1] for name, _ in model.named_modules()}

    direct = [name for name in configured if name in available]
    if direct:
        return direct

    fallbacks = [
        "query_key_value",
        "c_attn",
        "c_proj",
        "fc1",
        "fc2",
        "Wqkv",
        "out_proj",
        "proj",
    ]
    resolved = [name for name in fallbacks if name in available]
    if resolved:
        return resolved

    # Last-resort heuristic: pick module names containing "proj" or "attn".
    heuristic: list[str] = []
    for name, _ in model.named_modules():
        leaf = name.split(".")[-1]
        if leaf in heuristic:
            continue
        if "proj" in leaf or "attn" in leaf:
            heuristic.append(leaf)
    if heuristic:
        return heuristic[:12]

    return list(configured)


def create_unsloth_lora_model(
    lora: LoRAHyperParams | None = None,
    cfg: SFTTrainConfig | None = None,
) -> tuple[Any, Any]:
    """Initialize a base model and attach LoRA adapters with Unsloth."""
    require_finetune_stack(require_unsloth=True)
    lora = lora or LoRAHyperParams()
    cfg = cfg or SFTTrainConfig()

    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg.base_model,
        max_seq_length=cfg.max_seq_length,
        dtype=None,
        load_in_4bit=cfg.load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora.r,
        target_modules=list(lora.target_modules),
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=cfg.seed,
    )
    return model, tokenizer


def create_peft_lora_model_fallback(
    lora: LoRAHyperParams | None = None,
    cfg: SFTTrainConfig | None = None,
) -> tuple[Any, Any]:
    """Fallback LoRA model creation using Transformers + PEFT.

    This is used when the Unsloth path is unavailable at runtime while the
    user still requests a real NB11 training run.
    """
    require_finetune_stack(require_unsloth=False)
    lora = lora or LoRAHyperParams()
    cfg = cfg or SFTTrainConfig()

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=dtype,
    )
    target_modules = _resolve_target_modules_for_model(model, lora.target_modules)
    logger.info("Using LoRA target modules: {}", target_modules)

    peft_cfg = LoraConfig(
        r=lora.r,
        lora_alpha=lora.lora_alpha,
        lora_dropout=lora.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    return model, tokenizer


def load_sft_dataset_dict(train_jsonl: Path, eval_jsonl: Path) -> Any:
    """Load JSONL train/eval rows as a DatasetDict."""
    require_finetune_stack(require_unsloth=False)
    from datasets import load_dataset

    dataset = load_dataset(
        "json",
        data_files={
            "train": str(train_jsonl),
            "eval": str(eval_jsonl),
        },
    )
    return dataset


def create_sft_trainer(
    *,
    model: Any,
    tokenizer: Any,
    dataset: Any,
    cfg: SFTTrainConfig | None = None,
    output_dir: Path | None = None,
) -> Any:
    """Create a TRL SFTTrainer instance with PEFT-backed LoRA model."""
    require_finetune_stack(require_unsloth=False)
    cfg = cfg or SFTTrainConfig()
    output = output_dir or (settings.finetune_adapter_dir / settings.finetune_adapter_name)
    output.mkdir(parents=True, exist_ok=True)

    from trl import SFTConfig, SFTTrainer

    base_kwargs = {
        "output_dir": str(output),
        "per_device_train_batch_size": cfg.train_batch_size,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "learning_rate": cfg.learning_rate,
        "max_steps": cfg.max_steps,
        "warmup_steps": cfg.warmup_steps,
        "logging_steps": cfg.logging_steps,
        "save_steps": cfg.save_steps,
        "eval_steps": cfg.eval_steps,
        "seed": cfg.seed,
    }
    rich_kwargs = {
        **base_kwargs,
        "evaluation_strategy": "steps",
        "dataset_text_field": "text",
        "max_seq_length": cfg.max_seq_length,
        "packing": False,
    }

    try:
        args = SFTConfig(**rich_kwargs)
    except TypeError:
        # Handle versions that use `eval_strategy` instead of `evaluation_strategy`.
        alt_kwargs = {**rich_kwargs}
        alt_kwargs.pop("evaluation_strategy", None)
        alt_kwargs["eval_strategy"] = "steps"
        try:
            args = SFTConfig(**alt_kwargs)
        except TypeError:
            # Last-resort compatibility: keep only TrainingArguments-level fields.
            args = SFTConfig(**base_kwargs)

    # TRL versions differ in whether they expect tokenizer or processing_class.
    try:
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=dataset["eval"],
            args=args,
            dataset_text_field="text",
            max_seq_length=cfg.max_seq_length,
            packing=False,
        )
    except TypeError:
        try:
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                args=args,
                dataset_text_field="text",
                max_seq_length=cfg.max_seq_length,
                packing=False,
            )
        except TypeError:
            trainer = SFTTrainer(
                model=model,
                processing_class=tokenizer,
                train_dataset=dataset["train"],
                eval_dataset=dataset["eval"],
                args=args,
            )
    return trainer


def save_adapter_bundle(
    *,
    model: Any,
    tokenizer: Any,
    adapter_dir: Path | None = None,
    train_cfg: SFTTrainConfig | None = None,
    lora_cfg: LoRAHyperParams | None = None,
) -> dict[str, Any]:
    """Save adapter artifacts and metadata for later Ollama import."""
    adapter_path = adapter_dir or (settings.finetune_adapter_dir / settings.finetune_adapter_name)
    adapter_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    meta = {
        "adapter_dir": str(adapter_path),
        "base_model": (train_cfg or SFTTrainConfig()).base_model,
        "lora": asdict(lora_cfg or LoRAHyperParams()),
        "training": asdict(train_cfg or SFTTrainConfig()),
    }
    save_json(meta, adapter_path / "adapter_metadata.json")
    logger.info("Saved adapter bundle to {}", adapter_path)
    return meta


def write_ollama_modelfile_template(
    *,
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    system_prompt: str | None = None,
) -> Path:
    """Write an Ollama Modelfile template referencing a fine-tuned adapter."""
    sys_prompt = system_prompt or (
        "You are a biomedical research assistant. Always ground claims in cited evidence."
    )
    content = "\n".join(
        [
            f"FROM {base_model}",
            f"ADAPTER {adapter_path}",
            "PARAMETER temperature 0.2",
            'SYSTEM """',
            sys_prompt,
            '"""',
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")
    return output_path


def persist_finetune_placeholder_report(
    *,
    stack: dict[str, bool],
    train_examples: int,
    eval_examples: int,
    out_path: Path | None = None,
) -> Path:
    """Persist schema-complete placeholder report for implementation-only phase."""
    target = out_path or (settings.finetune_reports_dir / "nb11_finetune_report.json")
    payload = {
        "mode": "placeholder",
        "stack_status": stack,
        "dataset": {
            "train_examples": train_examples,
            "eval_examples": eval_examples,
        },
        "training_metrics": {
            "loss": None,
            "eval_loss": None,
            "steps": None,
            "tokens_per_second": None,
            "placeholder_note": "Populate by enabling NB11 training and running the notebook.",
        },
        "post_run_quality": {
            "retrieval_metrics_delta": None,
            "generation_metrics_delta": None,
            "rag_metrics_delta": None,
            "judge_metrics_delta": None,
            "latency_delta_ms": None,
            "placeholder_note": "Populate after explicit end-to-end execution and evaluation.",
        },
    }
    save_json(payload, target)
    return target


def append_jsonl_metrics(metrics_path: Path, row: dict[str, Any]) -> None:
    """Append one JSON row to a line-delimited metrics file."""
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=True))
        handle.write("\n")
