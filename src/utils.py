"""Shared utilities for reproducibility, logging, timing, and serialization."""

from __future__ import annotations

import json
import os
import random
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Iterator

import numpy as np
from loguru import logger

from src.config import settings


def configure_logging() -> None:
    """Configure a concise, deterministic log format for notebooks and scripts."""
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="INFO",
    )


def set_global_seed(seed: int) -> None:
    """Set seeds across Python and NumPy to improve reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


@contextmanager
def timed_block(name: str) -> Iterator[None]:
    """Context manager for timing notebook sections and pipeline stages."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logger.info(f"{name} completed in {elapsed:.2f}s")


def timer(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Function decorator version of the timing helper."""

    @wraps(fn)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{fn.__name__} completed in {elapsed:.2f}s")
        return result

    return wrapped


def save_json(data: Any, path: Path) -> None:
    """Write data to JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=True)


def load_json(path: Path) -> Any:
    """Load JSON from disk."""
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dataframe_csv(df: Any, path: Path) -> None:
    """Persist a pandas or polars dataframe to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_plot(path: Path) -> None:
    """Persist current matplotlib figure and ensure directory exists."""
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")


def slugify(text: str) -> str:
    """Create a filesystem-safe identifier for notebook outputs and IDs."""
    cleaned = "".join(ch.lower() if ch.isalnum() else "-" for ch in text)
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")


configure_logging()
set_global_seed(settings.random_seed)
