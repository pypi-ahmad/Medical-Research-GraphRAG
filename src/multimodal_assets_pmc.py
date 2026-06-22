"""Open biomedical multimodal asset acquisition for NB08-NB10.

Despite the historical module name, this implementation intentionally fetches
real open biomedical/health datasets and generates reproducible charts/tables
for multimodal RAG execution without anti-bot scraping failures.
"""

from __future__ import annotations

import csv
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import requests
from loguru import logger

from src.config import settings
from src.utils import save_json


@dataclass(slots=True)
class BiomedicalChartTarget:
    """One chart target built from an open biomedical dataset."""

    dataset_url: str
    metric_id: str
    metric_label: str
    entity: str


@dataclass(slots=True)
class BiomedicalTableTarget:
    """One table target built from an open biomedical dataset."""

    dataset_url: str
    metric_id: str
    metric_label: str
    top_n: int = 20


@dataclass(slots=True)
class PMCMultimodalAsset:
    """Persisted multimodal asset metadata for evaluation provenance."""

    asset_id: str
    modality: str
    local_path: str
    source_url: str
    pmcid: str
    title: str
    caption: str
    question: str
    reference_answer: str
    license: str
    retrieved_at_unix: float


DEFAULT_CHART_TARGETS: list[BiomedicalChartTarget] = [
    BiomedicalChartTarget(
        dataset_url="https://ourworldindata.org/grapher/diabetes-prevalence.csv",
        metric_id="diabetes_prevalence",
        metric_label="Diabetes prevalence (% of adults 20-79)",
        entity="United States",
    ),
    BiomedicalChartTarget(
        dataset_url="https://ourworldindata.org/grapher/diabetes-prevalence.csv",
        metric_id="diabetes_prevalence",
        metric_label="Diabetes prevalence (% of adults 20-79)",
        entity="India",
    ),
    BiomedicalChartTarget(
        dataset_url="https://ourworldindata.org/grapher/share-of-adults-defined-as-obese.csv",
        metric_id="adult_obesity",
        metric_label="Adult obesity prevalence (%)",
        entity="United States",
    ),
    BiomedicalChartTarget(
        dataset_url="https://ourworldindata.org/grapher/hospital-beds-per-1000-people.csv",
        metric_id="hospital_beds",
        metric_label="Hospital beds per 1,000 people",
        entity="United States",
    ),
    BiomedicalChartTarget(
        dataset_url="https://ourworldindata.org/grapher/life-expectancy.csv",
        metric_id="life_expectancy",
        metric_label="Life expectancy (years)",
        entity="United States",
    ),
]


DEFAULT_TABLE_TARGETS: list[BiomedicalTableTarget] = [
    BiomedicalTableTarget(
        dataset_url="https://ourworldindata.org/grapher/diabetes-prevalence.csv",
        metric_id="diabetes_prevalence",
        metric_label="Diabetes prevalence (% of adults 20-79)",
        top_n=20,
    ),
    BiomedicalTableTarget(
        dataset_url="https://ourworldindata.org/grapher/share-of-adults-defined-as-obese.csv",
        metric_id="adult_obesity",
        metric_label="Adult obesity prevalence (%)",
        top_n=20,
    ),
    BiomedicalTableTarget(
        dataset_url="https://ourworldindata.org/grapher/hospital-beds-per-1000-people.csv",
        metric_id="hospital_beds",
        metric_label="Hospital beds per 1,000 people",
        top_n=20,
    ),
]


def _safe_stem(value: str) -> str:
    stem = "".join(ch if ch.isalnum() else "-" for ch in value.lower())
    while "--" in stem:
        stem = stem.replace("--", "-")
    return stem.strip("-")


def _request_dataframe(url: str, timeout: int = 45) -> pd.DataFrame:
    """Fetch a CSV dataframe with deterministic headers."""
    headers = {
        "User-Agent": "Medical-Research-GraphRAG/1.0 (open biomedical asset collection)",
    }
    response = requests.get(url, timeout=timeout, headers=headers)
    response.raise_for_status()
    return pd.read_csv(pd.io.common.StringIO(response.text))


def _value_column(df: pd.DataFrame) -> str:
    """Return primary numeric metric column from OWID-like dataframe."""
    blocklist = {
        "Entity",
        "Code",
        "Year",
        "World region according to OWID",
    }
    candidates = [c for c in df.columns if c not in blocklist]
    for col in candidates:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col

    for col in candidates:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() > 0:
            df[col] = coerced
            return col

    raise ValueError("Could not infer numeric metric column")


def _fmt(value: float) -> str:
    return f"{value:.2f}"


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    """Save dataframe to CSV with deterministic encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([str(col) for col in df.columns.tolist()])
        for _, row in df.iterrows():
            writer.writerow([row[col] for col in df.columns.tolist()])


def _first_row_reference(df: pd.DataFrame) -> str:
    """Construct compact table reference answer from first row."""
    if df.empty:
        return ""
    row = df.iloc[0]
    cols = list(df.columns[: min(6, len(df.columns))])
    parts = [f"{col}={row[col]}" for col in cols]
    return "; ".join(parts)


def _build_chart_asset(
    target: BiomedicalChartTarget,
    image_dir: Path,
) -> PMCMultimodalAsset:
    """Create one chart image from a real biomedical dataset series."""
    df = _request_dataframe(target.dataset_url)
    value_col = _value_column(df)

    if "Entity" not in df.columns or "Year" not in df.columns:
        raise ValueError("Dataset must include Entity and Year columns")

    subset = df[df["Entity"] == target.entity].copy()
    subset[value_col] = pd.to_numeric(subset[value_col], errors="coerce")
    subset = subset.dropna(subset=["Year", value_col]).sort_values("Year")
    if subset.empty:
        raise ValueError(f"No rows found for entity={target.entity}")

    x = subset["Year"].astype(int).to_numpy()
    y = subset[value_col].astype(float).to_numpy()
    start_year, end_year = int(x[0]), int(x[-1])
    start_val, end_val = float(y[0]), float(y[-1])
    trend_word = "increased" if end_val >= start_val else "decreased"

    filename = f"owid_{_safe_stem(target.metric_id)}_{_safe_stem(target.entity)}.png"
    local_path = image_dir / filename

    plt.figure(figsize=(10, 5))
    plt.plot(x, y, linewidth=2.0, marker="o", markersize=3)
    plt.title(f"{target.metric_label} - {target.entity}")
    plt.xlabel("Year")
    plt.ylabel(target.metric_label)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(local_path, dpi=160)
    plt.close()

    caption = (
        f"{target.metric_label} for {target.entity} {trend_word} from "
        f"{_fmt(start_val)} in {start_year} to {_fmt(end_val)} in {end_year}."
    )
    question = (
        f"What trend does the chart show for {target.metric_label.lower()} "
        f"in {target.entity}?"
    )

    return PMCMultimodalAsset(
        asset_id=f"owidfig_{_safe_stem(target.metric_id)}_{_safe_stem(target.entity)}",
        modality="image",
        local_path=str(local_path),
        source_url=target.dataset_url,
        pmcid=f"OWID:{target.metric_id}",
        title=f"{target.metric_label} - {target.entity}",
        caption=caption,
        question=question,
        reference_answer=caption,
        license="OWID open data (CC BY where applicable)",
        retrieved_at_unix=time.time(),
    )


def _build_table_asset(
    target: BiomedicalTableTarget,
    table_dir: Path,
) -> PMCMultimodalAsset:
    """Create one table CSV from latest values of a real biomedical dataset."""
    df = _request_dataframe(target.dataset_url)
    value_col = _value_column(df)

    if "Entity" not in df.columns or "Year" not in df.columns:
        raise ValueError("Dataset must include Entity and Year columns")

    work = df[["Entity", "Code", "Year", value_col]].copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=["Entity", "Year", value_col])

    latest = work.sort_values(["Entity", "Year"]).groupby("Entity", as_index=False).tail(1)
    latest = latest.sort_values(value_col, ascending=False).head(max(1, target.top_n)).reset_index(drop=True)
    latest.rename(columns={value_col: target.metric_label}, inplace=True)

    filename = f"owid_{_safe_stem(target.metric_id)}_latest_top{target.top_n}.csv"
    local_path = table_dir / filename
    _save_csv(latest, local_path)

    caption = (
        f"Top {target.top_n} entities by latest {target.metric_label} based on OWID dataset."
    )
    question = f"Which entities have the highest latest values for {target.metric_label.lower()}?"
    reference = _first_row_reference(latest) or caption

    return PMCMultimodalAsset(
        asset_id=f"owidtbl_{_safe_stem(target.metric_id)}_top{target.top_n}",
        modality="table",
        local_path=str(local_path),
        source_url=target.dataset_url,
        pmcid=f"OWID:{target.metric_id}",
        title=f"Top {target.top_n} {target.metric_label}",
        caption=caption,
        question=question,
        reference_answer=reference,
        license="OWID open data (CC BY where applicable)",
        retrieved_at_unix=time.time(),
    )


def fetch_pmc_multimodal_assets(
    *,
    max_images: int = 5,
    max_tables: int = 3,
    image_targets: list[BiomedicalChartTarget] | None = None,
    table_targets: list[BiomedicalTableTarget] | None = None,
) -> dict[str, Any]:
    """Fetch real multimodal biomedical assets and persist a manifest.

    The output contract is intentionally unchanged so NB08-NB10 can consume
    `data/multimodal/pmc_asset_manifest.json` without additional wiring.
    """
    image_targets = image_targets or DEFAULT_CHART_TARGETS
    table_targets = table_targets or DEFAULT_TABLE_TARGETS

    image_dir = settings.multimodal_dir / "images"
    table_dir = settings.multimodal_dir / "tables"
    image_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    assets: list[PMCMultimodalAsset] = []
    failures: list[dict[str, str]] = []

    for target in image_targets[:max_images]:
        try:
            assets.append(_build_chart_asset(target, image_dir=image_dir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to build chart asset {} {}: {}", target.metric_id, target.entity, exc)
            failures.append(
                {
                    "modality": "image",
                    "target": f"{target.metric_id}:{target.entity}",
                    "error": str(exc),
                }
            )

    for target in table_targets[:max_tables]:
        try:
            assets.append(_build_table_asset(target, table_dir=table_dir))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Failed to build table asset {}: {}", target.metric_id, exc)
            failures.append(
                {
                    "modality": "table",
                    "target": f"{target.metric_id}:top{target.top_n}",
                    "error": str(exc),
                }
            )

    manifest = {
        "source_policy": "Real open biomedical datasets (OWID). No synthetic records.",
        "created_at_unix": time.time(),
        "assets": [asdict(asset) for asset in assets],
        "failures": failures,
    }
    manifest_path = settings.multimodal_dir / "pmc_asset_manifest.json"
    save_json(manifest, manifest_path)
    logger.info(
        "Saved multimodal manifest: {} assets, {} failures -> {}",
        len(assets),
        len(failures),
        manifest_path,
    )
    return {"manifest_path": manifest_path, "assets": assets, "failures": failures}
