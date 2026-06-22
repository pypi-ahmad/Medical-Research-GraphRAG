#!/usr/bin/env python3
"""Fetch real PMC open-access multimodal assets for NB08-NB10."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.multimodal_assets_pmc import fetch_pmc_multimodal_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-images", type=int, default=5, help="Maximum number of figure assets.")
    parser.add_argument("--max-tables", type=int, default=3, help="Maximum number of table assets.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = fetch_pmc_multimodal_assets(
        max_images=args.max_images,
        max_tables=args.max_tables,
    )
    print(f"Manifest: {result['manifest_path']}")
    print(f"Assets: {len(result['assets'])}")
    print(f"Failures: {len(result['failures'])}")


if __name__ == "__main__":
    main()
