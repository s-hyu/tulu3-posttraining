#!/usr/bin/env python3
"""Create a deterministic local SFT subset from the downloaded Tulu-3 shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    source_files = sorted(args.source_dir.glob("*.parquet"))
    if not source_files:
        raise FileNotFoundError(f"No Parquet files found in {args.source_dir}")

    dataset = load_dataset("parquet", data_files=[str(path) for path in source_files], split="train")
    if args.samples > len(dataset):
        raise ValueError(f"Requested {args.samples} samples from a dataset of {len(dataset)} rows")

    subset = dataset.shuffle(seed=args.seed).select(range(args.samples))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(str(args.output))

    metadata = {
        "purpose": "single-GPU BF16 full-parameter SFT pilot",
        "sampling_method": "datasets.Dataset.shuffle(seed).select(range(samples))",
        "seed": args.seed,
        "requested_samples": args.samples,
        "source_rows": len(dataset),
        "source_files": [
            {"name": path.name, "sha256": sha256(path)} for path in source_files
        ],
        "output_file": args.output.name,
        "output_rows": len(subset),
        "columns": subset.column_names,
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(subset)} rows to {args.output}")
    print(f"Wrote metadata to {metadata_path}")


if __name__ == "__main__":
    main()
