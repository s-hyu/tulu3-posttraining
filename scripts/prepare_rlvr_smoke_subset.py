#!/usr/bin/env python3
"""Create a deterministic, verifier-compatible RLVR smoke subset."""

from pathlib import Path

from datasets import load_dataset


ROOT = Path("/home/vipuser/tulu3")
SOURCE = ROOT / "data/tulu3/rlvr/RLVR-GSM-MATH-IF-Mixed-Constraints/data/train-00000-of-00001.parquet"
OUTPUT = ROOT / "data/tulu3/rlvr/rlvr_smoke_512_seed42.parquet"
SEED = 42
SIZE = 512


def normalize_verifier(row: dict) -> dict:
    # This dataset's IFEval ground truth is the legacy JSON constraint schema.
    # Open-Instruct exposes its matching verifier as `ifeval_old`.
    if row["dataset"].lower() == "ifeval":
        row["dataset"] = "ifeval_old"
    return row


def main() -> None:
    ds = load_dataset("parquet", data_files=str(SOURCE), split="train")
    subset = ds.shuffle(seed=SEED).select(range(SIZE)).map(normalize_verifier)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    subset.to_parquet(str(OUTPUT))
    print(f"wrote {len(subset)} records to {OUTPUT}")
    print(subset.to_pandas()["dataset"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
