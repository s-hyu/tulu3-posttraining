#!/usr/bin/env python3
"""Create a reproducible, row-level random DPO subset from local Parquet shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    files = sorted(args.source_dir.rglob("*.parquet"))
    if not files:
        raise SystemExit(f"No Parquet files found in {args.source_dir}")
    file_rows = [pq.ParquetFile(path).metadata.num_rows for path in files]
    total = sum(file_rows)
    if not 0 < args.count <= total:
        raise SystemExit(f"--count must be in [1, {total}], got {args.count}")

    selected = np.sort(np.random.default_rng(args.seed).choice(total, size=args.count, replace=False))
    tables: list[pa.Table] = []
    offset = 0
    for path, rows in zip(files, file_rows, strict=True):
        positions = selected[(selected >= offset) & (selected < offset + rows)] - offset
        if len(positions):
            parquet = pq.ParquetFile(path)
            row_offset = 0
            for group in range(parquet.num_row_groups):
                group_rows = parquet.metadata.row_group(group).num_rows
                local = positions[(positions >= row_offset) & (positions < row_offset + group_rows)] - row_offset
                if len(local):
                    tables.append(parquet.read_row_group(group).take(pa.array(local, type=pa.int64())))
                row_offset += group_rows
        offset += rows

    subset = pa.concat_tables(tables)
    # Guard against accidental order/count changes before writing a training input.
    if subset.num_rows != args.count:
        raise RuntimeError(f"Expected {args.count} rows, materialized {subset.num_rows}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(subset, args.output, compression="zstd")
    manifest = {
        "source_dir": str(args.source_dir.resolve()),
        "source_files": [str(path.resolve()) for path in files],
        "source_rows": total,
        "selected_rows": args.count,
        "seed": args.seed,
        "columns": subset.column_names,
    }
    args.output.with_suffix(args.output.suffix + ".metadata.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
