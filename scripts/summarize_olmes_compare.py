#!/usr/bin/env python3
"""Merge OLMES shard metrics and compare exact task aliases to a baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def alias(row: dict) -> str:
    return row["task_config"]["metadata"]["alias"]


def read_expected(task_dir: Path) -> list[str]:
    values: list[str] = []
    for task_file in sorted(task_dir.glob("*.txt")):
        values.extend(
            line.strip() for line in task_file.read_text().splitlines() if line.strip() and not line.startswith("#")
        )
    return values


def load_candidate(result_root: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for metric_file in sorted((result_root / "results").glob("**/metrics-all.jsonl")):
        for row in read_jsonl(metric_file):
            rows[alias(row)] = row
    return rows


def score(rows: dict[str, dict], aliases: list[str]) -> float | None:
    values = [rows[item]["metrics"]["primary_score"] for item in aliases if item in rows]
    return fmean(values) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--gate", action="store_true")
    args = parser.parse_args()

    expected = read_expected(args.task_dir)
    if len(expected) != len(set(expected)):
        raise SystemExit("Task list contains duplicate aliases")

    base = {alias(row): row for row in read_jsonl(args.baseline)}
    candidate = load_candidate(args.result_root)
    missing_base = [item for item in expected if item not in base]
    missing_candidate = [item for item in expected if item not in candidate]
    if missing_base:
        raise SystemExit(f"Missing baseline aliases: {missing_base}")
    if missing_candidate:
        raise SystemExit(f"Incomplete result: {len(missing_candidate)} aliases missing: {missing_candidate}")

    ordered_candidate = [candidate[item] for item in expected]
    output_jsonl = args.result_root / "metrics-all.jsonl"
    output_jsonl.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in ordered_candidate))

    groups = {
        "MMLU MC (acc_raw)": [item for item in expected if ":mc::olmes" in item],
        "MMLU RC (acc_per_char)": [item for item in expected if ":rc::olmes" in item],
        "GSM8K (exact_match)": [item for item in expected if item == "gsm8k::olmes"],
        "All task primary scores": expected,
    }
    summaries: list[dict] = []
    for name, items in groups.items():
        if not items:
            continue
        base_score = score(base, items)
        candidate_score = score(candidate, items)
        assert base_score is not None and candidate_score is not None
        summaries.append(
            {
                "group": name,
                "tasks": len(items),
                "baseline": base_score,
                "candidate": candidate_score,
                "delta": candidate_score - base_score,
                "delta_percentage_points": (candidate_score - base_score) * 100,
            }
        )

    decision = "complete"
    if args.gate:
        mc = next(item for item in summaries if item["group"] == "MMLU MC (acc_raw)")
        rc = next(item for item in summaries if item["group"] == "MMLU RC (acc_per_char)")
        all_mmlu = next(item for item in summaries if item["group"] == "All task primary scores")
        decision = (
            "stop_recommended"
            if all_mmlu["delta"] <= -0.02 and max(mc["delta"], rc["delta"]) <= 0
            else "continue_to_full_olmes"
        )

    result = {
        "expected_tasks": len(expected),
        "completed_tasks": len(ordered_candidate),
        "decision": decision,
        "groups": summaries,
    }
    (args.result_root / "comparison.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    (args.result_root / "metrics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")

    lines = [
        "# OLMES SFT vs Base comparison",
        "",
        f"- Tasks completed: {len(ordered_candidate)}/{len(expected)}",
        f"- Decision: **{decision}**",
        "",
        "| Metric group | Tasks | Base | SFT | Delta (pp) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {item['group']} | {item['tasks']} | {item['baseline']:.4f} | "
        f"{item['candidate']:.4f} | {item['delta_percentage_points']:+.2f} |"
        for item in summaries
    )
    if args.gate:
        lines += [
            "",
            "Gate rule: stop only if the MMLU gate macro score falls by at least 2.00 pp "
            "and neither its MC nor RC macro improves.",
        ]
    (args.result_root / "COMPARISON.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
