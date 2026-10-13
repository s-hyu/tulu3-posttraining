#!/usr/bin/env bash
# Run exactly the 115 aliases in the completed Base OLMES baseline, in four-task shards.
set -u

ROOT=/home/vipuser/tulu3
MODEL="${MODEL:-$ROOT/results/qwen25_3b_sft_bf16_taskbook_1epoch_seed42}"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
OUT_ROOT="${OUT_ROOT:-$ROOT/eval_results/qwen25_3b_sft_taskbook_olmes_full}"
TASK_DIR="$OUT_ROOT/task_lists"
OUT_DIR="$OUT_ROOT/results"
LOG_DIR="${LOG_DIR:-$ROOT/logs/olmes_sft_full}"
BASE_METRICS="$ROOT/eval_results/qwen25_3b_base_olmes_microshards/metrics-all.jsonl"
MAX_PARALLEL="${MAX_PARALLEL:-2}"

export PATH="$ROOT/envs/eval/bin:$PATH"
export HF_HOME="$ROOT/cache/olmes_hf"
export HF_DATASETS_CACHE="$ROOT/cache/olmes_hf/datasets"
export HF_HUB_CACHE="$ROOT/cache/olmes_hf/hub"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TORCHINDUCTOR_COMPILE_THREADS=1
export TOKENIZERS_PARALLELISM=true

[[ -f "$MODEL/model.safetensors" ]] || { echo "Missing final SFT model: $MODEL" >&2; exit 1; }
[[ -f "$BASE_METRICS" ]] || { echo "Missing Base metrics: $BASE_METRICS" >&2; exit 1; }
[[ "$MAX_PARALLEL" =~ ^[1-9][0-9]*$ ]] || { echo "MAX_PARALLEL must be a positive integer" >&2; exit 1; }
mkdir -p "$TASK_DIR" "$OUT_DIR" "$LOG_DIR"
cat > "$OUT_ROOT/evaluation_manifest.txt" <<EOF
model=$MODEL
tokenizer=$TOKENIZER
baseline_metrics=$BASE_METRICS
task_count=115
model_type=hf
dtype=float16
batch_size=2
max_length=8192
mmlu_num_shots=5
fewshot_seed=1234
max_parallel_processes=$MAX_PARALLEL
EOF

if ! compgen -G "$TASK_DIR/chunk_*.txt" > /dev/null; then
    BASE_METRICS="$BASE_METRICS" TASK_DIR="$TASK_DIR" "$ROOT/envs/eval/bin/python" - <<'PY'
import json
import os
from pathlib import Path

rows = [json.loads(line) for line in Path(os.environ["BASE_METRICS"]).read_text().splitlines() if line.strip()]
aliases = sorted(row["task_config"]["metadata"]["alias"] for row in rows)
assert len(aliases) == 115 and len(set(aliases)) == 115
task_dir = Path(os.environ["TASK_DIR"])
for start in range(0, len(aliases), 4):
    (task_dir / f"chunk_{start // 4:03d}.txt").write_text("\n".join(aliases[start:start + 4]) + "\n")
PY
fi

run_chunk() {
    local task_file="$1"
    local chunk="$(basename "$task_file" .txt)"
    local chunk_out="$OUT_DIR/$chunk"
    local chunk_log="$LOG_DIR/$chunk.log"
    local status_file="$LOG_DIR/$chunk.status"
    local tasks=()
    mapfile -t tasks < "$task_file"
    if [[ -f "$chunk_out/metrics-all.jsonl" ]] && [[ "$(wc -l < "$chunk_out/metrics-all.jsonl")" -eq "${#tasks[@]}" ]]; then
        printf 'already_complete\n' > "$status_file"
        return 0
    fi
    mkdir -p "$chunk_out"
    printf 'running %s (%s tasks)\n' "$chunk" "${#tasks[@]}" > "$status_file"
    setsid olmes --model "$MODEL" --model-type hf --task "${tasks[@]}" \
        --output-dir "$chunk_out" --batch-size 2 --gpus 1 \
        --model-args "{\"dtype\":\"float16\",\"max_length\":8192,\"tokenizer\":\"$TOKENIZER\"}" \
        > "$chunk_log" 2>&1 &
    local child=$!
    while kill -0 "$child" 2>/dev/null; do
        if [[ -f "$chunk_out/metrics-all.jsonl" ]] && [[ "$(wc -l < "$chunk_out/metrics-all.jsonl")" -eq "${#tasks[@]}" ]]; then
            # OLMES can hang after final metrics are safely written. Stop that isolated process group.
            kill -TERM -- "-$child" 2>/dev/null || true
            wait "$child" || true
            printf 'exit=0 (metrics_complete)\n' > "$status_file"
            return 0
        fi
        sleep 2
    done
    wait "$child"
    local rc=$?
    printf 'exit=%s\n' "$rc" > "$status_file"
    return "$rc"
}

active_jobs() { jobs -rp | wc -l; }
started=0
for task_file in "$TASK_DIR"/chunk_*.txt; do
    while [[ "$(active_jobs)" -ge "$MAX_PARALLEL" ]]; do wait -n || true; done
    # Match the Base run: avoid four model loads starting together and exhausting host RAM.
    if [[ "$started" -gt 0 ]] && [[ "$(active_jobs)" -lt 4 ]]; then sleep 15; fi
    run_chunk "$task_file" &
    started=$((started + 1))
done
wait

"$ROOT/envs/eval/bin/python" "$ROOT/scripts/summarize_olmes_compare.py" \
    --baseline "$BASE_METRICS" --result-root "$OUT_ROOT" --task-dir "$TASK_DIR"
