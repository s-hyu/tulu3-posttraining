#!/usr/bin/env bash

set -u

ROOT=/home/vipuser/tulu3
TASK_DIR="$ROOT/eval_results/qwen25_3b_base_olmes_microshards/task_lists"
OUT_DIR="$ROOT/eval_results/qwen25_3b_base_olmes_microshards/results"
LOG_DIR="$ROOT/logs/olmes_microshards"

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

mkdir -p "$OUT_DIR" "$LOG_DIR"

run_chunk() {
    local task_file="$1"
    local chunk
    chunk="$(basename "$task_file" .txt)"
    local chunk_out="$OUT_DIR/$chunk"
    local chunk_log="$LOG_DIR/$chunk.log"
    local status_file="$LOG_DIR/$chunk.status"
    local tasks=()

    mapfile -t tasks < "$task_file"
    mkdir -p "$chunk_out"
    printf 'running %s (%s tasks)\n' "$chunk" "${#tasks[@]}" > "$status_file"

    olmes \
        --model "$ROOT/models/Qwen2.5-3B" \
        --model-type hf \
        --task "${tasks[@]}" \
        --output-dir "$chunk_out" \
        --batch-size 2 \
        --gpus 1 \
        --model-args '{"dtype":"float16","max_length":8192}' \
        > "$chunk_log" 2>&1
    local rc=$?
    printf 'exit=%s\n' "$rc" > "$status_file"
    return "$rc"
}

active_jobs() {
    jobs -rp | wc -l
}

started=0
for task_file in "$TASK_DIR"/chunk_*.txt; do
    while [ "$(active_jobs)" -ge 4 ]; do
        wait -n || true
    done

    if [ "$started" -gt 0 ] && [ "$(active_jobs)" -lt 4 ]; then
        sleep 15
    fi
    run_chunk "$task_file" &
    started=$((started + 1))
done

wait
