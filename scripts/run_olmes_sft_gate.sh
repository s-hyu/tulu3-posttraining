#!/usr/bin/env bash
# Run the fixed MMLU gate before allocating time to the 115-task OLMES run.
set -u

ROOT=/home/vipuser/tulu3
MODEL="$ROOT/results/qwen25_3b_sft_bf16_taskbook_1epoch_seed42"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
TASK_DIR="$ROOT/eval_plans/sft_olmes_gate_tasks"
OUT_ROOT="$ROOT/eval_results/qwen25_3b_sft_taskbook_olmes_gate"
OUT_DIR="$OUT_ROOT/results"
LOG_DIR="$ROOT/logs/olmes_sft_gate"
BASE_METRICS="$ROOT/eval_results/qwen25_3b_base_olmes_microshards/metrics-all.jsonl"

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
mkdir -p "$OUT_DIR" "$LOG_DIR"

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

for task_file in "$TASK_DIR"/chunk_*.txt; do
    run_chunk "$task_file" &
done
wait

"$ROOT/envs/eval/bin/python" "$ROOT/scripts/summarize_olmes_compare.py" \
    --baseline "$BASE_METRICS" --result-root "$OUT_ROOT" --task-dir "$TASK_DIR" --gate
