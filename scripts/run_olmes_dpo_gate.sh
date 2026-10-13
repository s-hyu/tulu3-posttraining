#!/usr/bin/env bash
# Fixed local OLMES comparison for the completed 10k-pair DPO model.
set -euo pipefail

ROOT=/home/vipuser/tulu3
MODEL="$ROOT/results/qwen25_3b_dpo_10k_seed42"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
TASK_DIR="$ROOT/eval_plans/sft_olmes_gate_tasks"
OUT_ROOT="$ROOT/eval_results/qwen25_3b_dpo_10k_seed42_olmes_gate"
OUT_DIR="$OUT_ROOT/results"
LOG_DIR="$ROOT/logs/olmes_dpo_gate"
BASE_METRICS="$ROOT/eval_results/qwen25_3b_base_olmes_microshards/metrics-all.jsonl"

export PATH="$ROOT/envs/eval/bin:$PATH"
export HF_HOME="$ROOT/cache/olmes_hf"
export HF_DATASETS_CACHE="$ROOT/cache/olmes_hf/datasets"
export HF_HUB_CACHE="$ROOT/cache/olmes_hf/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=true

[[ -f "$MODEL/model.safetensors" ]] || { echo "Missing final DPO model: $MODEL/model.safetensors" >&2; exit 1; }
[[ -f "$BASE_METRICS" ]] || { echo "Missing Base metrics: $BASE_METRICS" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"

run_chunk() {
    local task_file="$1" chunk chunk_out chunk_log status_file
    chunk="$(basename "$task_file" .txt)"
    chunk_out="$OUT_DIR/$chunk"
    chunk_log="$LOG_DIR/$chunk.log"
    status_file="$LOG_DIR/$chunk.status"
    local tasks=()
    mapfile -t tasks < "$task_file"
    mkdir -p "$chunk_out"
    printf 'running %s (%s tasks)\n' "$chunk" "${#tasks[@]}" > "$status_file"
    setsid olmes --model "$MODEL" --model-type hf --task "${tasks[@]}" \
        --output-dir "$chunk_out" --batch-size 2 --gpus 1 \
        --model-args "{\"dtype\":\"float16\",\"max_length\":8192,\"tokenizer\":\"$TOKENIZER\"}" \
        > "$chunk_log" 2>&1 &
    local child=$!
    while kill -0 "$child" 2>/dev/null; do
        if [[ -f "$chunk_out/metrics-all.jsonl" ]] && [[ "$(wc -l < "$chunk_out/metrics-all.jsonl")" -eq "${#tasks[@]}" ]]; then
            kill -TERM -- "-$child" 2>/dev/null || true
            wait "$child" || true
            printf 'exit=0 (metrics_complete)\n' > "$status_file"
            return 0
        fi
        sleep 2
    done
    wait "$child"
    printf 'exit=%s\n' "$?" > "$status_file"
}

for task_file in "$TASK_DIR"/chunk_*.txt; do run_chunk "$task_file" & done
wait
"$ROOT/envs/eval/bin/python" "$ROOT/scripts/summarize_olmes_compare.py" \
    --baseline "$BASE_METRICS" --result-root "$OUT_ROOT" --task-dir "$TASK_DIR" --gate
