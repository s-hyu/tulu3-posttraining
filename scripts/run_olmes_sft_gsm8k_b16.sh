#!/usr/bin/env bash
# Match the successful Base GSM8K run: batch 16, local Qwen tokenizer, 512-token task cap.
set -u

ROOT=/home/vipuser/tulu3
MODEL="$ROOT/results/qwen25_3b_sft_bf16_seed42/model"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
OUT_DIR="$ROOT/eval_results/qwen25_3b_sft_olmes_full/gsm8k_b16"
LOG_DIR="$ROOT/logs/olmes_sft_step8000_gsm8k_b16"

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
setsid olmes --model "$MODEL" --model-type hf --task gsm8k::olmes \
    --output-dir "$OUT_DIR" --batch-size 16 --gpus 1 \
    --model-args "{\"dtype\":\"float16\",\"max_length\":8192,\"tokenizer\":\"$TOKENIZER\"}" \
    > "$LOG_DIR/gsm8k.log" 2>&1 &
child=$!
printf 'running pid=%s\n' "$child" > "$LOG_DIR/status"
while kill -0 "$child" 2>/dev/null; do
    if [[ -f "$OUT_DIR/metrics-all.jsonl" ]] && [[ "$(wc -l < "$OUT_DIR/metrics-all.jsonl")" -eq 1 ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        wait "$child" || true
        printf 'exit=0 (metrics_complete)\n' > "$LOG_DIR/status"
        exit 0
    fi
    sleep 2
done
wait "$child"
rc=$?
printf 'exit=%s\n' "$rc" > "$LOG_DIR/status"
exit "$rc"
