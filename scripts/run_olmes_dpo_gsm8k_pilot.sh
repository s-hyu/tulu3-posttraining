#!/usr/bin/env bash
# Same 132-item GSM8K subset and protocol as the saved Base/SFT pilot results.
set -euo pipefail

ROOT=/home/vipuser/tulu3
MODEL="$ROOT/results/qwen25_3b_dpo_10k_seed42"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
OUT_DIR="$ROOT/eval_results/qwen25_3b_dpo_10k_seed42_gsm8k_pilot_seed1234"
LOG_DIR="$ROOT/logs/olmes_dpo_gsm8k_pilot_seed1234"

export PATH="$ROOT/envs/eval/bin:$PATH"
export HF_HOME="$ROOT/cache/olmes_hf"
export HF_DATASETS_CACHE="$ROOT/cache/olmes_hf/datasets"
export HF_HUB_CACHE="$ROOT/cache/olmes_hf/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=true

[[ -f "$MODEL/model.safetensors" ]] || { echo "Missing final DPO model: $MODEL/model.safetensors" >&2; exit 1; }
mkdir -p "$OUT_DIR" "$LOG_DIR"
setsid olmes --model "$MODEL" --model-type hf --task gsm8k::olmes \
    --limit 0.1 --random-subsample-seed 1234 \
    --output-dir "$OUT_DIR" --batch-size 16 --gpus 1 \
    --model-args "{\"dtype\":\"float16\",\"max_length\":8192,\"tokenizer\":\"$TOKENIZER\"}" \
    > "$LOG_DIR/gsm8k.log" 2>&1 &
child=$!
while kill -0 "$child" 2>/dev/null; do
    if [[ -f "$OUT_DIR/metrics-all.jsonl" ]] && [[ "$(wc -l < "$OUT_DIR/metrics-all.jsonl")" -eq 1 ]]; then
        kill -TERM -- "-$child" 2>/dev/null || true
        wait "$child" || true
        exit 0
    fi
    sleep 2
done
wait "$child"
