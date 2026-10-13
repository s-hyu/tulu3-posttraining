#!/usr/bin/env bash
# Retry the only MMLU shard interrupted by exit 137, without changing evaluation settings.
set -u

ROOT=/home/vipuser/tulu3
MODEL="$ROOT/results/qwen25_3b_sft_bf16_seed42/model"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
OUT_DIR="$ROOT/eval_results/qwen25_3b_sft_olmes_full/results/chunk_020"
LOG_DIR="$ROOT/logs/olmes_sft_step8000_retry_chunk020"
TASKS=(mmlu_marketing:rc::olmes mmlu_medical_genetics:mc::olmes mmlu_medical_genetics:rc::olmes mmlu_miscellaneous:mc::olmes)

export PATH="$ROOT/envs/eval/bin:$PATH"
export HF_HOME="$ROOT/cache/olmes_hf" HF_DATASETS_CACHE="$ROOT/cache/olmes_hf/datasets" HF_HUB_CACHE="$ROOT/cache/olmes_hf/hub"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LITELLM_LOCAL_MODEL_COST_MAP=True TORCHINDUCTOR_COMPILE_THREADS=1 TOKENIZERS_PARALLELISM=true

mkdir -p "$OUT_DIR" "$LOG_DIR"
setsid olmes --model "$MODEL" --model-type hf --task "${TASKS[@]}" --output-dir "$OUT_DIR" --batch-size 2 --gpus 1 \
  --model-args "{\"dtype\":\"float16\",\"max_length\":8192,\"tokenizer\":\"$TOKENIZER\"}" > "$LOG_DIR/chunk_020.log" 2>&1 &
child=$!
printf 'running pid=%s\n' "$child" > "$LOG_DIR/status"
while kill -0 "$child" 2>/dev/null; do
  if [[ -f "$OUT_DIR/metrics-all.jsonl" ]] && [[ "$(wc -l < "$OUT_DIR/metrics-all.jsonl")" -eq 4 ]]; then
    kill -TERM -- "-$child" 2>/dev/null || true
    wait "$child" || true
    printf 'exit=0 (metrics_complete)\n' > "$LOG_DIR/status"
    exit 0
  fi
  sleep 2
done
wait "$child"; rc=$?
printf 'exit=%s\n' "$rc" > "$LOG_DIR/status"
exit "$rc"
