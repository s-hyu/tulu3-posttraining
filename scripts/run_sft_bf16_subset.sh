#!/usr/bin/env bash
# Run persistently, for example:
#   setsid nohup bash /home/vipuser/tulu3/scripts/run_sft_bf16_subset.sh \
#     </dev/null > /home/vipuser/tulu3/logs/qwen25_3b_sft_bf16_taskbook_1epoch_launcher.log 2>&1 &
set -euo pipefail

PROJECT_ROOT="/home/vipuser/tulu3"
TRAIN_PYTHON="${PROJECT_ROOT}/envs/train/bin/python"
MODEL_DIR="${PROJECT_ROOT}/models/Qwen2.5-3B"
DATASET_FILE="${PROJECT_ROOT}/data/derived/tulu3_sft_subset_seed42.parquet"

# Agreed deviations from the task-book recipe:
#   1. Use this deterministic local subset instead of the full mixture.
#   2. Use one A100: microbatch 4 x accumulation 8 = global batch 32,
#      matching 4 GPUs x microbatch 1 x accumulation 8 in the task book.
#   3. Save a recovery checkpoint every 500 optimizer steps and retain the latest
#      one. This is an operational safeguard approved by the user.
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/results/qwen25_3b_sft_bf16_taskbook_1epoch_seed42}"
TRAIN_BATCH_SIZE=4
GRADIENT_ACCUMULATION_STEPS=8
NUM_TRAIN_EPOCHS=1
EFFECTIVE_GLOBAL_BATCH=$((TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
CHECKPOINTING_STEPS=500
KEEP_LAST_N_CHECKPOINTS=1
TOKENIZED_CACHE_DIR="${PROJECT_ROOT}/cache/tokenized/sft_subset_seed42_l4096_taskbook"
LOG_FILE="${OUTPUT_DIR}/training.log"

[[ -x "${TRAIN_PYTHON}" ]] || { echo "Missing training Python: ${TRAIN_PYTHON}" >&2; exit 1; }
[[ -f "${MODEL_DIR}/config.json" ]] || { echo "Missing local model: ${MODEL_DIR}" >&2; exit 1; }
[[ -f "${DATASET_FILE}" ]] || { echo "Missing local dataset: ${DATASET_FILE}" >&2; exit 1; }
[[ "${EFFECTIVE_GLOBAL_BATCH}" -eq 32 ]] || { echo "Expected global batch 32" >&2; exit 1; }

if [[ -e "${OUTPUT_DIR}" ]] && [[ -n "$(find "${OUTPUT_DIR}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "Refusing to reuse a non-empty output directory: ${OUTPUT_DIR}" >&2
  exit 1
fi
mkdir -p "${OUTPUT_DIR}" "${TOKENIZED_CACHE_DIR}"

export PYTHONPATH="${PROJECT_ROOT}/open-instruct${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${OUTPUT_DIR}/wandb"

# The current Open-Instruct revision auto-detects attention backends; its old
# task-book flag --use_flash_attn is no longer accepted. Require the equivalent
# Flash Attention 2 backend instead of silently falling back to PyTorch SDPA.
"${TRAIN_PYTHON}" - <<'PY'
from open_instruct.model_utils import detect_hf_attn_implementation
backend = detect_hf_attn_implementation()
assert backend == "flash_attention_2", f"Expected Flash Attention 2, got {backend}"
print(f"Attention backend: {backend}")
PY

cat > "${OUTPUT_DIR}/run_manifest.txt" <<EOF
model=${MODEL_DIR}
dataset=${DATASET_FILE}
seed=42
precision=bf16
attention=flash_attention_2 (auto-detected by current Open-Instruct)
max_seq_length=4096
per_device_train_batch_size=${TRAIN_BATCH_SIZE}
gradient_accumulation_steps=${GRADIENT_ACCUMULATION_STEPS}
world_size=1
effective_global_batch=${EFFECTIVE_GLOBAL_BATCH}
learning_rate=2e-5
lr_scheduler_type=linear
warmup_ratio=0.03
weight_decay=0.0
num_train_epochs=${NUM_TRAIN_EPOCHS}
gradient_checkpointing=true
checkpointing_steps=${CHECKPOINTING_STEPS}
keep_last_n_checkpoints=${KEEP_LAST_N_CHECKPOINTS}
EOF

echo "=== SFT training configuration ==="
cat "${OUTPUT_DIR}/run_manifest.txt"
echo "output_dir=${OUTPUT_DIR}"

"${TRAIN_PYTHON}" -m accelerate.commands.launch \
  --mixed_precision bf16 \
  --num_processes 1 \
  "${PROJECT_ROOT}/open-instruct/open_instruct/finetune.py" \
  --model_name_or_path "${MODEL_DIR}" \
  --tokenizer_name "${MODEL_DIR}" \
  --dataset_mixer_list "${DATASET_FILE}" 1.0 \
  --dataset_mixer_list_splits train \
  --dataset_cache_mode local \
  --dataset_local_cache_dir "${TOKENIZED_CACHE_DIR}" \
  --max_seq_length 4096 \
  --preprocessing_num_workers 16 \
  --per_device_train_batch_size "${TRAIN_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
  --learning_rate 2e-5 \
  --lr_scheduler_type linear \
  --warmup_ratio 0.03 \
  --weight_decay 0.0 \
  --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
  --gradient_checkpointing True \
  --logging_steps 10 \
  --checkpointing_steps "${CHECKPOINTING_STEPS}" \
  --keep_last_n_checkpoints "${KEEP_LAST_N_CHECKPOINTS}" \
  --clean_checkpoints_at_end False \
  --seed 42 \
  --output_dir "${OUTPUT_DIR}" \
  --do_not_randomize_output_dir True \
  --with_tracking True \
  --report_to wandb \
  --wandb_project_name tulu3-nju-posttraining \
  --exp_name qwen25-3b-sft-taskbook-1epoch-seed42 \
  --push_to_hub False \
  --try_launch_beaker_eval_jobs False \
  --try_auto_save_to_beaker False \
  2>&1 | tee "${LOG_FILE}"
