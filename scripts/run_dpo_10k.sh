#!/usr/bin/env bash
# Approved DPO run: 10,000 fixed-seed preference pairs, one epoch, full-parameter BF16.
set -euo pipefail

ROOT=/home/vipuser/tulu3
TORCHRUN="$ROOT/envs/train/bin/torchrun"
MODEL="$ROOT/results/qwen25_3b_sft_bf16_seed42/model"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
DATA="$ROOT/data/derived/tulu3_dpo_10k_seed42.parquet"
OUT="$ROOT/results/qwen25_3b_dpo_10k_seed42"
LOG="$ROOT/logs/qwen25_3b_dpo_10k_seed42_retry_mb1_ga8.log"

export PYTHONPATH="$ROOT/open-instruct${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/cache/train_hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REFERENCE_LOGPROBS_CACHE_PATH="$ROOT/cache/dpo_reference_logprobs"
export WANDB_PROJECT=tulu3-nju-posttraining
export ACCELERATE_MIXED_PRECISION=bf16
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
mkdir -p "$REFERENCE_LOGPROBS_CACHE_PATH" "$OUT"

exec "$TORCHRUN" --standalone --nproc_per_node=1 "$ROOT/open-instruct/open_instruct/dpo_tune_cache.py" \
  --exp_name qwen25_3b_dpo_10k_seed42 \
  --do_not_randomize_output_dir true \
  --model_name_or_path "$MODEL" \
  --tokenizer_name_or_path "$TOKENIZER" \
  --mixer_list "$DATA" 1.0 \
  --mixer_list_splits train \
  --cache_mode local --local_cache_dir "$ROOT/cache/dpo_dataset_cache" \
  --max_seq_length 2048 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 8 \
  --learning_rate 5e-7 --lr_scheduler_type linear --warmup_ratio 0.1 \
  --weight_decay 0.0 --max_grad_norm 1.0 \
  --num_epochs 1 --loss_type dpo_norm --beta 5 \
  --checkpointing_steps 250 --keep_last_n_checkpoints 2 \
  --output_dir "$OUT" --logging_steps 10 \
  --seed 42 --with_tracking --wandb_project "$WANDB_PROJECT" \
  --push_to_hub false --try_launch_beaker_eval_jobs false --try_auto_save_to_beaker false \
  2>&1 | tee "$LOG"
