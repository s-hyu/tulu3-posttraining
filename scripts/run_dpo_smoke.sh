#!/usr/bin/env bash
# Ten-pair DPO validation: data transform, reference cache, loss, update, checkpoint, and reload artifact.
set -euo pipefail

ROOT=/home/vipuser/tulu3
TORCHRUN="$ROOT/envs/train/bin/torchrun"
MODEL="$ROOT/results/qwen25_3b_sft_bf16_seed42/model"
TOKENIZER="$ROOT/models/Qwen2.5-3B"
DATA="$ROOT/data/derived/tulu3_dpo_10_seed42.parquet"
OUT="$ROOT/results/qwen25_3b_dpo_smoke_10_seed42_torchrun"
LOG="$ROOT/logs/qwen25_3b_dpo_smoke_10_seed42_torchrun.log"

export PYTHONPATH="$ROOT/open-instruct${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/cache/train_hf"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export REFERENCE_LOGPROBS_CACHE_PATH="$ROOT/cache/dpo_reference_logprobs"
export WANDB_PROJECT=tulu3-nju-posttraining
export ACCELERATE_MIXED_PRECISION=bf16
export TOKENIZERS_PARALLELISM=false
mkdir -p "$REFERENCE_LOGPROBS_CACHE_PATH" "$OUT"

exec "$TORCHRUN" --standalone --nproc_per_node=1 "$ROOT/open-instruct/open_instruct/dpo_tune_cache.py" \
  --exp_name qwen25_3b_dpo_smoke_10_seed42_torchrun \
  --do_not_randomize_output_dir true \
  --model_name_or_path "$MODEL" \
  --tokenizer_name_or_path "$TOKENIZER" \
  --mixer_list "$DATA" 1.0 \
  --mixer_list_splits train \
  --cache_mode local --local_cache_dir "$ROOT/cache/dpo_dataset_cache_smoke_torchrun" \
  --max_seq_length 2048 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 4 \
  --learning_rate 5e-7 --lr_scheduler_type linear --warmup_ratio 0.1 \
  --weight_decay 0.0 --max_grad_norm 1.0 \
  --num_epochs 1 --loss_type dpo_norm --beta 5 \
  --checkpointing_steps 1 --keep_last_n_checkpoints 1 \
  --output_dir "$OUT" --logging_steps 1 \
  --seed 42 --with_tracking --wandb_project "$WANDB_PROJECT" \
  --push_to_hub false --try_launch_beaker_eval_jobs false --try_auto_save_to_beaker false \
  2>&1 | tee "$LOG"
