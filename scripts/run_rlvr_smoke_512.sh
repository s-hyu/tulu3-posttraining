#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/vipuser/tulu3
PYTHON=${ROOT}/envs/train/bin/python
MODEL=${ROOT}/results/qwen25_3b_sft_bf16_seed42/model
DATA=${ROOT}/data/tulu3/rlvr/rlvr_smoke_512_seed42.parquet
OUT=${ROOT}/results/qwen25_3b_sft_rlvr_smoke
STATE=${ROOT}/checkpoints/qwen25_3b_sft_rlvr_smoke
CACHE=${ROOT}/data/cache/rlvr_smoke_512

cd ${ROOT}/open-instruct
export PYTHONPATH=${ROOT}/open-instruct
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export WANDB_PROJECT=tulu3-nju-posttraining
# vLLM 0.19.1 needs pickle fallback for a local Ray worker to pass torch.dtype.
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
mkdir -p "${OUT}" "${CACHE}" "${ROOT}/logs" "${ROOT}/checkpoints"
"${PYTHON}" "${ROOT}/scripts/prepare_rlvr_smoke_subset.py"

exec "${PYTHON}" open_instruct/grpo_fast.py \
  --exp_name qwen25_3b_sft_rlvr_smoke \
  --seed 42 \
  --model_name_or_path "${MODEL}" \
  --tokenizer_name_or_path "${MODEL}" \
  --use_peft true \
  --dataset_mixer_list "${DATA}" 512 \
  --dataset_mixer_list_splits train \
  --dataset_transform_fn rlvr_tokenize_v1 rlvr_max_length_filter_v1 \
  --dataset_local_cache_dir "${CACHE}" \
  --max_prompt_token_length 512 \
  --response_length 256 \
  --pack_length 768 \
  --max_seq_length 768 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --num_unique_prompts_rollout 4 \
  --num_samples_per_prompt_rollout 4 \
  --temperature 0.7 \
  --apply_verifiable_reward true \
  --verification_reward 1.0 \
  --apply_r1_style_format_reward false \
  --ground_truths_key ground_truth \
  --learning_rate 3e-7 \
  --warmup_ratio 0.05 \
  --num_epochs 1 \
  --total_episodes 208 \
  --beta 0.02 \
  --load_ref_policy true \
  --deepspeed_stage 2 \
  --num_learners_per_node 1 \
  --vllm_tensor_parallel_size 1 \
  --vllm_gpu_memory_utilization 0.10 \
  --vllm_sync_backend gloo \
  --vllm_enforce_eager \
  --gradient_checkpointing \
  --local_eval_every -1 \
  --output_dir "${OUT}" \
  --checkpoint_state_dir "${STATE}" \
  --checkpoint_state_freq 4 \
  --save_freq 4 \
  --with_tracking \
  --wandb_project tulu3-nju-posttraining \
  --push_to_hub false \
  --single_gpu_mode \
  "$@"
