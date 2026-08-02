#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../.env"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sft

export WANDB_MODE=offline
export WANDB_DIR=$WANDB_DIR
export WANDB_DATA_DIR="$WANDB_DIR/wandb_data"
export WANDB_PROJECT=triplet_new_mcq_batch2_sft

cd "${PROJECT_ROOT}/LLaMA-Factory"
CUDA_VISIBLE_DEVICES=0 \
llamafactory-cli train examples/train_lora/cross_scale_vqa_lora_sft.yaml \
2>&1 | tee "$LOG_DIR/sft_lora_cross_scale_vqa.log"
