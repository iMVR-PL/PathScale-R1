#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../.env"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sft

cd "${PROJECT_ROOT}/LLaMA-Factory"

llamafactory-cli export examples/merge_lora/cross_scale_vqa_lora_sft.yaml