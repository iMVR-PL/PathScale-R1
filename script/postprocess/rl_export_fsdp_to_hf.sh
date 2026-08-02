#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/../_env.sh"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir $RAW_CKPT_DIR/global_step_500/actor \
    --target_dir $OUR_MODEL_DIR


