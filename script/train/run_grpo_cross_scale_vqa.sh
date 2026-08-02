#!/usr/bin/env bash
set -xeuo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/.env"

export WANDB_MODE=offline
export WANDB_DIR=$WANDB_DIR
export HYDRA_FULL_ERROR=1

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

INFER_BACKEND=${INFER_BACKEND:-vllm}

PROJECT_NAME=${PROJECT_NAME:-"cross_scale_mcq"}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"grpo_pathor1"}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
NNODES=${NNODES:-1}

GEN_TP=${GEN_TP:-1}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.7}

ACTOR_MODEL_PATH=${ACTOR_MODEL_PATH:-"$ACTOR_MODEL_DIR"}

CKPTS_DIR=${CKPTS_DIR:-"$RESULTS_DIR/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
ROLLOUT_DIR=${ROLLOUT_DIR:-"$RESULTS_DIR/rollouts/${PROJECT_NAME}/${EXPERIMENT_NAME}"}
VAL_DIR=${VAL_DIR:-"$RESULTS_DIR/val-generation/${PROJECT_NAME}/${EXPERIMENT_NAME}"}

TRAIN_FILE=${TRAIN_FILE:-"$PROCESSED_DIR/rl_parquet/train.parquet"}
TEST_FILE=${TEST_FILE:-"$PROCESSED_DIR/rl_parquet/val.parquet"}
REWARD_FILE=${REWARD_FILE:-"$REPO_ROOT/verl/verl/utils/reward_score/cross_scale_vqa.py"}
REWARD_FN=${REWARD_FN:-compute_score}
LOG_FILE=${LOG_FILE:-"$LOG_DIR/grpo_cross_scale_vqa.log"}

########################### parameter arrays ###########################
DATA=(
    algorithm.adv_estimator=grpo
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=32
    data.max_prompt_length=5120
    data.max_response_length=4096
    data.filter_overlong_prompts=True
)

MODEL=(
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_activation_offload=True
)

ACTOR=(
    actor_rollout_ref.actor.ppo_mini_batch_size=32
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.model_dtype=bf16
)

REF=(
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.model_dtype=bf16
)

ROLLOUT=(
    actor_rollout_ref.rollout.name="${INFER_BACKEND}"
    actor_rollout_ref.rollout.n=5
    +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_mm_preprocessor_cache=true
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEM_UTIL}"
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}"
    actor_rollout_ref.rollout.dtype=bfloat16
)

TRAINER=(
    trainer.nnodes="${NNODES}"
    trainer.n_gpus_per_node="${NDEVICES_PER_NODE}"
    trainer.logger='["console","wandb"]'
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.rollout_data_dir="${ROLLOUT_DIR}"
    trainer.validation_data_dir="${VAL_DIR}"
    trainer.resume_mode=auto
    trainer.total_epochs=5
    trainer.save_freq=20
    trainer.test_freq=5
)

REWARD=(
    custom_reward_function.path="${REWARD_FILE}"
    custom_reward_function.name="${REWARD_FN}"
)

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "$@" 2>&1 | tee "${LOG_FILE}"