#!/usr/bin/env bash

#SBATCH --job-name=robotwin10_lava_v5_actionsim_final_1h100nvl
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h100-nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err
#SBATCH --open-mode=append

# Training parameters
EPOCHS=4
TASK_SET="10"                    # must match Stage 1
LEARNING_RATE="4e-5"
LR_MIN="5e-5"
BATCH_SIZE=128
GRAD_ACCUM_STEPS=1
SEED=42
SAVE_INTERVAL_EPOCH=2
STAGE1_FINAL_EPOCH=12
STAGE1_CHECKPOINT=""             # empty: find epoch 12 under the same RUN_NAME
RESUME_CHECKPOINT=""             # empty: auto-resume Stage 2 before stage switching

# LAVA parameters must match Stage 1. The initialization warmup is already done.
LAMBDA_LAVA="0.01"
LAVA_SCALES="[1,2,4,8,16]"
LAVA_TEMPERATURE="0.07"
LAVA_SAMPLE_RATIO="0.125"
LAVA_SAMPLING_BALANCE="task_episode"
LAVA_NEGATIVE_MODE="mixed"
LAVA_NEGATIVE_WINDOW_MULTIPLIER=4
LAVA_NEGATIVE_WINDOW_MAX=32
LAVA_WARMUP_RATIO="0.0"
LAVA_SCALE_SAMPLING="batch_uniform"
LAVA_ORDER_NEGATIVE="true"      # contiguous block swap + full derangement
LAVA_LOGSIG_DEPTH=2
LAVA_ACTION_TARGET_LAYER="final"
LAVA_ACTION_SIMILARITY_WEIGHTING="true"
LAVA_ACTION_SIMILARITY_MIN_WEIGHT="0.1"
LAVA_ACTION_SIMILARITY_BETA_MOMENTUM="0.99"
LAVA_ACTION_SIMILARITY_BETA_MULTIPLIER="1.0"
LAVA_ACTION_GRIPPER_INDICES="[6,13]"
LAVA_ACTION_GRIPPER_STATE_WEIGHT="0.5"
LAVA_ACTION_GRIPPER_CHANGE_WEIGHT="0.5"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")}"
STAGE="stage2"
SAVE_ROOT="${SAVE_ROOT:-${WORKSPACE_ROOT}/runs/LAVA/robotwin/stage2}"
STAGE1_SAVE_ROOT="${STAGE1_SAVE_ROOT:-${WORKSPACE_ROOT}/runs/LAVA/robotwin/stage1}"

# shellcheck source=../train_robotwin_lava_common.sh
source "${SCRIPT_DIR}/../train_robotwin_lava_common.sh"
