#!/usr/bin/env bash

#SBATCH --job-name=robotwin10_lava_layer6_balanced_1h100nvl
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
EPOCHS=12
TASK_SET="10"                    # "10" or "50"
LEARNING_RATE="2e-4"
LR_MIN="5e-5"
BATCH_SIZE=128
GRAD_ACCUM_STEPS=1
SEED=42
SAVE_INTERVAL_EPOCH=2
RESUME_CHECKPOINT=""             # empty: auto-resume the newest Stage-1 checkpoint

# LAVA parameters
LAMBDA_LAVA="0.01"
LAVA_SCALES="[1,2,4,8,16]"
LAVA_TEMPERATURE="0.07"
LAVA_SAMPLE_RATIO="0.25"
LAVA_SAMPLING_BALANCE="task_episode"
LAVA_WARMUP_RATIO="0.05"
LAVA_SCALE_SAMPLING="uniform"
LAVA_ORDER_NEGATIVE="true"      # full shuffle for every L >= 2
LAVA_LOGSIG_DEPTH=2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")}"
STAGE="stage1"
SAVE_ROOT="${SAVE_ROOT:-${WORKSPACE_ROOT}/runs/LAVA/robotwin/stage1}"

# shellcheck source=../train_robotwin_lava_common.sh
source "${SCRIPT_DIR}/../train_robotwin_lava_common.sh"
