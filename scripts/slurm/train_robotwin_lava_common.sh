#!/usr/bin/env bash

# Shared implementation for the Stage-1 and Stage-2 Slurm entry points.
# This file is sourced by an sbatch script after that script defines its
# experiment parameters.

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "Submit the stage script with sbatch; do not run it with bash." >&2
  exit 2
fi
if (( $# != 0 )); then
  echo "Edit parameters at the top of the stage script; CLI arguments are disabled." >&2
  exit 2
fi

COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${COMMON_DIR}/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-$(dirname "${REPO_ROOT}")}"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/.conda/envs/lila-wam/bin/python}"
DINO_MODEL="${DINO_MODEL:-${WORKSPACE_ROOT}/data/models/dinov3-vitl16-pretrain-lvd1689m}"
RUN_NAME="${SLURM_JOB_NAME}"
RUN_SAVE_DIR="${SAVE_ROOT}/${RUN_NAME}"

find_latest_checkpoint() {
  local search_dir="$1" pattern="$2" max_epoch="$3"
  local candidate filename epoch_text epoch_num mtime
  local latest_path="" latest_epoch=-1 latest_mtime=-1

  [[ -d "${search_dir}" ]] || return 0
  while IFS= read -r -d '' candidate; do
    filename="${candidate##*/}"
    epoch_text="${filename##*_epoch_}"
    epoch_text="${epoch_text%.pt}"
    [[ "${epoch_text}" =~ ^[0-9]+$ ]] || continue
    epoch_num=$((10#${epoch_text}))
    (( epoch_num <= max_epoch )) || continue
    mtime="$(stat -c '%Y' "${candidate}")"
    if (( epoch_num > latest_epoch || (epoch_num == latest_epoch && mtime > latest_mtime) )); then
      latest_path="${candidate}"
      latest_epoch="${epoch_num}"
      latest_mtime="${mtime}"
    fi
  done < <(find "${search_dir}" -type f -name "${pattern}" -print0 2>/dev/null)
  printf '%s' "${latest_path}"
}

case "${TASK_SET}" in
  50)
    ROBOTWIN_DATA="${ROBOTWIN_DATA_50:-${WORKSPACE_ROOT}/data/robotwin}"
    ROBOTWIN_VTT="${ROBOTWIN_VTT_50:-${REPO_ROOT}/data-500-taskcond}"
    NORM_STATS="${NORM_STATS_50:-${REPO_ROOT}/utils/stat-500-all.json}"
    ;;
  10)
    ROBOTWIN_DATA="${ROBOTWIN_DATA_10:-${WORKSPACE_ROOT}/data/robotwin_200_10}"
    ROBOTWIN_VTT="${ROBOTWIN_VTT_10:-${WORKSPACE_ROOT}/data/robotwin_200_10_assets/vtt-local-200-10}"
    NORM_STATS="${NORM_STATS_10:-${WORKSPACE_ROOT}/data/robotwin_200_10_assets/stat-local-200-10.json}"
    ;;
  *) echo "TASK_SET must be 10 or 50, got ${TASK_SET}" >&2; exit 2 ;;
esac

[[ "${RUN_NAME}" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid RUN_NAME: ${RUN_NAME}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python environment: ${PYTHON_BIN}" >&2; exit 1; }
[[ -d "${ROBOTWIN_DATA}" ]] || { echo "Missing dataset: ${ROBOTWIN_DATA}" >&2; exit 1; }
[[ -f "${DINO_MODEL}/config.json" ]] || { echo "Missing DINOv3 checkpoint: ${DINO_MODEL}" >&2; exit 1; }
[[ -f "${ROBOTWIN_VTT}/meta.json" ]] || { echo "Missing VTT data: ${ROBOTWIN_VTT}" >&2; exit 1; }
[[ -f "${NORM_STATS}" ]] || { echo "Missing normalization statistics: ${NORM_STATS}" >&2; exit 1; }

TRAIN_ARGS=(
  --config "${REPO_ROOT}/configs/robotwin_all.yaml"
  --norm_stats_path "${NORM_STATS}"
  --save_dir "${RUN_SAVE_DIR}"
)

if [[ "${STAGE}" == "stage1" ]]; then
  if [[ -z "${RESUME_CHECKPOINT}" ]]; then
    RESUME_CHECKPOINT="$(find_latest_checkpoint \
      "${RUN_SAVE_DIR}" "checkpoint_${RUN_NAME}_epoch_*.pt" "${EPOCHS}")"
  fi
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    echo "Resuming Stage 1 from ${RESUME_CHECKPOINT}"
    TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
  else
    echo "Starting Stage 1 from scratch"
  fi
elif [[ "${STAGE}" == "stage2" ]]; then
  if [[ -z "${RESUME_CHECKPOINT}" ]]; then
    RESUME_CHECKPOINT="$(find_latest_checkpoint \
      "${RUN_SAVE_DIR}" "checkpoint_${RUN_NAME}_epoch_*.pt" "${EPOCHS}")"
  fi
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    echo "Resuming Stage 2 from ${RESUME_CHECKPOINT}"
    TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
  else
    if [[ -z "${STAGE1_CHECKPOINT}" ]]; then
      STAGE1_DIR="${STAGE1_SAVE_ROOT}/${RUN_NAME}"
      STAGE1_CHECKPOINT="$({
        find "${STAGE1_DIR}" -type f \
          -name "checkpoint_${RUN_NAME}_epoch_${STAGE1_FINAL_EPOCH}.pt" \
          -printf '%T@ %p\n' 2>/dev/null || true
      } | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
    fi
    [[ -n "${STAGE1_CHECKPOINT}" && -f "${STAGE1_CHECKPOINT}" ]] || {
      echo "No Stage-1 epoch-${STAGE1_FINAL_EPOCH} checkpoint found for ${RUN_NAME}" >&2
      exit 2
    }
    echo "Initializing Stage 2 from ${STAGE1_CHECKPOINT}"
    TRAIN_ARGS+=(--init_from "${STAGE1_CHECKPOINT}")
  fi
else
  echo "Unknown STAGE=${STAGE}" >&2
  exit 2
fi

OVERRIDES=(
  "dataset.dataset_dir=${ROBOTWIN_DATA}"
  "dataset.task_cond_dir=${ROBOTWIN_VTT}"
  "dataset.task_set=${TASK_SET}"
  "model.vision_encoder.checkpoint_path=${DINO_MODEL}"
  "model.lava.enabled=true"
  "model.lava.logsig_depth=${LAVA_LOGSIG_DEPTH}"
  "training.epochs=${EPOCHS}"
  "training.learning_rate=${LEARNING_RATE}"
  "training.lr_min=${LR_MIN}"
  "training.batch_size=${BATCH_SIZE}"
  "training.grad_accum_steps=${GRAD_ACCUM_STEPS}"
  "training.seed=${SEED}"
  "training.save_interval_epoch=${SAVE_INTERVAL_EPOCH}"
  "training.checkpoint_tag=${RUN_NAME}"
  "training.lambda_lava=${LAMBDA_LAVA}"
  "training.lava_scales=${LAVA_SCALES}"
  "training.lava_temperature=${LAVA_TEMPERATURE}"
  "training.lava_sample_ratio=${LAVA_SAMPLE_RATIO}"
  "training.lava_sampling_balance=${LAVA_SAMPLING_BALANCE}"
  "training.lava_warmup_ratio=${LAVA_WARMUP_RATIO}"
  "training.lava_scale_sampling=${LAVA_SCALE_SAMPLING}"
  "training.lava_order_negative=${LAVA_ORDER_NEGATIVE}"
)

mkdir -p "${RUN_SAVE_DIR}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Job ${SLURM_JOB_ID} | run ${RUN_NAME} | ${STAGE} | task set ${TASK_SET}"
echo "LAVA lambda=${LAMBDA_LAVA} scales=${LAVA_SCALES} tau=${LAVA_TEMPERATURE} ratio=${LAVA_SAMPLE_RATIO} balance=${LAVA_SAMPLING_BALANCE}"
echo "Output: ${RUN_SAVE_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" train.py "${TRAIN_ARGS[@]}" --set "${OVERRIDES[@]}"
