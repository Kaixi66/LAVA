#!/usr/bin/env bash

#SBATCH --job-name=robotwin10_lava_v7.14_1h200
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:h200-sxm:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=1-06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --chdir=/fs/cml-projects/WAM
#SBATCH --output=/fs/cml-projects/WAM/runs/slurm/%x-%j.out
#SBATCH --error=/fs/cml-projects/WAM/runs/slurm/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --requeue

set -euo pipefail

# =============================================================================
# LAVA RoboTwin Stage 1：每次实验只需要修改这一段。
# 对应的 Stage 1/Stage 2 必须使用相同的 --job-name（唯一 RUN_NAME）。
# =============================================================================
EPOCHS=12
TASK_SET="10"                   # "10"（首轮 subset）或 "50"
LEARNING_RATE="2e-4"
LR_MIN="5e-5"
BATCH_SIZE=128
GRAD_ACCUM_STEPS=1
SEED=42
SAVE_INTERVAL_EPOCH=2
SAVE_DIR="/fs/cml-projects/WAM/runs/LAVA/robotwin/stage1"
RESUME_CHECKPOINT=""            # 留空时自动续接同 RUN_NAME 最新 Stage 1 checkpoint

# V7.14 on H200 SXM (preemptible scavenger; requeued jobs resume from the latest checkpoint): V7.12 plus one same-start cross-scale negative per anchor: extend (used only when the non-shared part moves; joint-action path length >= 0.25x shared part).
LAMBDA_LAVA="0.01"
LAVA_SCALES="[1,2,4,8,16]"
LAVA_TEMPERATURE="0.07"
LAVA_SAMPLE_RATIO="0.125"          # ~16 anchors/batch; positive + up to 1 same-episode + 4 cross-episode
LAVA_SAMPLING_BALANCE="task_episode" # LAVA-only: task uniform, then episode uniform
LAVA_NEGATIVE_MODE="episode_balanced"
LAVA_NUM_SAME_EPISODE_NEGATIVES=1
LAVA_NUM_CROSS_EPISODE_NEGATIVES=4
LAVA_NEGATIVE_EXCLUSION_MULTIPLIER=2
LAVA_WARMUP_RATIO="0.05"
LAVA_WEIGHT_SCHEDULE="constant"
LAVA_MIN_WEIGHT="0.0"
LAVA_DECAY_START_RATIO="1.0"
LAVA_SCALE_SAMPLING="batch_uniform"
LAVA_ORDER_NEGATIVE="false"
LAVA_WORLD_ENCODING="state_delta"
LAVA_TIME_CHANNEL="true"
LAVA_RESIDUAL_DIM=128
LAVA_NUM_QUERIES=8
LAVA_QUERY_DIM=16
LAVA_LOGSIG_DEPTH=2
LAVA_SIGNATURE_NORMALIZATION="graded_soft"
LAVA_SIGNATURE_SCORE="neg_l2"
LAVA_SIGNATURE_RHO="1.0"
LAVA_ACTION_TARGET_LAYER="final"
LAVA_ACTION_SIMILARITY_WEIGHTING="false"
LAVA_ACTION_COMPONENT_CALIBRATION="false"
LAVA_QFORMER_LR_SCALE="${LAVA_QFORMER_LR_SCALE:-1.0}" # 0.25: slow world Q-Former experiment

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This is a Slurm submission script; submit it with sbatch, not bash." >&2
  exit 2
fi
if (( $# != 0 )); then
  echo "This script does not accept CLI training arguments; edit the parameters above." >&2
  exit 2
fi

WAM_ROOT="/fs/cml-projects/WAM"
LAVA_ROOT="${WAM_ROOT}/LiLa-WAM"
LAVA_ENV="${WAM_ROOT}/.conda/envs/lila-wam"
LAVA_PYTHON="${LAVA_ENV}/bin/python"
DINO_MODEL="${WAM_ROOT}/data/models/dinov3-vitl16-pretrain-lvd1689m"
RUN_NAME="${SLURM_JOB_NAME}"
RUN_SAVE_DIR="${SAVE_DIR}/${RUN_NAME}"
TRAIN_SOURCE_ROOT="${RUN_SAVE_DIR}/provenance/source"
[[ -f "${TRAIN_SOURCE_ROOT}/train.py" ]] || { echo "Missing validated V7.14 source snapshot" >&2; exit 1; }

find_latest_checkpoint() {
  local search_dir="$1"
  local pattern="$2"
  local max_epoch="$3"
  local candidate filename epoch_text epoch_num mtime
  local latest_path=""
  local latest_epoch=-1
  local latest_mtime=-1

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
    ROBOTWIN_DATA="${WAM_ROOT}/data/robotwin"
    ROBOTWIN_VTT="${LAVA_ROOT}/data-500-taskcond"
    NORM_STATS="${LAVA_ROOT}/utils/stat-500-all.json"
    ;;
  10)
    ROBOTWIN_DATA="${WAM_ROOT}/data/robotwin_200_10"
    ROBOTWIN_VTT="${WAM_ROOT}/data/robotwin_200_10_assets/vtt-local-200-10"
    NORM_STATS="${WAM_ROOT}/data/robotwin_200_10_assets/stat-local-200-10.json"
    ;;
  *)
    echo "TASK_SET must be 50 or 10, got: ${TASK_SET}" >&2
    exit 2
    ;;
esac

[[ "${RUN_NAME}" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid Slurm job name: ${RUN_NAME}" >&2; exit 2; }
[[ -x "${LAVA_PYTHON}" ]] || { echo "Missing environment: ${LAVA_ENV}" >&2; exit 1; }
[[ -d "${ROBOTWIN_DATA}" ]] || { echo "Missing dataset: ${ROBOTWIN_DATA}" >&2; exit 1; }
[[ -f "${DINO_MODEL}/config.json" ]] || { echo "Missing DINOv3 model: ${DINO_MODEL}" >&2; exit 1; }
[[ -f "${ROBOTWIN_VTT}/meta.json" ]] || { echo "Missing VTT data: ${ROBOTWIN_VTT}" >&2; exit 1; }
[[ -f "${NORM_STATS}" ]] || { echo "Missing normalization stats: ${NORM_STATS}" >&2; exit 1; }

TRAIN_ARGS=(
  --config "${TRAIN_SOURCE_ROOT}/configs/robotwin_lava_v714.yaml"
  --norm_stats_path "${NORM_STATS}"
  --save_dir "${RUN_SAVE_DIR}"
)
if [[ -z "${RESUME_CHECKPOINT}" ]]; then
  RESUME_CHECKPOINT="$(find_latest_checkpoint \
    "${RUN_SAVE_DIR}" "checkpoint_${RUN_NAME}_epoch_*.pt" "${EPOCHS}")"
  if [[ -n "${RESUME_CHECKPOINT}" ]]; then
    echo "Auto-selected Stage 1 resume checkpoint: ${RESUME_CHECKPOINT}"
  else
    echo "No Stage 1 checkpoint found; starting LAVA Stage 1 from scratch."
  fi
fi
if [[ -n "${RESUME_CHECKPOINT}" ]]; then
  [[ -f "${RESUME_CHECKPOINT}" ]] || { echo "Missing resume checkpoint: ${RESUME_CHECKPOINT}" >&2; exit 2; }
  TRAIN_ARGS+=(--resume "${RESUME_CHECKPOINT}")
fi

OVERRIDES=(
  "dataset.dataset_dir=${ROBOTWIN_DATA}"
  "dataset.task_cond_dir=${ROBOTWIN_VTT}"
  "dataset.task_set=${TASK_SET}"
  "model.vision_encoder.checkpoint_path=${DINO_MODEL}"
  "model.lava.enabled=true"
  "model.lava.world_condition.enabled=false"
  "model.lava.world_encoding=${LAVA_WORLD_ENCODING}"
  "model.lava.time_channel=${LAVA_TIME_CHANNEL}"
  "model.lava.residual_dim=${LAVA_RESIDUAL_DIM}"
  "model.lava.qformer.num_queries=${LAVA_NUM_QUERIES}"
  "model.lava.qformer.query_dim=${LAVA_QUERY_DIM}"
  "model.lava.action_target_layer=${LAVA_ACTION_TARGET_LAYER}"
  "model.lava.logsig_depth=${LAVA_LOGSIG_DEPTH}"
  "model.lava.signature_normalization.type=${LAVA_SIGNATURE_NORMALIZATION}"
  "model.lava.signature_normalization.rho=${LAVA_SIGNATURE_RHO}"
  "model.lava.signature_score=${LAVA_SIGNATURE_SCORE}"
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
  "training.lava_negative_mode=${LAVA_NEGATIVE_MODE}"
  "training.lava_num_same_episode_negatives=${LAVA_NUM_SAME_EPISODE_NEGATIVES}"
  "training.lava_num_cross_episode_negatives=${LAVA_NUM_CROSS_EPISODE_NEGATIVES}"
  "training.lava_negative_exclusion_multiplier=${LAVA_NEGATIVE_EXCLUSION_MULTIPLIER}"
  "training.lava_reuse_world_signatures=true"
  "training.lava_cross_scale_negatives=[extend]"
  "training.lava_cross_scale_min_motion=0.25"
  "training.lava_warmup_ratio=${LAVA_WARMUP_RATIO}"
  "training.lava_weight_schedule=${LAVA_WEIGHT_SCHEDULE}"
  "training.lava_min_weight=${LAVA_MIN_WEIGHT}"
  "training.lava_decay_start_ratio=${LAVA_DECAY_START_RATIO}"
  "training.lava_scale_sampling=${LAVA_SCALE_SAMPLING}"
  "training.lava_order_negative=${LAVA_ORDER_NEGATIVE}"
  "training.lava_action_similarity_weighting=${LAVA_ACTION_SIMILARITY_WEIGHTING}"
  "training.lava_action_component_calibration=${LAVA_ACTION_COMPONENT_CALIBRATION}"
  "training.lava_qformer_lr_scale=${LAVA_QFORMER_LR_SCALE}"
)

mkdir -p "${WAM_ROOT}/runs/slurm" "${RUN_SAVE_DIR}"
PROVENANCE_DIR="${RUN_SAVE_DIR}/provenance"
mkdir -p "${PROVENANCE_DIR}"
git -C "${LAVA_ROOT}" rev-parse HEAD > "${PROVENANCE_DIR}/git_commit_at_launch.txt"
git -C "${LAVA_ROOT}" status --short > "${PROVENANCE_DIR}/git_status_at_launch.txt"
git -C "${LAVA_ROOT}" diff --binary > "${PROVENANCE_DIR}/working_tree_at_launch.patch"
cp "${BASH_SOURCE[0]}" "${PROVENANCE_DIR}/stage1_submission.sbatch"
"${LAVA_PYTHON}" -m pip freeze > "${PROVENANCE_DIR}/environment.freeze.txt"
"${LAVA_PYTHON}" - "${PROVENANCE_DIR}" <<'PYVERIFY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
for name, digest in json.loads((root / 'source_sha256.json').read_text()).items():
    actual = hashlib.sha256((root / 'source' / name).read_bytes()).hexdigest()
    if actual != digest:
        raise RuntimeError(f'Frozen training source mismatch: {name}')
print('Validated frozen V7.14 source hashes.')
PYVERIFY
export TMPDIR=/tmp
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="${TRAIN_SOURCE_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

echo "Job ID: ${SLURM_JOB_ID}"
echo "Job/Run name: ${RUN_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Stage: stage1 | Task set: ${TASK_SET} | Seed: ${SEED}"
echo "LAVA V7.14: action_tap=${LAVA_ACTION_TARGET_LAYER}, lambda=${LAMBDA_LAVA}, normalization=${LAVA_SIGNATURE_NORMALIZATION}, world_film=false, reuse_world_signatures=true, cross_scale=extend, negative_mode=episode_balanced, same=${LAVA_NUM_SAME_EPISODE_NEGATIVES}, cross=${LAVA_NUM_CROSS_EPISODE_NEGATIVES}, exclusion=${LAVA_NEGATIVE_EXCLUSION_MULTIPLIER}L"
echo "Save directory: ${RUN_SAVE_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

cd "${TRAIN_SOURCE_ROOT}"
exec "${LAVA_PYTHON}" train.py "${TRAIN_ARGS[@]}" --set "${OVERRIDES[@]}"
