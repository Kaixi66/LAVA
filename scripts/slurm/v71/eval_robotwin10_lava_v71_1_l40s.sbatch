#!/usr/bin/env bash

#SBATCH --job-name=robotwin10_lava_v7.1_1h100nvl
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --chdir=/fs/cml-projects/WAM
#SBATCH --output=/fs/cml-projects/WAM/runs/slurm/%x-eval-%j.out
#SBATCH --error=/fs/cml-projects/WAM/runs/slurm/%x-eval-%j.err

set -euo pipefail

# =============================================================================
# LAVA RoboTwin 10-task clean evaluation
# --job-name is the training RUN_NAME and locates the selected stage checkpoint.
# =============================================================================
CHECKPOINT_EPOCH="${CHECKPOINT_EPOCH:-12}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"  # empty: find this RUN_NAME/group/epoch checkpoint
CHECKPOINT_GROUP="${CHECKPOINT_GROUP:-stage1}"  # stage1 / stage2 / continuous
TASK_CONFIG="demo_clean"
INSTRUCTION_TYPE="unseen"
TEST_NUM=50
SEED="${SEED:-0}"
NUM_WORKERS=1                      # one SAPIEN/Vulkan process per GPU
SAVE_VIDEOS="${SAVE_VIDEOS:-1}"   # record every rollout for case-level analysis
EVAL_DEVICE_TAG="${EVAL_DEVICE_TAG:-l40s}"
SAVE_DIR="/fs/cml-projects/WAM/runs/LAVA/robotwin/eval"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "This is a Slurm script; submit it with sbatch, not bash." >&2
  exit 2
fi
if (( $# != 0 )); then
  echo "This script accepts no CLI arguments; edit its parameters at the top." >&2
  exit 2
fi
if [[ "${TASK_CONFIG}" != "demo_clean" && "${TASK_CONFIG}" != "demo_randomized" ]]; then
  echo "TASK_CONFIG must be demo_clean or demo_randomized, got ${TASK_CONFIG}" >&2
  exit 2
fi
if [[ "${INSTRUCTION_TYPE}" != "unseen" && "${INSTRUCTION_TYPE}" != "seen" ]]; then
  echo "INSTRUCTION_TYPE must be unseen or seen, got ${INSTRUCTION_TYPE}" >&2
  exit 2
fi
(( CHECKPOINT_EPOCH > 0 && TEST_NUM > 0 && NUM_WORKERS > 0 )) || {
  echo "CHECKPOINT_EPOCH, TEST_NUM and NUM_WORKERS must be positive." >&2
  exit 2
}
[[ "${SAVE_VIDEOS}" == "0" || "${SAVE_VIDEOS}" == "1" ]] || {
  echo "SAVE_VIDEOS must be 0 or 1." >&2
  exit 2
}

WAM_ROOT="/fs/cml-projects/WAM"
LAVA_ROOT="/fs/cml-projects/WAM/runs/LAVA/robotwin/stage1/robotwin10_lava_v7.1_1h100nvl/provenance/source"
ROBOTWIN_ROOT="${WAM_ROOT}/RoboTwin"
EVAL_ENV="${WAM_ROOT}/.conda/envs/robotwin-lila"
EVAL_PYTHON="${EVAL_ENV}/bin/python"
NORM_STATS="${WAM_ROOT}/data/robotwin_200_10_assets/stat-local-200-10.json"
RUN_NAME="${SLURM_JOB_NAME}"
case "${CHECKPOINT_GROUP}" in
  stage1|stage2|continuous) ;;
  *)
    echo "CHECKPOINT_GROUP must be stage1, stage2, or continuous, got ${CHECKPOINT_GROUP}" >&2
    exit 2
    ;;
esac
CHECKPOINT_SEARCH_DIR="${WAM_ROOT}/runs/LAVA/robotwin/${CHECKPOINT_GROUP}/${RUN_NAME}"
JOB_SAVE_DIR="${SAVE_DIR}/${RUN_NAME}/${TASK_CONFIG}/${EVAL_DEVICE_TAG}_job_${SLURM_JOB_ID}"
LOG_DIR="${JOB_SAVE_DIR}/logs"
RESULT_DIR="${JOB_SAVE_DIR}/bridge_results"
SUMMARY_PATH="${JOB_SAVE_DIR}/summary.txt"

TASKS=(
  adjust_bottle
  grab_roller
  hanging_mug
  move_stapler_pad
  open_microwave
  press_stapler
  scan_object
  stack_blocks_two
  stamp_seal
  turn_switch
)

[[ "${RUN_NAME}" =~ ^[a-zA-Z0-9._-]+$ ]] || {
  echo "Invalid RUN_NAME from Slurm job name: ${RUN_NAME}" >&2
  exit 2
}
[[ -x "${EVAL_PYTHON}" ]] || {
  echo "Missing RoboTwin/LAVA evaluation environment: ${EVAL_ENV}" >&2
  exit 1
}
[[ -f "${LAVA_ROOT}/eval_vla_bridge.py" ]] || {
  echo "Missing LAVA evaluation bridge: ${LAVA_ROOT}/eval_vla_bridge.py" >&2
  exit 1
}
[[ -f "${ROBOTWIN_ROOT}/task_config/${TASK_CONFIG}.yml" ]] || {
  echo "Missing RoboTwin task config: ${TASK_CONFIG}" >&2
  exit 1
}
[[ -d "${ROBOTWIN_ROOT}/assets/objects" && -d "${ROBOTWIN_ROOT}/assets/embodiments" ]] || {
  echo "RoboTwin simulator assets are missing under ${ROBOTWIN_ROOT}/assets" >&2
  exit 1
}
[[ -f "${NORM_STATS}" ]] || {
  echo "Missing subset normalization statistics: ${NORM_STATS}" >&2
  exit 1
}
if [[ "${SAVE_VIDEOS}" == "1" ]]; then
  EVAL_FFMPEG_BIN="$("${EVAL_PYTHON}" -c \
    'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
  [[ -x "${EVAL_FFMPEG_BIN}" ]] || {
    echo "SAVE_VIDEOS=1 but the environment ffmpeg binary is unavailable: ${EVAL_FFMPEG_BIN}" >&2
    exit 1
  }
  export EVAL_FFMPEG_BIN
fi

if [[ -z "${CHECKPOINT_PATH}" ]]; then
  CHECKPOINT_PATH="$({
    find "${CHECKPOINT_SEARCH_DIR}" -type f \
      -name "checkpoint_${RUN_NAME}_epoch_${CHECKPOINT_EPOCH}.pt" \
      -printf '%T@ %p\n' 2>/dev/null || true
  } | sort -nr | sed -n '1p' | cut -d' ' -f2-)"
  [[ -n "${CHECKPOINT_PATH}" ]] || {
    echo "No LAVA ${CHECKPOINT_GROUP} epoch-${CHECKPOINT_EPOCH} checkpoint found under:" >&2
    echo "  ${CHECKPOINT_SEARCH_DIR}" >&2
    exit 2
  }
  echo "Auto-selected checkpoint: ${CHECKPOINT_PATH}"
fi
[[ -f "${CHECKPOINT_PATH}" ]] || {
  echo "Missing checkpoint: ${CHECKPOINT_PATH}" >&2
  exit 2
}

CONFIG_PATH="$(dirname "${CHECKPOINT_PATH}")/config.yaml"
[[ -f "${CONFIG_PATH}" ]] || {
  echo "Missing checkpoint configuration: ${CONFIG_PATH}" >&2
  exit 2
}

mkdir -p "${LOG_DIR}" "${RESULT_DIR}"
export TMPDIR=/tmp
export ROBOTWIN_ROOT
export PYTHONPATH="${ROBOTWIN_ROOT}:${LAVA_ROOT}:${PYTHONPATH:-}"
source /etc/profile.d/modules.sh 2>/dev/null || true
module load cuda/12.8.1
export LD_LIBRARY_PATH="${EVAL_ENV}/lib/python3.10/site-packages/torch/lib:${EVAL_ENV}/lib/python3.10/site-packages/nvidia/cuda_runtime/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export EVAL_SAVE_VIDEOS="${SAVE_VIDEOS}"

{
  echo "run_name=${RUN_NAME}"
  echo "slurm_job_id=${SLURM_JOB_ID}"
  echo "device_tag=${EVAL_DEVICE_TAG}"
  echo "checkpoint=${CHECKPOINT_PATH}"
  echo "checkpoint_group=${CHECKPOINT_GROUP}"
  echo "checkpoint_epoch=${CHECKPOINT_EPOCH}"
  echo "task_config=${TASK_CONFIG}"
  echo "instruction_type=${INSTRUCTION_TYPE}"
  echo "rollouts_per_task=${TEST_NUM}"
  echo "seed=${SEED}"
  echo "save_videos=${SAVE_VIDEOS}"
  echo "ffmpeg_bin=${EVAL_FFMPEG_BIN:-disabled}"
  echo "git_commit=$(git -C "${LAVA_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "started_at=$(date --iso-8601=seconds)"
} > "${JOB_SAVE_DIR}/eval_manifest.txt"

echo "Job ID: ${SLURM_JOB_ID}"
echo "LAVA RUN_NAME: ${RUN_NAME}"
echo "Node: ${SLURMD_NODENAME}"
echo "Checkpoint: ${CHECKPOINT_PATH}"
echo "Checkpoint group: ${CHECKPOINT_GROUP}"
echo "Evaluation: RoboTwin 10-task subset | ${TASK_CONFIG}"
echo "Instruction type: ${INSTRUCTION_TYPE}"
echo "Rollouts per task: ${TEST_NUM}"
echo "Workers: ${NUM_WORKERS}"
echo "Save videos: ${SAVE_VIDEOS}"
echo "Output: ${JOB_SAVE_DIR}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee "${JOB_SAVE_DIR}/gpu_info.csv"

pids=()
worker_ids=()
for ((worker_id = 0; worker_id < NUM_WORKERS; worker_id++)); do
  shard=()
  for ((task_id = worker_id; task_id < ${#TASKS[@]}; task_id += NUM_WORKERS)); do
    shard+=("${TASKS[task_id]}")
  done
  (( ${#shard[@]} > 0 )) || continue

  echo "[worker ${worker_id}] tasks: ${shard[*]}"
  (
    cd "${ROBOTWIN_ROOT}"
    "${EVAL_PYTHON}" "${LAVA_ROOT}/eval_vla_bridge.py" \
      --task_names "${shard[@]}" \
      --task_config "${TASK_CONFIG}" \
      --instruction_type "${INSTRUCTION_TYPE}" \
      --ckpt_setting "${RUN_NAME}" \
      --checkpoint_ep "${CHECKPOINT_EPOCH}" \
      --checkpoint_path "${CHECKPOINT_PATH}" \
      --model_base_path "${LAVA_ROOT}" \
      --norm_stats_path "${NORM_STATS}" \
      --config_path "${CONFIG_PATH}" \
      --vla_root "${LAVA_ROOT}" \
      --save_root "${JOB_SAVE_DIR}" \
      --result_json "${RESULT_DIR}/worker_${worker_id}.json" \
      --seed "${SEED}" \
      --test_num "${TEST_NUM}" \
      >"${LOG_DIR}/worker_${worker_id}.log" 2>&1
  ) &
  pids+=("$!")
  worker_ids+=("${worker_id}")
done

worker_fail=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[index]}"; then
    echo "[worker ${worker_ids[index]}] failed; inspect ${LOG_DIR}/worker_${worker_ids[index]}.log" >&2
    worker_fail=1
  fi
done

"${EVAL_PYTHON}" - "${RESULT_DIR}" "${SUMMARY_PATH}" <<'PY'
import glob
import json
import os
import sys

result_dir, summary_path = sys.argv[1:]
results = []
for path in sorted(glob.glob(os.path.join(result_dir, "worker_*.json"))):
    if path.endswith("_status.json"):
        continue
    with open(path, encoding="utf-8") as stream:
        results.extend(json.load(stream)["results"])

lines = ["LAVA RoboTwin 10-task subset evaluation", "=" * 50]
successes = 0
rollouts = 0
for result in results:
    successes += result["success"]
    rollouts += result["test_num"]
    lines.append(
        f"{result['task_name']}: {result['success']}/{result['test_num']} "
        f"({result['success_rate'] * 100:.1f}%)"
    )
lines.append("=" * 50)
if rollouts:
    lines.append(
        f"Average success: {successes}/{rollouts} "
        f"({successes / rollouts * 100:.2f}%)"
    )
else:
    lines.append("No task produced results; inspect worker logs.")

summary = "\n".join(lines) + "\n"
print(summary, end="")
with open(summary_path, "w", encoding="utf-8") as stream:
    stream.write(summary)
PY

if (( worker_fail != 0 )); then
  echo "Evaluation was incomplete because at least one worker failed." >&2
  exit 1
fi
