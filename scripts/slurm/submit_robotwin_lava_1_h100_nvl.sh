#!/usr/bin/env bash

set -euo pipefail

# The Slurm job name is the unique RUN_NAME for both stages. Edit it once here.
RUN_NAME="robotwin10_lava_layer6_balanced_1h100nvl"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STAGE1_SCRIPT="${SCRIPT_DIR}/stage1/train_robotwin_lava_stage1_1_h100_nvl.sbatch"
STAGE2_SCRIPT="${SCRIPT_DIR}/stage2/train_robotwin_lava_stage2_1_h100_nvl.sbatch"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SUBMISSION_DIR="${REPO_ROOT}/runs/submissions"

if (( $# != 0 )); then
  echo "Edit RUN_NAME at the top; CLI arguments are disabled." >&2
  exit 2
fi
[[ "${RUN_NAME}" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "Invalid RUN_NAME" >&2; exit 2; }
command -v sbatch >/dev/null || { echo "sbatch is unavailable" >&2; exit 1; }

ACTIVE_JOBS="$(squeue --noheader --user "$(id -un)" --name "${RUN_NAME}" --format='%A %T' || true)"
if [[ -n "${ACTIVE_JOBS}" ]]; then
  echo "Active jobs already use RUN_NAME=${RUN_NAME}:" >&2
  echo "${ACTIVE_JOBS}" >&2
  exit 2
fi

STAGE1_JOB_ID="$(sbatch --parsable --job-name="${RUN_NAME}" "${STAGE1_SCRIPT}")"
STAGE1_JOB_ID="${STAGE1_JOB_ID%%;*}"
[[ "${STAGE1_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "Invalid Stage-1 job ID" >&2; exit 1; }

STAGE2_JOB_ID="$(sbatch --parsable --job-name="${RUN_NAME}" \
  --dependency="afterok:${STAGE1_JOB_ID}" --kill-on-invalid-dep=yes "${STAGE2_SCRIPT}")"
STAGE2_JOB_ID="${STAGE2_JOB_ID%%;*}"
[[ "${STAGE2_JOB_ID}" =~ ^[0-9]+$ ]] || { echo "Invalid Stage-2 job ID" >&2; exit 1; }

mkdir -p "${SUBMISSION_DIR}"
RECORD_PATH="${SUBMISSION_DIR}/${RUN_NAME}_$(date '+%Y-%m-%d_%H-%M-%S').txt"
printf 'run_name=%s\nstage1_job_id=%s\nstage2_job_id=%s\nstage2_dependency=afterok:%s\n' \
  "${RUN_NAME}" "${STAGE1_JOB_ID}" "${STAGE2_JOB_ID}" "${STAGE1_JOB_ID}" > "${RECORD_PATH}"

echo "Submitted ${RUN_NAME}: Stage1=${STAGE1_JOB_ID}, Stage2=${STAGE2_JOB_ID} (afterok)"
echo "Record: ${RECORD_PATH}"
