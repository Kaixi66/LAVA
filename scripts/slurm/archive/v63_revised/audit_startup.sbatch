#!/usr/bin/env bash
#SBATCH --job-name=lava_v63_revised_startup
#SBATCH --account=scavenger
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:10:00
#SBATCH --chdir=/fs/cml-projects/WAM
#SBATCH --output=/fs/cml-projects/WAM/runs/slurm/%x-%j.out
#SBATCH --error=/fs/cml-projects/WAM/runs/slurm/%x-%j.err
set -euo pipefail
export TMPDIR=/tmp
exec /fs/cml-projects/WAM/.conda/envs/lila-wam/bin/python -u /fs/cml-projects/WAM/runs/LAVA/validation/v63_revised_20260912/audit_training.py --run-dir /fs/cml-projects/WAM/runs/LAVA/robotwin/stage1/robotwin10_lava_v6.3_1h100nvl --job-id "${V63_TRAIN_JOB:?missing train job}" --min-steps 400 --wait-seconds 3600 --output /fs/cml-projects/WAM/runs/LAVA/robotwin/stage1/robotwin10_lava_v6.3_1h100nvl/provenance/production_startup_audit.json
