#!/usr/bin/env bash
#SBATCH --job-name=v71_nvl_health_test
#SBATCH --account=cml-director
#SBATCH --partition=cml-director
#SBATCH --qos=cml-high_long
#SBATCH --gres=gpu:h100-nvl:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --chdir=/fs/cml-projects/WAM
#SBATCH --output=/fs/cml-projects/WAM/runs/slurm/%x-%j.out
#SBATCH --error=/fs/cml-projects/WAM/runs/slurm/%x-%j.err
set -euo pipefail
TASK_SOURCE=/fs/cml-projects/WAM/runs/LAVA/robotwin/stage1/robotwin10_lava_v7.1_1h100nvl/provenance/source
TASK_PYTHON=/fs/cml-projects/WAM/.conda/envs/lila-wam/bin/python
export TMPDIR=/tmp
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${TASK_SOURCE}:${PYTHONPATH:-}"
nvidia-smi
cd "${TASK_SOURCE}"
exec "${TASK_PYTHON}" -u train.py --config configs/robotwin_lava_v71.yaml \
  --norm_stats_path /fs/cml-projects/WAM/data/robotwin_200_10_assets/stat-local-200-10.json \
  --save_dir "/fs/cml-projects/WAM/runs/LAVA/validation/v71_nvl_health_${SLURM_JOB_ID}" \
  --max_steps 20 --no_save \
  --set training.lava_warmup_ratio=0.0 training.lava_grad_diagnostics_interval=1 \
  training.checkpoint_tag=v71_nvl_health_test
