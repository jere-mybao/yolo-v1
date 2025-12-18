#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8            # 1 process per GPU
#SBATCH --gpus-per-task=1              # each task gets 1 GPU
#SBATCH --cpus-per-task=6              # 48 / 8 = 6 CPU cores per process
#SBATCH --mem=150G
#SBATCH --time=72:00:00
#SBATCH --exclude=neu[301,306]
#SBATCH --job-name=train_yolov1
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out
#SBATCH --error=slurm_outputs/%x/err_log_%x_%j.err

set -euo pipefail

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[INFO] Not inside a Slurm job. Submitting via sbatch..."
  exec sbatch "$0"
fi

cd /n/fs/jborz/projects/casi
module load cudatoolkit/12.8

export PATH="/n/fs/jborz/projects/casi/.venv/bin:$PATH"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

torchrun --standalone --nproc_per_node=8 -m yolo_v1.trainer.trainer