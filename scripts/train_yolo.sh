#!/bin/bash

#SBATCH --nodes=1                                      ## Node count
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48                              ## Give the single process lots of CPU
#SBATCH --mem=150G                                      ## RAM per node
#SBATCH --time=72:00:00                                 ## Walltime
#SBATCH --gres=gpu:8                                    ## Number of GPUs
#SBATCH --exclude=neu[301,306]                          ## Exclude some nodes
#SBATCH --job-name=train_yolov1                         ## Job Name
#SBATCH --output=slurm_outputs/%x/out_log_%x_%j.out     ## Stdout File
#SBATCH --error=slurm_outputs/%x/err_log_%x_%j.err      ## Stderr File

set -euo pipefail

# If not running under Slurm, auto-submit this script to avoid login-node execution.
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "[INFO] Not inside a Slurm job. Submitting via sbatch to avoid running on the login node..."
  exec sbatch "$0"
fi

cd /n/fs/jborz/projects/casi
module load cudatoolkit/12.8

# Use the correct Python executable from the virtual environment
PYTHON_EXE="/n/fs/jborz/projects/casi/.venv/bin/python"

# Add virtual environment bin to PATH so ninja and other tools are accessible
export PATH="/n/fs/jborz/projects/casi/.venv/bin:$PATH"

# SLURM-specific threading
export SLURM_CPUS_PER_TASK=48
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

$PYTHON_EXE -m yolo_v1.trainer.trainer
