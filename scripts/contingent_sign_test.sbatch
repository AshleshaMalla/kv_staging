#!/bin/bash
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH -o data/raw/contingent_sign_test_%j.out
#SBATCH -J sign_test

set -euo pipefail

eval "$(/mnt/REPACSS/home/asmalla/miniforge3/bin/conda shell.bash hook)"
conda activate m1
export PYTHONNOUSERSITE=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

python3 scripts/contingent_sign_test.py

echo ""
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
