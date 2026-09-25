#!/bin/bash
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH -o data/raw/metastability_deep_%j.out
#SBATCH -J meta_deep

set -euo pipefail

eval "$(/mnt/REPACSS/home/asmalla/miniforge3/bin/conda shell.bash hook)"
conda activate m1
export PYTHONNOUSERSITE=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

python3 scripts/metastability_deep.py

echo ""
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
