#!/bin/bash
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH -o data/raw/stability_seeded_%j.out
#SBATCH -J stab_seed

set -euo pipefail

eval "$(/mnt/REPACSS/home/asmalla/miniforge3/bin/conda shell.bash hook)"
conda activate m1
export PYTHONNOUSERSITE=1

echo "Job $SLURM_JOB_ID on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

python3 scripts/stability_seeded.py

echo ""
echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
