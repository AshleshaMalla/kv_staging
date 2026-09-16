#!/usr/bin/env bash
#SBATCH --job-name=kv_fetch_oh
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=data/raw/fetch_overhead_%j.out
#SBATCH --error=data/raw/fetch_overhead_%j.err

set -euo pipefail

cd /mnt/REPACSS/home/asmalla/kv_staging

eval "$(/mnt/REPACSS/home/asmalla/miniforge3/bin/conda shell.bash hook)"
conda activate m1

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,clocks.sm --format=csv,noheader)"
echo "Memory: $(free -g | awk '/^Mem:/{print $2}') GB"
echo ""

PYTHONNOUSERSITE=1 python3 scripts/measure_fetch_overhead.py --target-dir /mnt/REPACSS --reps 15
