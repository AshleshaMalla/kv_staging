#!/bin/bash
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --gpus=1
#SBATCH --mem=200G
#SBATCH --time=02:00:00
#SBATCH -o data/raw/qn_extended_%j.out
#SBATCH -J qn_extended

set -euo pipefail

module load cuda/13.0.2
eval "$(/mnt/REPACSS/home/asmalla/miniforge3/bin/conda shell.bash hook)"
conda activate m1
export PYTHONNOUSERSITE=1
export LD_PRELOAD=/mnt/REPACSS/home/asmalla/miniforge3/envs/m1/lib/libstdc++.so.6

echo "Job $SLURM_JOB_ID on $(hostname) at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit,memory.total --format=csv
echo ""

python3 scripts/qn_extended_sweep.py \
    --model /mnt/SHARED-AREA/Llama-series/Llama-3.1-8B \
    --lengths 4096,16384,32768 \
    --concurrencies 1,2,4,8,16,24,32,48,64 \
    --reps 3 \
    --warmup 1 \
    --gpu-mem 0.90 \
    --max-model-len 131072 \
    --max-num-batched-tokens 131072 \
    --async-points 1,4,16,32,48,64

echo "Done at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
