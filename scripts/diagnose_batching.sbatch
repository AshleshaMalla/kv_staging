#!/usr/bin/env bash
#SBATCH --job-name=kv_diag
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=data/raw/diagnose_batching_%j.out
#SBATCH --error=data/raw/diagnose_batching_%j.err

set -euo pipefail

cd /mnt/REPACSS/home/asmalla/kv_staging

export LD_LIBRARY_PATH="/mnt/REPACSS/home/asmalla/miniforge3/envs/m1/lib:${LD_LIBRARY_PATH:-}"

if ! command -v nvcc &>/dev/null; then
    for candidate in \
        /opt/apps/nfs/spack-1.1.0/opt/spack/linux-sapphirerapids/cuda-12.9.1-tio2hjc6xnw4bpsn37tz5fmwkz4dabp7 \
        /opt/apps/nfs/spack-1.0.1/opt/spack/linux-sapphirerapids/cuda-12.9.0-u2ppthmoi4r6ddyxdqesrj5oqja3byph \
        /usr/local/cuda; do
        if [[ -x "${candidate}/bin/nvcc" ]]; then
            export CUDA_HOME="${candidate}"
            export PATH="${CUDA_HOME}/bin:${PATH}"
            break
        fi
    done
fi

eval "$(/mnt/REPACSS/home/asmalla/miniforge3/bin/conda shell.bash hook)"
conda activate m1

echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name,clocks.sm,clocks.max.sm --format=csv,noheader)"
echo ""

PYTHONNOUSERSITE=1 python3 scripts/diagnose_batching.py \
    --model /mnt/SHARED-AREA/Llama-series/Llama-3.1-8B \
    --length 4096 \
    --concurrency 4 \
    --gpu-mem 0.90
