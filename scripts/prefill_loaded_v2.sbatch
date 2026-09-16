#!/usr/bin/env bash
#SBATCH --job-name=kv_load_v2
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=data/raw/prefill_loaded_v2_%j.out
#SBATCH --error=data/raw/prefill_loaded_v2_%j.err

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
echo "Memory: $(free -g | awk '/^Mem:/{print $2}') GB"
echo "CUDA_HOME: ${CUDA_HOME:-unset}"
echo "nvcc: $(which nvcc 2>/dev/null || echo not found)"
echo ""

PYTHONNOUSERSITE=1 python3 scripts/prefill_loaded_sweep_v2.py \
    --model /mnt/SHARED-AREA/Llama-series/Llama-3.1-8B \
    --lengths 2048,4096,8192,16384,32768,65536 \
    --concurrencies 1,2,4,8,16 \
    --batch-token-configs 16384,32768,65536 \
    --reps 3 \
    --gpu-mem 0.90
