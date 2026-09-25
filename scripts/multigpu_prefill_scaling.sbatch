#!/bin/bash
#SBATCH --job-name=multigpu_scale
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --mem=500G
#SBATCH --time=02:00:00
#SBATCH --output=data/raw/multigpu_scaling_%j.out
#SBATCH --error=data/raw/multigpu_scaling_%j.out

set -euo pipefail

echo "=== Job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds) ==="
nvidia-smi --query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,power.limit \
    --format=csv,noheader
echo

eval "$(/mnt/REPACSS/home/asmalla/miniforge3/bin/conda shell.bash hook)"
conda activate m1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONNOUSERSITE=1
# CUDA 12.9.1 from Spack — this is the CUDA_HOME used in all working vLLM runs
# (job 155219, 2026-09-16). NOT the conda env's nvidia/cu13 (CUDA 13), which
# causes FlashInfer CCCL header incompatibility.
export CUDA_HOME=/opt/apps/nfs/spack-1.1.0/opt/spack/linux-sapphirerapids/cuda-12.9.1-tio2hjc6xnw4bpsn37tz5fmwkz4dabp7
export PATH="$CUDA_HOME/bin:$PATH"

echo "CUDA_HOME=$CUDA_HOME"
echo "nvcc: $(which nvcc)"
echo

# Memory diagnostics
echo "=== Memory diagnostics ==="
echo "Slurm allocated CPUs: ${SLURM_CPUS_ON_NODE:-?}"
echo "Slurm --mem requested: ${SLURM_MEM_PER_NODE:-default (DefMemPerCPU × CPUs)}"
CGMEM=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || echo "unknown")
echo "Cgroup memory.max: $CGMEM"
echo "Physical RAM: $(free -g | awk '/^Mem:/{print $2}') GiB"
echo "Available RAM: $(free -g | awk '/^Mem:/{print $7}') GiB"
echo

MODEL=/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B
LENGTHS=16384,32768
REPS=5
WARMUP=1
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUTDIR=data/raw/multigpu_scaling_${TS}
mkdir -p "$OUTDIR"

WORKER="scripts/multigpu_worker.py"
COMMON="--model $MODEL --lengths $LENGTHS --reps $REPS --warmup $WARMUP --gpu-mem 0.85 --max-model-len 33000 --max-num-batched-tokens 32768"

# Step 1: Pre-warm FlashInfer JIT cache with a single-GPU run.
# This prevents a cache race when multiple processes launch simultaneously.
echo "=== Pre-warming FlashInfer JIT cache (single GPU) ==="
CUDA_VISIBLE_DEVICES=0 python3 $WORKER $COMMON \
    --out "${OUTDIR}/prewarm_gpu0.json" 2>&1 | tee "${OUTDIR}/prewarm.log"
echo "=== Pre-warm complete ==="
echo

# Verify cache is intact
if [ -f "$HOME/.cache/flashinfer/0.6.18/90a/cached_ops/sampling/sampling.so" ]; then
    echo "FlashInfer sampling cache: OK"
else
    echo "WARNING: FlashInfer cache not found"
fi
echo

# Start clock+power logger at 100ms for all 4 GPUs
nvidia-smi \
    --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used \
    --format=csv -lms 100 > "${OUTDIR}/gpu_clocks.csv" &
CLOCK_PID=$!

run_gpus() {
    local NGPU=$1
    echo ""
    echo "========================================"
    echo "  n_gpus = $NGPU"
    echo "========================================"
    local PIDS=()
    for ((g=0; g<NGPU; g++)); do
        CUDA_VISIBLE_DEVICES=$g python3 $WORKER $COMMON \
            --out "${OUTDIR}/gpu${g}_n${NGPU}.json" \
            > "${OUTDIR}/gpu${g}_n${NGPU}.log" 2>&1 &
        PIDS+=($!)
    done

    local ALL_OK=true
    for ((g=0; g<NGPU; g++)); do
        if wait "${PIDS[$g]}"; then
            echo "  GPU $g: OK"
            grep '^\[GPU' "${OUTDIR}/gpu${g}_n${NGPU}.log" || true
        else
            echo "  GPU $g: FAILED (rc=$?)"
            echo "  --- last 20 lines of log ---"
            tail -20 "${OUTDIR}/gpu${g}_n${NGPU}.log"
            echo "  ---"
            ALL_OK=false
        fi
    done

    echo "  Host memory after G=$NGPU: $(free -g | awk '/^Mem:/{printf "%dG used / %dG total / %dG available", $3, $2, $7}')"
    $ALL_OK && echo "  All $NGPU GPU(s) succeeded" || echo "  Some GPU(s) failed"
}

for NGPU in 1 2 3 4; do
    run_gpus $NGPU
done

# Stop clock logger
kill $CLOCK_PID 2>/dev/null || true
wait $CLOCK_PID 2>/dev/null || true

echo ""
echo "=== Raw results in $OUTDIR ==="
ls -la "$OUTDIR"

# Run analysis
echo ""
python3 scripts/multigpu_analyze.py --data-dir "$OUTDIR"

echo ""
echo "=== Done at $(date -Iseconds) ==="
