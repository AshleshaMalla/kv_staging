#!/usr/bin/env bash
set -euo pipefail

# hw_snapshot.sh — Record hardware state for provenance.
# Run inside a Slurm GPU allocation. No root required.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -Iseconds)"
JOBID="${SLURM_JOB_ID:-nojob}"
HOST="$(hostname -s)"
OUTDIR="${REPO_ROOT}/data/hw/${TIMESTAMP}_${JOBID}_${HOST}"

mkdir -p "$OUTDIR"

echo "=== hw_snapshot: writing to $OUTDIR ==="

# GPU topology
nvidia-smi topo -m > "$OUTDIR/gpu_topo.txt" 2>&1 || true

# GPU properties as CSV
nvidia-smi --query-gpu=index,name,memory.total,power.limit,pcie.link.gen.current,pcie.link.width.current \
    --format=csv > "$OUTDIR/gpu_props.csv" 2>&1 || true

# NUMA topology
numactl --hardware > "$OUTDIR/numactl.txt" 2>&1 || true

# CPU info
lscpu > "$OUTDIR/lscpu.txt" 2>&1 || true

# Per-NUMA-node CPU lists
for node_dir in /sys/devices/system/node/node*; do
    if [ -f "$node_dir/cpulist" ]; then
        node_name="$(basename "$node_dir")"
        echo "${node_name}: $(cat "$node_dir/cpulist")"
    fi
done > "$OUTDIR/numa_cpulists.txt" 2>&1

# Block devices
lsblk -o NAME,SIZE,ROTA,MODEL,MOUNTPOINT > "$OUTDIR/lsblk.txt" 2>&1 || true

# Mount points
mount > "$OUTDIR/mounts.txt" 2>&1 || true

# Memory info
cp /proc/meminfo "$OUTDIR/meminfo.txt" 2>&1 || true

# Transparent Huge Pages setting
THP_PATH="/sys/kernel/mm/transparent_hugepage/enabled"
if [ -f "$THP_PATH" ]; then
    cat "$THP_PATH" > "$OUTDIR/thp.txt" 2>&1
else
    echo "THP sysfs not found" > "$OUTDIR/thp.txt"
fi

# NUMA balancing
NUMA_BAL="/proc/sys/kernel/numa_balancing"
if [ -f "$NUMA_BAL" ]; then
    cat "$NUMA_BAL" > "$OUTDIR/numa_balancing.txt" 2>&1
else
    echo "numa_balancing not found" > "$OUTDIR/numa_balancing.txt"
fi

# RAPL (Running Average Power Limit)
ls /sys/class/powercap/intel-rapl/ > "$OUTDIR/rapl_listing.txt" 2>&1 || echo "RAPL directory not found" > "$OUTDIR/rapl_listing.txt"

RAPL_ENERGY="/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj"
if [ -f "$RAPL_ENERGY" ]; then
    cat "$RAPL_ENERGY" > "$OUTDIR/rapl_energy.txt" 2>&1 || echo "RAPL NOT READABLE" > "$OUTDIR/rapl_energy.txt"
else
    echo "RAPL NOT READABLE" > "$OUTDIR/rapl_energy.txt"
fi

# Environment summary
{
    echo "timestamp: $(date -Iseconds)"
    echo "hostname: $(hostname -f 2>/dev/null || hostname)"
    echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-not set}"
    echo ""
    echo "--- torch/cuda versions ---"
    python3 -c "import torch; print(f'torch: {torch.__version__}'); print(f'cuda: {torch.version.cuda}')" 2>&1 || echo "torch not importable"
    echo ""
    echo "--- relevant packages ---"
    pip freeze 2>/dev/null | grep -iE 'torch|transformers|vllm|lmcache' || echo "pip freeze unavailable or no matches"
} > "$OUTDIR/env.txt" 2>&1

echo "=== hw_snapshot complete ==="
echo "$OUTDIR"
