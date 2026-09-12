#!/usr/bin/env bash
set -uo pipefail

# hw_snapshot.sh — Record hardware state for provenance.
# Run inside a Slurm GPU allocation. No root required.
# No set -e: every command is individually guarded so a missing tool
# (e.g., numactl) never prevents later commands from running.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -Iseconds)"
JOBID="${SLURM_JOB_ID:-nojob}"
HOST="$(hostname -s)"
OUTDIR="${REPO_ROOT}/data/hw/${TIMESTAMP}_${JOBID}_${HOST}"

mkdir -p "$OUTDIR"

echo "=== hw_snapshot: writing to $OUTDIR ==="

# GPU topology
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi topo -m > "$OUTDIR/gpu_topo.txt" 2>&1 || echo "nvidia-smi topo failed" > "$OUTDIR/gpu_topo.txt"
else
    echo "UNAVAILABLE: nvidia-smi not found" > "$OUTDIR/gpu_topo.txt"
fi

# GPU properties as CSV
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=index,name,memory.total,power.limit,pcie.link.gen.current,pcie.link.width.current \
        --format=csv > "$OUTDIR/gpu_props.csv" 2>&1 || echo "nvidia-smi query failed" > "$OUTDIR/gpu_props.csv"
else
    echo "UNAVAILABLE: nvidia-smi not found" > "$OUTDIR/gpu_props.csv"
fi

# NUMA topology
if command -v numactl &>/dev/null; then
    numactl --hardware > "$OUTDIR/numactl.txt" 2>&1 || echo "numactl --hardware failed" > "$OUTDIR/numactl.txt"
else
    echo "UNAVAILABLE: numactl not found" > "$OUTDIR/numactl.txt"
fi

# CPU info
if command -v lscpu &>/dev/null; then
    lscpu > "$OUTDIR/lscpu.txt" 2>&1 || echo "lscpu failed" > "$OUTDIR/lscpu.txt"
else
    echo "UNAVAILABLE: lscpu not found" > "$OUTDIR/lscpu.txt"
fi

# Per-NUMA-node CPU lists
if ls /sys/devices/system/node/node* &>/dev/null; then
    for node_dir in /sys/devices/system/node/node*; do
        if [ -f "$node_dir/cpulist" ]; then
            node_name="$(basename "$node_dir")"
            echo "${node_name}: $(cat "$node_dir/cpulist")"
        fi
    done > "$OUTDIR/numa_cpulists.txt" 2>&1
else
    echo "UNAVAILABLE: /sys/devices/system/node/node* not found" > "$OUTDIR/numa_cpulists.txt"
fi

# Block devices
if command -v lsblk &>/dev/null; then
    lsblk -o NAME,SIZE,ROTA,MODEL,MOUNTPOINT > "$OUTDIR/lsblk.txt" 2>&1 || echo "lsblk failed" > "$OUTDIR/lsblk.txt"
else
    echo "UNAVAILABLE: lsblk not found" > "$OUTDIR/lsblk.txt"
fi

# Mount points
if command -v mount &>/dev/null; then
    mount > "$OUTDIR/mounts.txt" 2>&1 || echo "mount failed" > "$OUTDIR/mounts.txt"
else
    echo "UNAVAILABLE: mount not found" > "$OUTDIR/mounts.txt"
fi

# Memory info
if [ -f /proc/meminfo ]; then
    cp /proc/meminfo "$OUTDIR/meminfo.txt" 2>&1 || echo "failed to copy /proc/meminfo" > "$OUTDIR/meminfo.txt"
else
    echo "UNAVAILABLE: /proc/meminfo not found" > "$OUTDIR/meminfo.txt"
fi

# Transparent Huge Pages setting
THP_PATH="/sys/kernel/mm/transparent_hugepage/enabled"
if [ -f "$THP_PATH" ]; then
    cat "$THP_PATH" > "$OUTDIR/thp.txt" 2>&1 || echo "failed to read THP" > "$OUTDIR/thp.txt"
else
    echo "UNAVAILABLE: THP sysfs not found" > "$OUTDIR/thp.txt"
fi

# NUMA balancing
NUMA_BAL="/proc/sys/kernel/numa_balancing"
if [ -f "$NUMA_BAL" ]; then
    cat "$NUMA_BAL" > "$OUTDIR/numa_balancing.txt" 2>&1 || echo "failed to read numa_balancing" > "$OUTDIR/numa_balancing.txt"
else
    echo "UNAVAILABLE: numa_balancing not found" > "$OUTDIR/numa_balancing.txt"
fi

# RAPL (Running Average Power Limit)
if [ -d /sys/class/powercap/intel-rapl/ ]; then
    ls /sys/class/powercap/intel-rapl/ > "$OUTDIR/rapl_listing.txt" 2>&1 || echo "failed to list RAPL" > "$OUTDIR/rapl_listing.txt"
else
    echo "UNAVAILABLE: RAPL directory not found" > "$OUTDIR/rapl_listing.txt"
fi

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
    if command -v python3 &>/dev/null; then
        python3 -c "import torch; print(f'torch: {torch.__version__}'); print(f'cuda: {torch.version.cuda}')" 2>&1 || echo "torch not importable"
    else
        echo "UNAVAILABLE: python3 not found"
    fi
    echo ""
    echo "--- relevant packages ---"
    if command -v pip &>/dev/null; then
        pip freeze 2>/dev/null | grep -iE 'torch|transformers|vllm|lmcache' || echo "no matching packages"
    else
        echo "UNAVAILABLE: pip not found"
    fi
} > "$OUTDIR/env.txt" 2>&1

echo "=== hw_snapshot complete ==="
echo "$OUTDIR"
