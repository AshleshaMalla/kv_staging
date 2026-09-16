#!/usr/bin/env bash
#SBATCH --job-name=kv_aggressor
#SBATCH --partition=zen4
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=/mnt/REPACSS/home/asmalla/kv_staging/data/raw/contention_20260913T234348Z/aggressor_%j.out
#SBATCH --error=/mnt/REPACSS/home/asmalla/kv_staging/data/raw/contention_20260913T234348Z/aggressor_%j.err
#SBATCH --time=140:00

set -euo pipefail

EXP_DIR="/mnt/REPACSS/home/asmalla/kv_staging/data/raw/contention_20260913T234348Z"
AGG_FILE_DIR="/mnt/REPACSS/asmalla/contention_aggressor"
SCHEDULE="${EXP_DIR}/schedule.txt"

echo "Aggressor allocation: $(scontrol show hostnames $SLURM_JOB_NODELIST | wc -l) nodes"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Started: $(date -Iseconds)"

mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")

# Create per-node test files (8G each, distinct from victim)
echo "Creating aggressor test files..."
for node in "${ALL_NODES[@]}"; do
    NODE_DIR="${AGG_FILE_DIR}/node_${node}"
    mkdir -p "$NODE_DIR"
    FPATH="${NODE_DIR}/testfile"
    EXPECTED=$((8 * 1024 * 1024 * 1024))
    if [[ -f "$FPATH" ]]; then
        ACTUAL=$(stat -c%s "$FPATH" 2>/dev/null || echo 0)
        if [[ "$ACTUAL" -ge "$EXPECTED" ]]; then
            continue
        fi
        rm -f "$FPATH"
    fi
    dd if=/dev/urandom of="$FPATH" bs=1M count=8192 status=none &
done
wait
echo "Aggressor files ready"

# Signal readiness
touch "${EXP_DIR}/aggressor_ready"

# Wait for T0
while grep -q "PLACEHOLDER" "$SCHEDULE" 2>/dev/null; do
    sleep 2
done

T0=$(grep "^T0=" "$SCHEDULE" | cut -d= -f2)
NOW=$(date +%s)
if [[ $T0 -gt $NOW ]]; then
    echo "Sleeping $((T0 - NOW))s until T0..."
    sleep $((T0 - NOW))
fi

echo "T0 reached: $(date -Iseconds)"

# Write the aggressor worker script
WORKER="${EXP_DIR}/_agg_worker.sh"
cat > "$WORKER" << 'WORKEREOF'
#!/usr/bin/env bash
set -euo pipefail
AGG_FILE_DIR="$1"; DURATION="$2"
MYHOST=$(hostname)
FPATH="${AGG_FILE_DIR}/node_${MYHOST}/testfile"
echo "$(date +%s.%N) ${MYHOST} START" >> "${3}"
fio --name=aggressor \
    --filename="${FPATH}" \
    --rw=read --bs=128k --direct=1 \
    --ioengine=posixaio --iodepth=16 --numjobs=4 \
    --group_reporting --size=8G \
    --runtime="${DURATION}" --time_based \
    --output-format=json --output=/dev/null 2>/dev/null
echo "$(date +%s.%N) ${MYHOST} STOP" >> "${3}"
WORKEREOF
chmod +x "$WORKER"

# Read schedule
declare -a CYC_NUM CYC_AGG CYC_LSTART CYC_LSTOP
IDX=0
while read -r line; do
    [[ "$line" =~ ^# ]] && continue
    [[ "$line" =~ ^T0 ]] && continue
    read -r cn ca qs ls lst rs <<< "$line"
    CYC_NUM[$IDX]=$cn; CYC_AGG[$IDX]=$ca
    CYC_LSTART[$IDX]=$ls; CYC_LSTOP[$IDX]=$lst
    IDX=$((IDX + 1))
done < "$SCHEDULE"

# Execute cycles
for i in "${!CYC_NUM[@]}"; do
    NC=${CYC_AGG[$i]}
    L_START=$((T0 + CYC_LSTART[$i]))
    L_STOP=$((T0 + CYC_LSTOP[$i]))
    LOAD_DUR=$((L_STOP - L_START))
    CYCLE=${CYC_NUM[$i]}

    SELECTED=("${ALL_NODES[@]:0:$NC}")
    NODELIST=$(IFS=,; echo "${SELECTED[*]}")

    AGG_LOG="${EXP_DIR}/agg_cycle${CYCLE}_timestamps.log"
    > "$AGG_LOG"

    # Sleep until load start
    NOW=$(date +%s)
    if [[ $L_START -gt $NOW ]]; then
        echo "Cycle ${CYCLE}: sleeping $((L_START - NOW))s until load start (${NC} nodes)..."
        sleep $((L_START - NOW))
    fi

    echo "Cycle ${CYCLE}: starting ${NC} aggressors at $(date -Iseconds)"
    echo "cycle_${CYCLE}_start $(date +%s.%N)" >> "${EXP_DIR}/aggressor_events.log"

    srun --nodes="${NC}" --ntasks="${NC}" --ntasks-per-node=1 \
        --nodelist="${NODELIST}" \
        bash "$WORKER" "$AGG_FILE_DIR" "$LOAD_DUR" "$AGG_LOG" &
    SRUN_PID=$!

    # Wait for the load phase to end
    sleep "$LOAD_DUR" 2>/dev/null || true
    # srun should have finished by now (fio --runtime matches LOAD_DUR)
    wait $SRUN_PID 2>/dev/null || true

    echo "Cycle ${CYCLE}: aggressors stopped at $(date -Iseconds)"
    echo "cycle_${CYCLE}_stop $(date +%s.%N)" >> "${EXP_DIR}/aggressor_events.log"
done

echo ""
echo "Aggressor job done: $(date -Iseconds)"
