#!/usr/bin/env bash
set -euo pipefail

# contention_experiment.sh — Orchestrate a controlled contention experiment.
#
# Submits three sbatch jobs:
#   1. victim:     single h100 node running continuous bandwidth measurement
#   2. aggressors: zen4 nodes generating read load in scheduled on/off cycles
#   3. (file prep is done inline before submission)
#
# The experiment uses a shared schedule file on NFS so victim and aggressor
# jobs coordinate by wall-clock epoch without needing to be in the same
# allocation.
#
# Usage: ./contention_experiment.sh [max_aggressor_nodes]
#   max_aggressor_nodes: largest aggressor count (default 64)

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
EXP_DIR="${REPO_ROOT}/data/raw/contention_${TIMESTAMP}"
mkdir -p "$EXP_DIR"

MAX_AGG="${1:-64}"
TARGET_DIR="/mnt/REPACSS"

# Aggressor node counts to sweep
AGG_COUNTS=(1 4 16)
if [[ $MAX_AGG -ge 64 ]]; then
    AGG_COUNTS=(1 4 16 64)
fi

# Phase durations (seconds)
QUIET_DUR=300    # 5 min quiet
LOAD_DUR=600     # 10 min loaded
RECOVER_DUR=600  # 10 min recovery

# Compute total experiment duration
NUM_CYCLES=${#AGG_COUNTS[@]}
CYCLE_DUR=$((QUIET_DUR + LOAD_DUR + RECOVER_DUR))
SETUP_SLACK=300  # 5 min for file creation + sync
TOTAL_DUR=$((SETUP_SLACK + NUM_CYCLES * CYCLE_DUR + QUIET_DUR))
TOTAL_MIN=$(( (TOTAL_DUR + 59) / 60 ))

VICTIM_FILE_DIR="${TARGET_DIR}/${USER}/contention_victim"
AGG_FILE_DIR="${TARGET_DIR}/${USER}/contention_aggressor"

echo "============================================================"
echo "  Contention experiment setup"
echo "============================================================"
echo "Experiment dir:  ${EXP_DIR}"
echo "Aggressor sweep: ${AGG_COUNTS[*]} nodes"
echo "Cycle:           ${QUIET_DUR}s quiet + ${LOAD_DUR}s load + ${RECOVER_DUR}s recover"
echo "Cycles:          ${NUM_CYCLES}"
echo "Total duration:  ~${TOTAL_MIN} min"
echo ""

# ── Build the schedule file ─────────────────────────────────────
# The schedule is written BEFORE submission. Both victim and aggressor
# jobs read it at runtime to coordinate. Each line:
#   cycle_num  aggressor_count  phase_start_epoch  load_start_epoch  load_stop_epoch  recover_stop_epoch
#
# We compute epochs relative to a "T0" that is far enough in the future
# for both jobs to start and create files. We write T0 into the schedule
# and each job sleeps until T0.

SCHEDULE="${EXP_DIR}/schedule.txt"
echo "# Contention experiment schedule" > "$SCHEDULE"
echo "# Generated: $(date -Iseconds)" >> "$SCHEDULE"

# T0 will be filled in by the submission wrapper after we know both jobs
# are queued. For now, use a placeholder.
echo "T0=PLACEHOLDER" >> "$SCHEDULE"
echo "# cycle  agg_nodes  quiet_start  load_start  load_stop  recover_stop" >> "$SCHEDULE"

OFFSET=$SETUP_SLACK
for i in "${!AGG_COUNTS[@]}"; do
    CYCLE=$((i + 1))
    NC=${AGG_COUNTS[$i]}
    Q_START=$OFFSET
    L_START=$((OFFSET + QUIET_DUR))
    L_STOP=$((OFFSET + QUIET_DUR + LOAD_DUR))
    R_STOP=$((OFFSET + CYCLE_DUR))
    echo "${CYCLE} ${NC} ${Q_START} ${L_START} ${L_STOP} ${R_STOP}" >> "$SCHEDULE"
    OFFSET=$((OFFSET + CYCLE_DUR))
done

echo "Schedule:"
cat "$SCHEDULE"
echo ""

# ── Write victim sbatch script ──────────────────────────────────
VICTIM_SCRIPT="${EXP_DIR}/victim.sbatch"
cat > "$VICTIM_SCRIPT" << 'VICTIMEOF'
#!/usr/bin/env bash
#SBATCH --job-name=kv_victim
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=__EXP_DIR__/victim_%j.out
#SBATCH --error=__EXP_DIR__/victim_%j.err
#SBATCH --time=__TIME__

set -euo pipefail

EXP_DIR="__EXP_DIR__"
VICTIM_FILE_DIR="__VICTIM_FILE_DIR__"
SCHEDULE="${EXP_DIR}/schedule.txt"
CSV="${EXP_DIR}/victim_timeseries.csv"

FIO_BS=128k
FIO_NJOBS=4
FIO_IOENGINE=posixaio
FIO_IODEPTH=16
FIO_SIZE=8G
FIO_RUNTIME=20

mkdir -p "$VICTIM_FILE_DIR"
echo "Victim node: $(hostname)"
echo "Started: $(date -Iseconds)"

# Ensure test file
FPATH="${VICTIM_FILE_DIR}/testfile"
EXPECTED=$((8 * 1024 * 1024 * 1024))
if [[ -f "$FPATH" ]]; then
    ACTUAL=$(stat -c%s "$FPATH" 2>/dev/null || echo 0)
    if [[ "$ACTUAL" -ge "$EXPECTED" ]]; then
        echo "Test file exists, reusing"
    else
        dd if=/dev/urandom of="$FPATH" bs=1M count=8192 status=progress 2>&1
    fi
else
    dd if=/dev/urandom of="$FPATH" bs=1M count=8192 status=progress 2>&1
fi

# Signal readiness
touch "${EXP_DIR}/victim_ready"
echo "Victim ready, waiting for T0..."

# Wait for T0 to be set
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

# Get experiment end time from schedule
LAST_RECOVER=$(tail -1 "$SCHEDULE" | awk '{print $6}')
END_EPOCH=$((T0 + LAST_RECOVER + 300))

echo "sample,timestamp,wall_time,phase,agg_nodes,bw_mbps,iops,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,lat_max_us" > "$CSV"

parse_fio() {
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
r = d['jobs'][0]['read']
bw = r['bw'] / 1024
iops = r['iops']
ns = r['clat_ns']
mean = ns['mean'] / 1000
mx = ns['max'] / 1000
p = ns.get('percentile', {})
def getp(t):
    for fmt in [f'{t:.6f}', f'{t:g}']:
        if fmt in p: return p[fmt] / 1000
    return 0.0
print(f'{bw:.1f} {iops:.0f} {mean:.1f} {getp(50):.1f} {getp(95):.1f} {getp(99):.1f} {mx:.1f}')
" "$1"
}

# Read schedule into arrays
declare -a CYC_NUM CYC_AGG CYC_QSTART CYC_LSTART CYC_LSTOP CYC_RSTOP
IDX=0
while read -r line; do
    [[ "$line" =~ ^# ]] && continue
    [[ "$line" =~ ^T0 ]] && continue
    read -r cn ca qs ls lst rs <<< "$line"
    CYC_NUM[$IDX]=$cn; CYC_AGG[$IDX]=$ca
    CYC_QSTART[$IDX]=$qs; CYC_LSTART[$IDX]=$ls
    CYC_LSTOP[$IDX]=$lst; CYC_RSTOP[$IDX]=$rs
    IDX=$((IDX + 1))
done < "$SCHEDULE"

get_phase() {
    local elapsed=$1
    for i in "${!CYC_NUM[@]}"; do
        if [[ $elapsed -ge ${CYC_QSTART[$i]} && $elapsed -lt ${CYC_LSTART[$i]} ]]; then
            echo "quiet_${CYC_NUM[$i]} ${CYC_AGG[$i]}"; return
        elif [[ $elapsed -ge ${CYC_LSTART[$i]} && $elapsed -lt ${CYC_LSTOP[$i]} ]]; then
            echo "load_${CYC_NUM[$i]} ${CYC_AGG[$i]}"; return
        elif [[ $elapsed -ge ${CYC_LSTOP[$i]} && $elapsed -lt ${CYC_RSTOP[$i]} ]]; then
            echo "recover_${CYC_NUM[$i]} ${CYC_AGG[$i]}"; return
        fi
    done
    echo "quiet_0 0"
}

SAMPLE=0
while [[ $(date +%s) -lt $END_EPOCH ]]; do
    SAMPLE=$((SAMPLE + 1))
    NOW_TS=$(date -Iseconds)
    NOW_EPOCH=$(date +%s)
    ELAPSED=$((NOW_EPOCH - T0))

    read -r PHASE AGG_N <<< "$(get_phase $ELAPSED)"

    FIO_OUT="${EXP_DIR}/victim_sample_$(printf '%04d' "$SAMPLE").json"

    fio --name=victim_read \
        --filename="${FPATH}" \
        --rw=read \
        --bs="${FIO_BS}" \
        --direct=1 \
        --ioengine="${FIO_IOENGINE}" \
        --iodepth="${FIO_IODEPTH}" \
        --numjobs="${FIO_NJOBS}" \
        --group_reporting \
        --size="${FIO_SIZE}" \
        --runtime="${FIO_RUNTIME}" \
        --time_based \
        --output-format=json \
        --output="${FIO_OUT}" 2>/dev/null || continue

    PARSED=$(parse_fio "$FIO_OUT") || continue
    read -r BW IOPS LAT_MEAN LAT_P50 LAT_P95 LAT_P99 LAT_MAX <<< "$PARSED"

    printf "%d,%s,%d,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" \
        "$SAMPLE" "$NOW_TS" "$ELAPSED" "$PHASE" "$AGG_N" \
        "$BW" "$IOPS" "$LAT_MEAN" "$LAT_P50" "$LAT_P95" "$LAT_P99" "$LAT_MAX" \
        >> "$CSV"

    printf "#%-4d t=%5ds %-12s agg=%-3s %8s MB/s p99=%8s us\n" \
        "$SAMPLE" "$ELAPSED" "$PHASE" "$AGG_N" "$BW" "$LAT_P99"
done

echo ""
echo "Victim done: $(date -Iseconds), ${SAMPLE} samples"
VICTIMEOF

# Fill in placeholders
sed -i "s|__EXP_DIR__|${EXP_DIR}|g" "$VICTIM_SCRIPT"
sed -i "s|__VICTIM_FILE_DIR__|${VICTIM_FILE_DIR}|g" "$VICTIM_SCRIPT"
sed -i "s|__TIME__|$((TOTAL_MIN + 30)):00|g" "$VICTIM_SCRIPT"

# ── Write aggressor sbatch script ───────────────────────────────
AGG_SCRIPT="${EXP_DIR}/aggressor.sbatch"
cat > "$AGG_SCRIPT" << 'AGGEOF'
#!/usr/bin/env bash
#SBATCH --job-name=kv_aggressor
#SBATCH --partition=zen4
#SBATCH --nodes=__MAX_AGG__
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --output=__EXP_DIR__/aggressor_%j.out
#SBATCH --error=__EXP_DIR__/aggressor_%j.err
#SBATCH --time=__TIME__

set -euo pipefail

EXP_DIR="__EXP_DIR__"
AGG_FILE_DIR="__AGG_FILE_DIR__"
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
AGGEOF

sed -i "s|__EXP_DIR__|${EXP_DIR}|g" "$AGG_SCRIPT"
sed -i "s|__AGG_FILE_DIR__|${AGG_FILE_DIR}|g" "$AGG_SCRIPT"
sed -i "s|__MAX_AGG__|${MAX_AGG}|g" "$AGG_SCRIPT"
sed -i "s|__TIME__|$((TOTAL_MIN + 30)):00|g" "$AGG_SCRIPT"

echo "--- Submitting jobs ---"

# Submit victim
VICTIM_JOB=$(sbatch --parsable "$VICTIM_SCRIPT")
echo "Victim job:     ${VICTIM_JOB}"

# Submit aggressor
AGG_JOB=$(sbatch --parsable "$AGG_SCRIPT")
echo "Aggressor job:  ${AGG_JOB}"

echo ""
echo "Waiting for both jobs to be running and ready..."

# Wait for both to signal readiness (or timeout after 10 min)
WAIT_DEADLINE=$(($(date +%s) + 600))
while true; do
    if [[ -f "${EXP_DIR}/victim_ready" && -f "${EXP_DIR}/aggressor_ready" ]]; then
        break
    fi
    if [[ $(date +%s) -gt $WAIT_DEADLINE ]]; then
        echo "WARNING: timed out waiting for readiness signals. Setting T0 anyway."
        break
    fi
    sleep 5
    # Show job states
    V_STATE=$(squeue -j "$VICTIM_JOB" -o "%T" -h 2>/dev/null || echo "UNKNOWN")
    A_STATE=$(squeue -j "$AGG_JOB" -o "%T" -h 2>/dev/null || echo "UNKNOWN")
    echo "  victim=$V_STATE  aggressor=$A_STATE  $(date +%H:%M:%S)"
done

# Set T0 = now + 60s (give everyone time to wake up)
T0=$(($(date +%s) + 60))
sed -i "s|T0=PLACEHOLDER|T0=${T0}|" "$SCHEDULE"
echo ""
echo "T0 set: epoch ${T0} ($(date -d "@${T0}" +%H:%M:%S))"
echo ""

echo "============================================================"
echo "  Experiment submitted and synchronized"
echo "============================================================"
echo "Victim job:     ${VICTIM_JOB}"
echo "Aggressor job:  ${AGG_JOB}"
echo "Experiment dir: ${EXP_DIR}"
echo "Schedule:       ${SCHEDULE}"
echo ""
echo "Monitor with:"
echo "  tail -f ${EXP_DIR}/victim_${VICTIM_JOB}.out"
echo "  squeue -u ${USER}"
echo ""
echo "When complete, analyze with:"
echo "  python3 analysis/contention_response.py ${EXP_DIR}"
