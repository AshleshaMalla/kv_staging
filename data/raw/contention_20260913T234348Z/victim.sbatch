#!/usr/bin/env bash
#SBATCH --job-name=kv_victim
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=/mnt/REPACSS/home/asmalla/kv_staging/data/raw/contention_20260913T234348Z/victim_%j.out
#SBATCH --error=/mnt/REPACSS/home/asmalla/kv_staging/data/raw/contention_20260913T234348Z/victim_%j.err
#SBATCH --time=140:00

set -euo pipefail

EXP_DIR="/mnt/REPACSS/home/asmalla/kv_staging/data/raw/contention_20260913T234348Z"
VICTIM_FILE_DIR="/mnt/REPACSS/asmalla/contention_victim"
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
