#!/usr/bin/env bash
set -euo pipefail

# bw_timeseries.sh — Repeated short NFS bandwidth measurements over an extended
# period. The ONLY variable is time; everything else is fixed. Characterizes the
# variance structure of NFS read throughput under uncontrolled background load.
#
# Usage: ./bw_timeseries.sh <target_dir> [duration_minutes]
#   Defaults to running until 10 minutes before the SLURM allocation expires,
#   or duration_minutes if given.

usage() {
    echo "Usage: $0 <target_dir> [duration_minutes]" >&2
    exit 1
}

[[ $# -lt 1 ]] && usage
TARGET_DIR="$1"
[[ ! -d "$TARGET_DIR" ]] && { echo "ERROR: '$TARGET_DIR' not a directory" >&2; exit 1; }
command -v fio &>/dev/null || { echo "ERROR: fio not in PATH" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_DIR="${REPO_ROOT}/data/raw/bw_timeseries_${TIMESTAMP}"
mkdir -p "$RAW_DIR"

# Determine how long to run
if [[ -n "${2:-}" ]]; then
    DURATION_S=$(( $2 * 60 ))
elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
    REMAINING=$(squeue -j "$SLURM_JOB_ID" -o "%L" -h 2>/dev/null || echo "")
    if [[ -n "$REMAINING" && "$REMAINING" =~ ([0-9]+):([0-9]+):([0-9]+) ]]; then
        REMAINING_S=$(( ${BASH_REMATCH[1]} * 3600 + ${BASH_REMATCH[2]} * 60 + ${BASH_REMATCH[3]} ))
        DURATION_S=$(( REMAINING_S - 600 ))  # stop 10 min before expiry
        [[ $DURATION_S -lt 300 ]] && { echo "ERROR: <5 min of usable time left" >&2; exit 1; }
    else
        DURATION_S=7200
    fi
else
    DURATION_S=7200
fi

# Fixed fio parameters — must match multinode test for comparability
FIO_BS=128k
FIO_NJOBS=4
FIO_IOENGINE=posixaio
FIO_IODEPTH=16
FIO_SIZE=8G
FIO_RUNTIME=20  # short measurement window

FILE_DIR="${TARGET_DIR}/${USER}/timeseries"
mkdir -p "$FILE_DIR"

echo "============================================================"
echo "  NFS bandwidth time series"
echo "============================================================"
echo "Node:       $(hostname)"
echo "Target:     ${FILE_DIR}"
echo "Duration:   $((DURATION_S / 60)) minutes"
echo "Interval:   ~${FIO_RUNTIME}s fio + overhead"
echo "Started:    $(date -Iseconds)"
echo "Output:     ${RAW_DIR}"
echo ""

# Ensure test file exists
EXPECTED_BYTES=$((8 * 1024 * 1024 * 1024))
FPATH="${FILE_DIR}/testfile"
if [[ -f "$FPATH" ]]; then
    ACTUAL=$(stat -c%s "$FPATH" 2>/dev/null || echo 0)
    if [[ "$ACTUAL" -ge "$EXPECTED_BYTES" ]]; then
        echo "Test file exists ($(numfmt --to=iec "$ACTUAL")), reusing"
    else
        echo "Test file too small, recreating..."
        rm -f "$FPATH"
        dd if=/dev/urandom of="$FPATH" bs=1M count=8192 status=progress 2>&1
    fi
else
    echo "Creating 8G test file..."
    dd if=/dev/urandom of="$FPATH" bs=1M count=8192 status=progress 2>&1
fi
echo ""

CSV="${RAW_DIR}/timeseries.csv"
echo "sample,timestamp,wall_time,bw_mbps,iops,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,lat_max_us" > "$CSV"

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

START_EPOCH=$(date +%s)
DEADLINE=$((START_EPOCH + DURATION_S))
SAMPLE=0

echo "Running measurements until $(date -d "@${DEADLINE}" +%H:%M:%S)..."
echo ""

while [[ $(date +%s) -lt $DEADLINE ]]; do
    SAMPLE=$((SAMPLE + 1))
    NOW_TS=$(date -Iseconds)
    WALL_S=$(( $(date +%s) - START_EPOCH ))

    FIO_OUT="${RAW_DIR}/sample_$(printf '%04d' "$SAMPLE").json"

    fio --name=timeseries_read \
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
        --output="${FIO_OUT}" 2>/dev/null || {
            echo "  WARNING: fio failed on sample ${SAMPLE}" >&2
            continue
        }

    PARSED=$(parse_fio "$FIO_OUT") || continue
    read -r BW IOPS LAT_MEAN LAT_P50 LAT_P95 LAT_P99 LAT_MAX <<< "$PARSED"

    printf "%d,%s,%d,%s,%s,%s,%s,%s,%s,%s\n" \
        "$SAMPLE" "$NOW_TS" "$WALL_S" \
        "$BW" "$IOPS" "$LAT_MEAN" "$LAT_P50" "$LAT_P95" "$LAT_P99" "$LAT_MAX" \
        >> "$CSV"

    printf "  #%-4d  t=%5ds  %8s MB/s  p99=%8s us\n" \
        "$SAMPLE" "$WALL_S" "$BW" "$LAT_P99"
done

echo ""
echo "============================================================"
echo "  Time series complete: ${SAMPLE} samples"
echo "============================================================"
echo "CSV:      ${CSV}"
echo "Raw:      ${RAW_DIR}"
echo "Finished: $(date -Iseconds)"
echo ""
echo "Next: python3 analysis/bw_variance.py ${CSV}"
