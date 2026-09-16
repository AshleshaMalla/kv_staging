#!/usr/bin/env bash
set -euo pipefail

# bw_engine_compare.sh — Compare posixaio vs libaio single-node bandwidth.
# Runs posixaio sweep matching the multinode test params, 3 reps.
# Designed to A/B against the earlier libaio sweep in bw_shared_nfs_*.

TARGET_DIR="${1:?Usage: $0 <target_dir>}"
[[ ! -d "$TARGET_DIR" ]] && { echo "ERROR: '$TARGET_DIR' not a directory" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_DIR="${REPO_ROOT}/data/raw/bw_posixaio_sweep_${TIMESTAMP}"
mkdir -p "$RAW_DIR"

FIO_DIR="${TARGET_DIR}/${USER}/engine_compare"
mkdir -p "$FIO_DIR"

FILE_SIZE_GB=8
EXPECTED_BYTES=$((FILE_SIZE_GB * 1024 * 1024 * 1024))
NUM_FILES=16

echo "============================================================"
echo "  posixaio single-node sweep (engine confound check)"
echo "============================================================"
echo "Node:     $(hostname)"
echo "Target:   ${FIO_DIR}"
echo "Started:  $(date -Iseconds)"
echo ""

echo "--- Ensuring ${NUM_FILES} x ${FILE_SIZE_GB}G test files ---"
for i in $(seq 0 $((NUM_FILES - 1))); do
    FPATH="${FIO_DIR}/testfile_${i}"
    if [[ -f "$FPATH" ]]; then
        ACTUAL=$(stat -c%s "$FPATH" 2>/dev/null || echo 0)
        if [[ "$ACTUAL" -ge "$EXPECTED_BYTES" ]]; then
            echo "  testfile_${i}: exists ($(numfmt --to=iec "$ACTUAL")), skipping"
            continue
        fi
        rm -f "$FPATH"
    fi
    echo "  testfile_${i}: creating ${FILE_SIZE_GB}G ..."
    dd if=/dev/urandom of="$FPATH" bs=1M count=$((FILE_SIZE_GB * 1024)) status=progress 2>&1
done
echo ""

SUMMARY="${RAW_DIR}/summary.csv"
echo "engine,njobs,rep,bw_mbps,iops,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,lat_max_us" > "$SUMMARY"

REPS=3

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

for NJOBS in 1 2 4 8 16; do
    echo "=== numjobs=${NJOBS} ==="
    for REP in $(seq 1 "$REPS"); do
        FIO_OUT="${RAW_DIR}/njobs${NJOBS}_rep${REP}.json"

        sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true

        fio \
            --name="posixaio_nj${NJOBS}" \
            --directory="$FIO_DIR" \
            --nrfiles="${NUM_FILES}" \
            --rw=read \
            --bs=128k \
            --direct=1 \
            --ioengine=posixaio \
            --iodepth=16 \
            --numjobs="${NJOBS}" \
            --group_reporting \
            --size="${FILE_SIZE_GB}G" \
            --runtime=30 \
            --time_based \
            --output-format=json \
            --output="$FIO_OUT" 2>/dev/null || {
                echo "  WARNING: fio failed for njobs=${NJOBS} rep ${REP}" >&2
                continue
            }

        PARSED=$(parse_fio "$FIO_OUT") || continue
        read -r BW IOPS LAT_MEAN LAT_P50 LAT_P95 LAT_P99 LAT_MAX <<< "$PARSED"

        printf "posixaio,%d,%d,%s,%s,%s,%s,%s,%s,%s\n" \
            "$NJOBS" "$REP" "$BW" "$IOPS" "$LAT_MEAN" "$LAT_P50" "$LAT_P95" "$LAT_P99" "$LAT_MAX" \
            >> "$SUMMARY"

        printf "  rep %d: %8s MB/s  %6s IOPS  lat_mean=%s us  p99=%s us\n" \
            "$REP" "$BW" "$IOPS" "$LAT_MEAN" "$LAT_P99"
    done
    echo ""
done

echo "--- Summary ---"
column -t -s',' "$SUMMARY" 2>/dev/null || cat "$SUMMARY"
echo ""
echo "Raw data: $RAW_DIR"
echo "Summary:  $SUMMARY"
echo "Finished: $(date -Iseconds)"
