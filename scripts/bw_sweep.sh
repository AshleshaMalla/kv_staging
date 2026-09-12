#!/usr/bin/env bash
set -euo pipefail

# bw_sweep.sh — Measure whether aggregate read bandwidth scales with parallelism.
#
# That single question decides whether our slow tier is single-stream-limited.
# If bandwidth plateaus at 1 stream, the tier is bottlenecked on something other
# than the link (e.g., NFS single-connection, metadata latency). If it scales to
# N streams, the bottleneck is per-stream and we can pipeline around it.
#
# Usage: ./bw_sweep.sh <target_dir> <tier_name>
#   target_dir: directory on the filesystem to test (will create test files there)
#   tier_name:  label for output (e.g., "hammerspace", "nvme")

usage() {
    echo "Usage: $0 <target_dir> <tier_name>" >&2
    echo "  target_dir: writable directory on the tier to benchmark" >&2
    echo "  tier_name:  label (e.g., hammerspace, nvme)" >&2
    exit 1
}

if [ $# -ne 2 ]; then
    usage
fi

TARGET_DIR="$1"
TIER="$2"

if [ ! -d "$TARGET_DIR" ]; then
    echo "ERROR: target_dir '$TARGET_DIR' does not exist or is not a directory" >&2
    exit 1
fi

if ! command -v fio &>/dev/null; then
    echo "ERROR: fio is not installed or not in PATH" >&2
    exit 1
fi

parse_fio_json() {
    python3 -c "
import json, sys
d = json.load(open(sys.argv[1]))
r = d['jobs'][0]['read']
bw = r['bw'] / 1024
iops = r['iops']
lat_mean = r['clat_ns']['mean'] / 1000
p = r['clat_ns'].get('percentile', {})
lat_p99 = p.get('99.000000', p.get('99.00000', 0)) / 1000
print(f'{bw:.1f} {iops:.0f} {lat_mean:.1f} {lat_p99:.1f}')
" "$1"
}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -Iseconds)"
RAW_DIR="${REPO_ROOT}/data/raw/bw_${TIER}_${TIMESTAMP}"
mkdir -p "$RAW_DIR"

FIO_DIR="${TARGET_DIR}/fio_testfiles"
mkdir -p "$FIO_DIR"

FILE_SIZE="1G"
NUM_FILES=16

# Create real (non-sparse) test files if they don't already exist.
# We need at least NUM_FILES files so that high-parallelism runs don't all hit the same file.
echo "--- Ensuring ${NUM_FILES} x ${FILE_SIZE} test files in ${FIO_DIR} ---"
for i in $(seq 0 $((NUM_FILES - 1))); do
    FPATH="${FIO_DIR}/testfile_${i}"
    if [ ! -f "$FPATH" ]; then
        echo "  Creating $FPATH ..."
        dd if=/dev/urandom of="$FPATH" bs=1M count=1024 status=progress 2>&1 || {
            echo "ERROR: failed to create test file $FPATH" >&2
            exit 1
        }
    else
        echo "  $FPATH already exists, skipping"
    fi
done

SUMMARY_CSV="${RAW_DIR}/summary.csv"
echo "tier,nstreams,agg_read_mbps,agg_iops,lat_mean_us,lat_p99_us" > "$SUMMARY_CSV"

echo ""
echo "--- Bandwidth sweep: tier=${TIER}, bs=128k, direct=1 ---"
echo ""

# --direct=1 bypasses the page cache. Without it, repeated reads of the same
# file would be served from DRAM, making the storage tier look like memory
# bandwidth. We want the actual device/network read speed.

for NJOBS in 1 2 4 8 16; do
    echo "=== numjobs=${NJOBS} ==="

    FIO_OUT="${RAW_DIR}/njobs${NJOBS}.json"

    fio \
        --name="read_${TIER}_nj${NJOBS}" \
        --directory="$FIO_DIR" \
        --nrfiles="${NUM_FILES}" \
        --rw=read \
        --bs=128k \
        --direct=1 \
        --ioengine=posixaio \
        --iodepth=16 \
        --numjobs="${NJOBS}" \
        --group_reporting \
        --size="${FILE_SIZE}" \
        --runtime=30 \
        --time_based \
        --output-format=json \
        --output="$FIO_OUT" 2>&1 || {
            echo "WARNING: fio failed for numjobs=${NJOBS}" >&2
            continue
        }

    PARSED=$(parse_fio_json "$FIO_OUT") || { echo "  WARNING: failed to parse $FIO_OUT" >&2; continue; }
    read -r AGG_BW AGG_IOPS LAT_MEAN LAT_P99 <<< "$PARSED"

    printf "%s,%d,%s,%s,%s,%s\n" "$TIER" "$NJOBS" "$AGG_BW" "$AGG_IOPS" "$LAT_MEAN" "$LAT_P99" >> "$SUMMARY_CSV"
    printf "  numjobs=%-2d  agg_read=%s MB/s  iops=%s  lat_mean=%s us  lat_p99=%s us\n" \
        "$NJOBS" "$AGG_BW" "$AGG_IOPS" "$LAT_MEAN" "$LAT_P99"
done

echo ""
echo "--- Summary ---"
column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo ""
echo "Raw data: $RAW_DIR"
echo "Summary:  $SUMMARY_CSV"
