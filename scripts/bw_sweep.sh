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

if ! command -v jq &>/dev/null; then
    echo "WARNING: jq not found; summary extraction will be skipped" >&2
    HAS_JQ=0
else
    HAS_JQ=1
fi

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
        --ioengine=libaio \
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

    if [ "$HAS_JQ" -eq 1 ]; then
        AGG_BW=$(jq '.jobs[0].read.bw / 1024' "$FIO_OUT" 2>/dev/null || echo "0")
        AGG_IOPS=$(jq '.jobs[0].read.iops' "$FIO_OUT" 2>/dev/null || echo "0")
        LAT_MEAN=$(jq '.jobs[0].read.clat_ns.mean / 1000' "$FIO_OUT" 2>/dev/null || echo "0")
        LAT_P99=$(jq '.jobs[0].read.clat_ns.percentile["99.000000"] / 1000' "$FIO_OUT" 2>/dev/null || echo "0")

        printf "%s,%d,%.1f,%.0f,%.1f,%.1f\n" "$TIER" "$NJOBS" "$AGG_BW" "$AGG_IOPS" "$LAT_MEAN" "$LAT_P99" >> "$SUMMARY_CSV"
        printf "  numjobs=%-2d  agg_read=%.1f MB/s  iops=%.0f  lat_mean=%.1f us  lat_p99=%.1f us\n" \
            "$NJOBS" "$AGG_BW" "$AGG_IOPS" "$LAT_MEAN" "$LAT_P99"
    else
        echo "  (jq not available, raw JSON written to $FIO_OUT)"
    fi
done

echo ""
echo "--- Summary ---"
column -t -s',' "$SUMMARY_CSV" 2>/dev/null || cat "$SUMMARY_CSV"
echo ""
echo "Raw data: $RAW_DIR"
echo "Summary:  $SUMMARY_CSV"
