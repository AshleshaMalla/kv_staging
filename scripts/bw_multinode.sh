#!/usr/bin/env bash
set -euo pipefail

# bw_multinode.sh — Measure whether NFS read bandwidth is a per-node or shared
# backend ceiling by running synchronized fio across multiple SLURM nodes.
#
# Usage: ./bw_multinode.sh <target_dir>
#   Must run inside an salloc with SLURM_NNODES >= 2.

usage() {
    echo "Usage: $0 <target_dir>" >&2
    echo "  target_dir: NFS directory to benchmark (e.g., /mnt/shared)" >&2
    echo "  Must run inside an salloc with >= 2 nodes." >&2
    exit 1
}

[[ $# -ne 1 ]] && usage

TARGET_DIR="$1"

[[ ! -d "$TARGET_DIR" ]] && { echo "ERROR: '$TARGET_DIR' is not a directory" >&2; exit 1; }
command -v fio &>/dev/null || { echo "ERROR: fio not in PATH" >&2; exit 1; }
command -v srun &>/dev/null || { echo "ERROR: srun not in PATH — are you on a SLURM cluster?" >&2; exit 1; }

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: SLURM_JOB_ID not set. Run this inside an salloc allocation." >&2
    exit 1
fi

if [[ "${SLURM_NNODES:-0}" -le 1 ]]; then
    echo "ERROR: SLURM_NNODES=${SLURM_NNODES:-unset}. This test requires >= 2 nodes." >&2
    echo "       A single-node run would produce a misleading result for the" >&2
    echo "       per-node vs shared-ceiling question. Use bw_sweep.sh instead." >&2
    exit 1
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RAW_DIR="${REPO_ROOT}/data/raw/bw_multinode_${TIMESTAMP}"
mkdir -p "$RAW_DIR"

mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
NNODES=${#ALL_NODES[@]}

echo "============================================================"
echo "  Multi-node NFS bandwidth scaling test"
echo "============================================================"
echo "Allocation:  job ${SLURM_JOB_ID}, ${NNODES} nodes: ${ALL_NODES[*]}"
echo "Target:      ${TARGET_DIR}"
echo "Started:     $(date -Iseconds)"
echo "Output:      ${RAW_DIR}"
echo ""

# ── Clock skew check ────────────────────────────────────────────
echo "--- Clock skew check ---"
SKEW_FILE="${RAW_DIR}/clock_skew.txt"
srun --nodes="${NNODES}" --ntasks="${NNODES}" --ntasks-per-node=1 \
    bash -c 'echo "$(hostname) $(date +%s.%N)"' | sort > "$SKEW_FILE"
cat "$SKEW_FILE"

SKEW_OK=true
python3 - "$SKEW_FILE" <<'PYSKEW' || SKEW_OK=false
import sys
lines = open(sys.argv[1]).read().strip().split("\n")
times = [(l.split()[0], float(l.split()[1])) for l in lines]
if len(times) < 2:
    print("  Only one timestamp — skew check N/A")
    sys.exit(0)
lo = min(t for _, t in times)
hi = max(t for _, t in times)
skew_ms = (hi - lo) * 1000
print(f"  Max clock skew: {skew_ms:.1f} ms")
if skew_ms > 500:
    print("  WARNING: skew > 500 ms — synchronization unreliable", file=sys.stderr)
    sys.exit(1)
elif skew_ms > 100:
    print("  CAUTION: skew > 100 ms — timestamps have limited precision")
else:
    print("  Acceptable for synchronization")
PYSKEW

if [[ "$SKEW_OK" != "true" ]]; then
    echo "ERROR: clock skew too large for meaningful synchronization" >&2
    echo "       Results would not answer the per-node vs shared question." >&2
    exit 1
fi
echo ""

# ── Test parameters (matching single-node sweep for comparability) ──
FIO_BS=128k
FIO_NJOBS=4
FIO_IOENGINE=posixaio
FIO_IODEPTH=16
FIO_SIZE=8G
FIO_RUNTIME=60
FILE_SIZE_GB=8
LEAD_TIME=8          # seconds of slack before sync epoch
REPS=3

# ── Create per-node test files ──────────────────────────────────
FILE_DIR_BASE="${TARGET_DIR}/${USER}/multinode"

echo "--- Creating per-node test files (${FILE_SIZE_GB}G each) ---"
EXPECTED_BYTES=$((FILE_SIZE_GB * 1024 * 1024 * 1024))
for node in "${ALL_NODES[@]}"; do
    NODE_DIR="${FILE_DIR_BASE}/node_${node}"
    mkdir -p "$NODE_DIR"
    FPATH="${NODE_DIR}/testfile"
    if [[ -f "$FPATH" ]]; then
        ACTUAL=$(stat -c%s "$FPATH" 2>/dev/null || echo 0)
        if [[ "$ACTUAL" -ge "$EXPECTED_BYTES" ]]; then
            echo "  ${node}: exists ($(numfmt --to=iec "$ACTUAL")), skipping"
            continue
        fi
        echo "  ${node}: exists but too small ($(numfmt --to=iec "$ACTUAL")), recreating"
        rm -f "$FPATH"
    fi
    echo "  ${node}: creating ${FPATH} ..."
    dd if=/dev/urandom of="$FPATH" bs=1M count=$((FILE_SIZE_GB * 1024)) status=progress 2>&1
done
echo ""

echo "--- Verifying test file sizes ---"
for node in "${ALL_NODES[@]}"; do
    FPATH="${FILE_DIR_BASE}/node_${node}/testfile"
    ACTUAL=$(stat -c%s "$FPATH" 2>/dev/null || echo 0)
    if [[ "$ACTUAL" -lt "$EXPECTED_BYTES" ]]; then
        echo "ERROR: ${node}: $(numfmt --to=iec "$ACTUAL"), expected ${FILE_SIZE_GB}G" >&2
        exit 1
    fi
    echo "  ${node}: $(numfmt --to=iec "$ACTUAL") OK"
done
echo ""

# ── Write worker script to shared NFS ───────────────────────────
# Avoids nested-quoting hell in srun bash -c.
WORKER="${RAW_DIR}/_worker.sh"
cat > "$WORKER" << 'WORKEREOF'
#!/usr/bin/env bash
set -euo pipefail
FILE_DIR_BASE="$1"; START_EPOCH="$2"; RUN_DIR="$3"
FIO_BS="$4"; FIO_NJOBS="$5"; FIO_IOENGINE="$6"; FIO_IODEPTH="$7"; FIO_SIZE="$8"; FIO_RUNTIME="$9"

MYHOST=$(hostname)
MY_DIR="${FILE_DIR_BASE}/node_${MYHOST}"
FIO_OUT="${RUN_DIR}/fio_${MYHOST}.json"
TS_FILE="${RUN_DIR}/ts_${MYHOST}.txt"

# Sleep until the synchronized start epoch
NOW=$(date +%s.%N)
WAIT=$(echo "${START_EPOCH} - ${NOW}" | bc)
if (( $(echo "${WAIT} > 0" | bc -l) )); then
    sleep "${WAIT}"
fi

START_TS=$(date +%s.%N)

fio --name=multinode_read \
    --filename="${MY_DIR}/testfile" \
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
    --output="${FIO_OUT}" 2>/dev/null

END_TS=$(date +%s.%N)
echo "${MYHOST} ${START_TS} ${END_TS}" > "${TS_FILE}"
WORKEREOF
chmod +x "$WORKER"

# ── Output CSVs ─────────────────────────────────────────────────
PER_NODE_CSV="${RAW_DIR}/summary_per_node.csv"
AGGREGATE_CSV="${RAW_DIR}/summary_aggregate.csv"
echo "nodes,rep,hostname,bw_mbps,iops,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,lat_max_us,start_ts,end_ts" \
    > "$PER_NODE_CSV"
echo "nodes,rep,aggregate_bw_mbps,mean_per_node_bw_mbps,overlap_fraction" \
    > "$AGGREGATE_CSV"

# ── Determine node counts ───────────────────────────────────────
NODE_COUNTS=()
for n in 1 2 4; do
    [[ $n -le $NNODES ]] && NODE_COUNTS+=("$n")
done

echo "Node counts:  ${NODE_COUNTS[*]}  (allocation has ${NNODES})"
echo "Reps:         ${REPS}"
echo "fio:          bs=${FIO_BS}  njobs=${FIO_NJOBS}  iodepth=${FIO_IODEPTH}  size=${FIO_SIZE}  runtime=${FIO_RUNTIME}s"
echo "              rw=read  direct=1  ioengine=${FIO_IOENGINE}  group_reporting"
echo ""

# ── Main test loop ──────────────────────────────────────────────
for NC in "${NODE_COUNTS[@]}"; do
    SELECTED=("${ALL_NODES[@]:0:$NC}")
    NODELIST=$(IFS=,; echo "${SELECTED[*]}")

    echo "========================================"
    echo "  ${NC} node(s): ${NODELIST}"
    echo "========================================"

    for REP in $(seq 1 "$REPS"); do
        echo ""
        echo "--- Rep ${REP}/${REPS}  (${NC} nodes) ---"

        RUN_DIR="${RAW_DIR}/n${NC}_rep${REP}"
        mkdir -p "$RUN_DIR"

        # Best-effort cache drop (may fail without root)
        srun --nodes="${NC}" --ntasks="${NC}" --ntasks-per-node=1 \
            --nodelist="${NODELIST}" \
            bash -c 'sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null || true' \
            2>/dev/null || true

        START_EPOCH=$(( $(date +%s) + LEAD_TIME ))
        echo "  Sync target: epoch ${START_EPOCH}  ($(date -d "@${START_EPOCH}" +%H:%M:%S))"

        srun --nodes="${NC}" --ntasks="${NC}" --ntasks-per-node=1 \
            --nodelist="${NODELIST}" \
            bash "$WORKER" \
                "$FILE_DIR_BASE" "$START_EPOCH" "$RUN_DIR" \
                "$FIO_BS" "$FIO_NJOBS" "$FIO_IOENGINE" "$FIO_IODEPTH" "$FIO_SIZE" "$FIO_RUNTIME"

        # ── Parse results for this rep ──────────────────────────
        python3 - "$RUN_DIR" "$NC" "$REP" "$PER_NODE_CSV" "$AGGREGATE_CSV" << 'PYPARSE'
import sys, json, os, glob

run_dir, nc, rep = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
per_node_csv, aggregate_csv = sys.argv[4], sys.argv[5]

timestamps = {}
for tf in sorted(glob.glob(os.path.join(run_dir, "ts_*.txt"))):
    parts = open(tf).read().strip().split()
    timestamps[parts[0]] = (float(parts[1]), float(parts[2]))

def getp(pct_dict, target):
    for fmt in [f"{target:.6f}", f"{target:g}"]:
        if fmt in pct_dict:
            return pct_dict[fmt] / 1000  # ns -> us
    return 0.0

node_data = []
for ff in sorted(glob.glob(os.path.join(run_dir, "fio_*.json"))):
    host = os.path.basename(ff).replace("fio_", "").replace(".json", "")
    with open(ff) as f:
        d = json.load(f)
    r = d["jobs"][0]["read"]
    bw = r["bw"] / 1024          # KB/s -> MB/s
    iops = r["iops"]
    ns = r["clat_ns"]
    lat_mean = ns["mean"] / 1000
    lat_max = ns["max"] / 1000
    p = ns.get("percentile", {})
    lat_p50 = getp(p, 50)
    lat_p95 = getp(p, 95)
    lat_p99 = getp(p, 99)

    start_ts, end_ts = timestamps.get(host, (0.0, 0.0))
    node_data.append(dict(
        host=host, bw=bw, iops=iops,
        lat_mean=lat_mean, lat_p50=lat_p50, lat_p95=lat_p95,
        lat_p99=lat_p99, lat_max=lat_max,
        start_ts=start_ts, end_ts=end_ts,
    ))

    with open(per_node_csv, "a") as f:
        f.write(f"{nc},{rep},{host},{bw:.1f},{iops:.0f},{lat_mean:.1f},"
                f"{lat_p50:.1f},{lat_p95:.1f},{lat_p99:.1f},{lat_max:.1f},"
                f"{start_ts:.3f},{end_ts:.3f}\n")

    print(f"    {host}: {bw:>8.1f} MB/s  {iops:>6.0f} IOPS  "
          f"lat_mean={lat_mean:>7.0f} us  p99={lat_p99:>7.0f} us")

# Overlap check
if len(node_data) > 1:
    latest_start = max(nd["start_ts"] for nd in node_data)
    earliest_end = min(nd["end_ts"] for nd in node_data)
    overlap = max(0, earliest_end - latest_start)
    span = max(nd["end_ts"] for nd in node_data) - min(nd["start_ts"] for nd in node_data)
    overlap_frac = overlap / span if span > 0 else 0.0
else:
    overlap_frac = 1.0

agg_bw = sum(nd["bw"] for nd in node_data)
mean_bw = agg_bw / len(node_data) if node_data else 0.0

with open(aggregate_csv, "a") as f:
    f.write(f"{nc},{rep},{agg_bw:.1f},{mean_bw:.1f},{overlap_frac:.4f}\n")

print(f"    AGGREGATE: {agg_bw:>8.1f} MB/s  "
      f"({len(node_data)} nodes, mean {mean_bw:.1f} MB/s/node)")
overlap_msg = f"    Overlap:   {overlap_frac:.3f}"
if overlap_frac < 0.8:
    overlap_msg += "  *** POOR OVERLAP — result may not reflect simultaneous load ***"
print(overlap_msg)
PYPARSE

    done
done

echo ""
echo "============================================================"
echo "  Test complete"
echo "============================================================"
echo "Raw data:       ${RAW_DIR}"
echo "Per-node CSV:   ${PER_NODE_CSV}"
echo "Aggregate CSV:  ${AGGREGATE_CSV}"
echo "Finished:       $(date -Iseconds)"
echo ""
echo "CAVEAT: This is a production shared cluster. Other tenants may have"
echo "been using the storage backend during this test. Background load is"
echo "uncontrolled. Record the time window above when interpreting results."
echo ""
echo "Next:  python3 analysis/multinode_scaling.py ${PER_NODE_CSV} ${AGGREGATE_CSV}"
