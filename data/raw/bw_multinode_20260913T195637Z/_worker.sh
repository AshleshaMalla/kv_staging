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
