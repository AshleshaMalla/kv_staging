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
