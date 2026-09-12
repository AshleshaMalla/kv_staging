# kv_staging

For LLM inference, at what context length does **fetching** a KV cache from a
given storage tier become faster than **recomputing** it on the GPU? The
crossover point is L\* = (KV\_bytes\_per\_token / BW - a) / b, where BW is the
tier's read bandwidth and T\_prefill(L) = a\*L + b\*L^2 models prefill cost.
Every term in that equation is directly measurable. This repo measures them.

## Hardware

Texas Tech REPACSS cluster, single GPU node:
- 4x NVIDIA H100 NVL (94 GB HBM each, two NVL pairs)
- 2x Intel Xeon Gold 6448Y
- ~503 GB DDR5 across 4 NUMA nodes
- Local NVMe, no swap
- NFSv4.2 over IPoIB to Hammerspace backend
- Slurm scheduled, no admin rights

## Usage

Run inside a Slurm GPU allocation, in order:

```bash
# 1. Record hardware state
bash scripts/hw_snapshot.sh

# 2. Measure storage bandwidth (run once per tier)
bash scripts/bw_sweep.sh /path/to/nfs/testdir hammerspace
bash scripts/bw_sweep.sh /tmp/fio_test nvme

# 3. Measure prefill latency vs context length
python scripts/prefill_sweep.py --model /path/to/model --out data/raw/

# 4. Compute crossover points and plot
python analysis/crossover.py \
    data/raw/prefill_*.json \
    data/raw/bw_hammerspace_*/summary.csv \
    data/raw/bw_nvme_*/summary.csv
```

## Current status

Nothing is measured yet. All entries in `config/measured_constants.yaml` are null.
