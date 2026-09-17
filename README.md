# kv_staging

For LLM inference, at what context length does **fetching** a KV cache from
storage become faster than **recomputing** it on the GPU?

The crossover is L\* = (KV\_bytes\_per\_token / BW - a) / b, where BW is read
bandwidth and T\_prefill(L) = a\*L + b\*L^2 models prefill cost. Every term is
directly measurable. This repo measures them — and then asks what happens under
concurrent load and time-varying bandwidth.

## Hardware

Texas Tech REPACSS cluster, single GPU node:
- 4x NVIDIA H100 NVL (94 GB HBM each, two NVL pairs)
- 2x Intel Xeon Gold 6448Y, ~503 GB DDR5 across 4 NUMA nodes
- Hammerspace NFS (NFSv4.2 over IPoIB), local NVMe
- Slurm scheduled (`h100` partition)

## Model

Llama-3.1-8B (bfloat16, 128 KiB KV per token, 128K max context).
Fits on 1x H100 NVL through 128K tokens (63.3 GiB peak HBM of 93 GiB).

## What's been measured

### Storage bandwidth

NFS bandwidth was measured across multiple conditions:

| Condition | Bandwidth |
|-----------|-----------|
| Degraded (external tenant load) | ~1.4 GB/s |
| Quiescent (103-min baseline, CV=0.65%) | ~2.6 GB/s |
| Quiet evening | ~3.8 GB/s |
| Peak single-stream | ~5.0 GB/s |
| Peak multi-stream | ~6.0 GB/s |

Bandwidth is flat across 1-16 streams (no parallelism benefit). Multi-node
scaling is ~90% efficient at 4 nodes (per-node ceiling, not shared).

### Prefill latency

Prefill coefficients for T(L) = a\*L + b\*L^2, measured with vLLM 0.29.0 +
FlashAttention v3 on H100 NVL at sustained SM clocks of 960-1035 MHz (400W TDP
power-limited):

- **a** = 3.5158e-5 s/token (linear term)
- **b** = 5.8979e-10 s/token^2 (quadratic term)

Earlier measurements were invalidated by a clock-lock at 345 MHz (admin-fixed
by Sep 15). HF-based coefficients are also recorded but superseded.

### Fetch overhead

End-to-end KV cache fetch cost (NFS read into pinned CUDA buffer + H2D
transfer): **47.3 us/token** (implied 2.9 GB/s disk-to-host, 55 GB/s H2D).
This is 1.37x the prefill linear coefficient. The loading mechanism matters
more than raw storage bandwidth at KV cache sizes.

## Key findings

### Crossover points (N=1, no queueing)

| Bandwidth state | Crossover L\* |
|-----------------|---------------|
| Degraded (1.4 GB/s) | ~101K tokens |
| Quiescent (2.6 GB/s) | ~30K tokens |
| Quiet evening (3.8 GB/s) | ~2.8K tokens |
| Peak (5.0+ GB/s) | Fetch always wins |

At quiescent bandwidth, fetch wins for any context above 30K tokens. At peak
bandwidth, fetch is faster than even the linear prefill term.

### Under concurrent load

GPU prefill throughput saturates at concurrency 1. With N concurrent requests,
the last request waits ~0.85\*(N-1) service times (verified via vLLM diagnostic
job). This means:

- **N=2, degraded BW:** crossover drops to ~21K tokens
- **N>=4, any BW:** fetch always wins

Serialized GPU queueing makes recompute uncompetitive once there is any
concurrency.

### Storage x compute coupling

The sign of the interaction between bandwidth variance and compute queueing
depends on whether concurrency can respond to bandwidth:

- **Fixed N (closed workload):** sub-additive. Queueing stretches the compute
  window, which *shields* against bandwidth stalls.
- **Feedback (open-loop arrivals, N responds to BW):** super-additive.
  Bandwidth drops grow the queue, which compounds the two penalties.

Both results are from simulation using our measured prefill coefficients, not
the paper's A100 numbers. The fixed-N result is the control showing what
ObjectCache/Cake's modeling assumption produces; the feedback result shows what
happens under realistic offered load.

### Metastability

A single 60s bandwidth degradation episode builds a queue that outlives the
episode by 1.5-2.6x after bandwidth fully recovers (94s drain at 70% load,
155s at 85%). A throughput-based bandwidth estimator (like CacheGen's) sees a
healthy path during the drain window while the system is still degraded.

## Repo structure

```
scripts/          Measurement scripts (bash + python)
  hw_snapshot.sh            Hardware state capture
  bw_sweep.sh               Single-node bandwidth sweep
  bw_multinode.sh            Multi-node bandwidth scaling
  bw_timeseries.sh           Bandwidth stability over time
  bw_engine_compare.sh       fio engine confound check
  contention_experiment.sh   Controlled contention with aggressors
  prefill_sweep.py           Prefill latency (HF transformers)
  prefill_sweep_vllm.py      Prefill latency (vLLM)
  prefill_loaded_sweep_v2.py Loaded prefill with clock logging
  measure_fetch_overhead_v2.py  KV fetch pipeline measurement
  diagnose_batching.py       vLLM batching behavior diagnostic

analysis/         Analysis and simulation
  crossover.py              Crossover point computation + plots
  queue_compute.py          Fixed-N queue-aware compute window
  queue_feedback.py         Open-loop concurrency feedback simulation
  corrections.py            Trace sensitivity / correction analysis
  bw_variance.py            Bandwidth time series analysis
  multinode_scaling.py       Multi-node scaling analysis
  contention_response.py    Contention experiment analysis

config/
  measured_constants.yaml   Single source of truth for all measurements

data/
  raw/                      Raw measurement outputs
  hw/                       Hardware snapshots
  *.png                     Generated plots

docs/
  DECISIONS.md              Decision log
  GATES.md                  Pre-registered decision thresholds
```

## Usage

Run inside a Slurm GPU allocation:

```bash
# Hardware snapshot
salloc -p h100 -N1 --gpus=1 --time=00:30:00
bash scripts/hw_snapshot.sh

# Bandwidth sweep
bash scripts/bw_sweep.sh /mnt/REPACSS shared_nfs

# Prefill sweep (vLLM, authoritative)
python3 scripts/prefill_loaded_sweep_v2.py

# Crossover analysis
python3 analysis/crossover.py data/raw/prefill_loaded_v2_*.json \
    data/raw/bw_shared_nfs_*/summary.csv

# Queue-aware compute window simulation
python3 analysis/queue_compute.py

# Concurrency feedback simulation
python3 analysis/queue_feedback.py
```

## Still pending

- Local NVMe bandwidth (OOM on test file creation in prior attempt)
- Host DRAM (PCIe) bandwidth tier
- Achieved TFLOPS re-measurement at boosted clocks
- Validation against a live serving system (all queue/feedback results are simulation)
