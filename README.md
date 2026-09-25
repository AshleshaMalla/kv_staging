# kv_staging

A measurement study of KV cache fetch-vs-recompute tradeoffs for LLM inference.

**Research question.** At what context length does fetching a KV cache from
storage become faster than recomputing it on the GPU?

The crossover is L\* = (KV\_bytes\_per\_token / BW + H + M - a) / b, where BW is
read bandwidth, H is host-to-GPU transfer cost, M is metadata overhead, and
T\_prefill(L) = a\*L + b\*L^2 models prefill cost. Every term is directly
measurable. This repo measures them.

## Project status

This is a measurement study. Four simulation-derived candidate findings
(metastability, super-additivity-as-coupling, sub-additivity-as-mechanism, and
the contingent sign of storage x compute coupling) were tested with controls
and all reduced to textbook queueing behavior or modeling assumptions. No
simulation finding survived controlled testing. What remains is the hardware
measurements and the crossover formula.

See [docs/STATUS.md](docs/STATUS.md) for the full status and
[docs/DECISIONS.md](docs/DECISIONS.md) for the decision log.

### What is measured and stands

- **Storage bandwidth:** 1,400-6,000 MB/s on Hammerspace NFS, bimodal and
  episodic, with a documented 2.3x spontaneous degradation event.
- **Prefill latency:** T(L) = a\*L + b\*L^2, a = 3.5158e-5 s/tok,
  b = 5.8979e-10 s/tok^2 (vLLM 0.29.0 + FlashAttention v3, H100 NVL at
  960-1035 MHz sustained under 400W TDP).
- **Fetch overhead:** 47.3 us/tok end-to-end (NFS read into pinned CUDA
  buffer + H2D). 1.35x the prefill linear coefficient. Mechanism progression:
  135.3 -> 67.7 -> 47.3 us/tok (naive -> v1 -> v2).
- **Crossover formula:** T\_fetch = S/BW + H + M. Audited, internally
  consistent. H-sensitivity: H\_critical = 0.71 us/tok, measured H = 2.38,
  3.3x threshold.
- **Q(N) = N x T\_service(1):** Prefill serializes completely (slope 0.98-1.02).
  Measured across 3 nodes, 5 batch budgets, 3 context lengths, N to 64.
- **Saturation zone:** 70-85% transitional at slope 1.0. Effective capacity
  materially below the 1.59 req/s compute-bound estimate.

### Crossover points

| Bandwidth state | BW (MB/s) | L\* (tokens) | Status |
|-----------------|-----------|-------------|--------|
| Degraded | 1,400 | 103,175 | Crossover |
| Quiescent | 2,588 | 30,307 | Crossover |
| Quiet evening | 3,806 | 2,826 | Fragile (margin 1.66 us) |
| Peak 1-stream | 4,993 | -- | Fetch always wins |

On a single GPU, concurrency does not move the crossover. Storage link, PCIe
link, and GPU compute are all shared among concurrent requests, so all scale
together: L\* is invariant in N to within ~10 tokens (the metadata term).

Measured (2026-09-24): across a node's GPUs, each GPU has its own compute and
PCIe link, while the node's storage ceiling is shared. Scaling efficiency
98.3-101.7% at G=1-4 (3 runs, 2 nodes). At quiet evening and peak BW — where
single-GPU fetch wins — recompute wins at G >= 2 (L=16K) or G >= 2-3 (L=32K;
peak L=32K G=2 is MARGINAL at 1.06x). Quiescent L=32K G=1 is also MARGINAL
(1.01x). See docs/STATUS.md for the full per-BW, per-L, per-G table.

### What was withdrawn

- **27x boundary movement:** Computed from superseded HF prefill coefficients.
  Retracted 2026-09-18. (DECISIONS.md 2026-09-18)
- **0.85 Q(N) slope:** Cold-start artifact from a single unreplicated
  measurement. Corrected to 1.0 on 2026-09-19. (DECISIONS.md 2026-09-19)
- **Metastability (130s / 2.16x drain persistence):** 30-seed replication showed
  median 1.09x, indistinguishable from ordinary backlog recovery. Withdrawn
  2026-09-19. (DECISIONS.md 2026-09-19)
- **Contingent sign of storage x compute coupling:** Super-additive half is
  queueing convexity (reproduced by textbook baseline); sub-additive half
  depends on overlap assumption that doesn't match vLLM's scheduler. Withdrawn
  2026-09-20. (DECISIONS.md 2026-09-20)
- **Loaded crossover "N>=4 fetch always wins":** Gave each concurrent fetch the
  full node storage bandwidth (and unshared PCIe). With all shared resources
  correctly divided, L\* is invariant in N. Withdrawn 2026-09-24.
  (DECISIONS.md 2026-09-24)

## Hardware

Texas Tech REPACSS cluster, single GPU node:
- 4x NVIDIA H100 NVL (94 GB HBM each, two NVL pairs)
- 2x Intel Xeon Gold 6448Y, ~503 GB DDR5 across 4 NUMA nodes
- Hammerspace NFS (NFSv4.2 over IPoIB), local NVMe
- Slurm scheduled (`h100` partition)

## Model

Llama-3.1-8B (bfloat16, 128 KiB KV per token, 128K max context).
Fits on 1x H100 NVL through 128K tokens (63.3 GiB peak HBM of 93 GiB).

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
  qn_extended_sweep.py       Q(N) extended sweep (N to 64)
  qn_batch_confound.py       Q(N) batch-budget confound check
  stability_seeded.py        Saturation zone seed sweep
  overlap_sweep.py           Transfer-compute overlap sweep
  contingent_sign_test.py    Contingent sign controls

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
  STATUS.md                 Current project status
  DECISIONS.md              Decision log
  GATES.md                  Decision gates applied during the project
  archive/                  Superseded planning documents
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
PYTHONNOUSERSITE=1 python3 scripts/prefill_loaded_sweep_v2.py

# Crossover analysis
python3 analysis/crossover.py data/raw/prefill_loaded_v2_*.json \
    data/raw/bw_shared_nfs_*/summary.csv

# Q(N) extended sweep
PYTHONNOUSERSITE=1 python3 scripts/qn_extended_sweep.py

# Q(N) batch-budget confound check
PYTHONNOUSERSITE=1 python3 scripts/qn_batch_confound.py
```

Requires the `m1` conda environment with `PYTHONNOUSERSITE=1` for GPU scripts
(avoids a conflicting torch in `~/.local`).

## Still pending

- Local NVMe bandwidth (OOM on test file creation in prior attempt)
- Host DRAM (PCIe) bandwidth tier
- Achieved TFLOPS re-measurement at boosted clocks
- Validation against a live serving system (all queue/feedback results are simulation)
