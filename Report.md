# Technical Report: KV Cache Fetch-vs-Recompute Tradeoffs for LLM Inference

## 1. Project Overview

### 1.1 Research Problem

Large language model (LLM) inference serving systems face a fundamental choice when resuming context: **fetch** a previously computed KV cache from storage, or **recompute** it via a GPU prefill pass. Fetching avoids redundant GPU work but depends on storage bandwidth; recomputing avoids I/O but burns GPU cycles that scale quadratically with context length. The optimal strategy depends on context length, storage bandwidth, and the GPU's prefill speed — all of which vary across deployments and over time.

### 1.2 Motivation and Real-World Relevance

KV cache reuse is a critical optimization for LLM serving at scale. Systems such as ObjectCache (arXiv:2605.22850), CacheGen, LMCache, Mooncake, and SGLang HiCache all implement some form of external KV cache storage and retrieval. These systems must decide, per request, whether to fetch cached KV states or recompute them. Making this decision correctly requires knowing the **crossover point**: the context length above which recomputation becomes cheaper than fetching. This crossover depends on hardware-specific constants that are rarely measured end-to-end, and the interaction between storage variability and GPU queueing has not been empirically characterized.

### 1.3 Main Research Question

**At what context length does fetching a KV cache from storage become faster than recomputing it on the GPU?**

The crossover length L\* is determined by:

```
L* = (T_fetch_per_tok − a) / b
```

where `T_fetch_per_tok = S/BW + H + M` is the per-token fetch cost, and `T_prefill(L) = a·L + b·L²` models prefill cost. Every term in both expressions — storage bandwidth (BW), host-to-GPU transfer cost (H), metadata overhead (M), KV bytes per token (S), and the prefill coefficients (a, b) — is directly measurable. This repository measures them.

### 1.4 Project Objectives

1. Measure all inputs to the crossover formula on production-class hardware (H100 NVL GPUs with Hammerspace NFS).
2. Characterize storage bandwidth variability across operating regimes.
3. Quantify how the KV cache loading mechanism (naive vs. optimized pipeline) affects fetch cost.
4. Determine whether GPU prefill serializes under concurrency and how this affects the crossover.
5. Measure multi-GPU scaling to determine node-level fetch-vs-recompute capacity.
6. Test whether storage degradation and compute contention compound (storage × compute coupling).

### 1.5 Project Status Summary

This is a measurement study. The hardware measurements and crossover formula are the primary deliverables. Four simulation-derived candidate findings (metastability, super-additivity-as-coupling, sub-additivity-as-mechanism, and the contingent sign of storage × compute coupling) were tested with controls and all reduced to textbook queueing behavior or modeling assumptions. No simulation finding survived controlled testing. What remains is the hardware measurements and the crossover formula.

---

## 2. System and Experimental Environment

### 2.1 Cluster

Texas Tech University REPACSS cluster, managed by Slurm.

- **GPU partition (`h100`):** 8 nodes, each with 4× NVIDIA H100 NVL GPUs.
- **CPU partition (`zen4`):** 110 AMD EPYC nodes, 256 cores each (used as storage aggressor nodes in contention experiments).

### 2.2 GPU Node Configuration

All measurements were performed on single GPU nodes from the `h100` partition.

| Component | Specification |
|-----------|---------------|
| GPUs | 4× NVIDIA H100 NVL (94 GB HBM3 each, two NVL pairs) |
| CPUs | 2× Intel Xeon Gold 6448Y |
| Host DRAM | ~503 GiB DDR5 across 4 NUMA nodes |
| PCIe | Gen5 x16 per GPU (~40–60 GB/s measured pinned H2D) |
| GPU TDP | 400W (PCIe form factor; lower than H100 SXM at 700W) |
| SM clocks (sustained) | 960–1035 MHz under continuous compute load at 400W TDP |
| SM clocks (max boost) | 1785 MHz (transient peak only) |

**Clock note:** An early measurement campaign (2026-09-12) discovered all GPUs had SM clocks locked at 345 MHz (the minimum supported frequency). This was resolved by cluster administrators by 2026-09-15. All authoritative measurements use boosted clocks (sustained 960–1035 MHz under load), confirmed by 200ms-interval nvidia-smi clock logging throughout runs.

Hardware provenance is captured by `scripts/hw_snapshot.sh` and stored in `data/hw/`.

### 2.3 Storage Architecture

| Storage | Details |
|---------|---------|
| Network filesystem | Hammerspace NFS (NFSv4.2 over IPoIB), mounted at `/mnt/REPACSS` |
| NFS mount options | `nconnect=4` |
| Local storage | NVMe (not yet benchmarked — OOM on test file creation) |
| Host DRAM tier | Not yet benchmarked |

### 2.4 Software Stack

| Software | Version/Details |
|----------|----------------|
| Conda environment | `m1` (activated with `PYTHONNOUSERSITE=1` to avoid conflicting `~/.local` packages) |
| vLLM | 0.29.0 (authoritative inference engine) |
| FlashAttention | v3 (used by vLLM for attention computation) |
| PyTorch | (version captured in `data/hw/*/env.txt`) |
| CUDA | 12.9.1 (Spack-installed; not conda env's CUDA 13) |
| fio | User-installed at `~/opt/bin/fio` (posixaio engine; libaio unavailable on h100 nodes) |
| Model | Llama-3.1-8B (bfloat16), path: `/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B` |

### 2.5 Model Parameters

| Parameter | Value |
|-----------|-------|
| Model | Llama-3.1-8B |
| Precision | bfloat16 |
| Layers | 32 |
| KV heads | 8 |
| Head dimension | 128 |
| KV bytes per token | 131,072 bytes (128 KiB) = 2 × 32 × 8 × 128 × 2 |
| Max context | 131,072 tokens (128K) |
| HBM at 128K context | 63.3 GiB of 93 GiB available (fits on 1× H100 NVL) |

---

## 3. Methodology and System Design

### 3.1 Overall Approach

The project follows a **measure-then-model** methodology. Rather than building a simulation from assumed parameters, every constant in the crossover formula is measured directly on the target hardware. The crossover analysis combines these measurements analytically.

The workflow proceeds in stages:

1. **Hardware characterization:** GPU topology, NUMA layout, memory, storage mounts.
2. **Storage bandwidth measurement:** fio sweeps across parallelism levels, time series, multi-node scaling, contention response.
3. **GPU prefill characterization:** Latency vs. context length under controlled conditions, fitting T(L) = a·L + b·L².
4. **KV cache fetch pipeline measurement:** End-to-end cost of reading KV data from NFS into GPU memory.
5. **Crossover computation:** Combining measurements via the analytical formula.
6. **Concurrency characterization:** Measuring Q(N) — how wall time scales with concurrent requests.
7. **Multi-GPU scaling:** Measuring aggregate throughput at 1–4 GPUs to determine node-level crossover.
8. **Simulation experiments** (subsequently withdrawn): Testing whether storage degradation and compute contention compound beyond independent effects.

### 3.2 Crossover Formula

The per-token cost of fetching a KV cache is:

```
T_fetch_per_tok = S/BW + H + M
```

where:
- **S** = 131,072 bytes/token (KV cache size per token)
- **BW** = storage read bandwidth in bytes/sec (measured, varies by operating regime)
- **H** = 2.38 × 10⁻⁶ s/token (PCIe host-to-device transfer, measured)
- **M** = 6.1 × 10⁻⁹ s/token (NFS metadata, amortized over chunk size)

The cost of recomputing a prefill of length L is:

```
T_recompute(L) = a·L + b·L²
```

where:
- **a** = 3.5158 × 10⁻⁵ s/token (linear coefficient, vLLM + FlashAttn v3, clock-logged)
- **b** = 5.8979 × 10⁻¹⁰ s/token² (quadratic coefficient, same conditions)

Setting `T_fetch_per_tok = a + b·L` (the per-token recompute marginal cost at length L) and solving:

```
L* = (T_fetch_per_tok − a) / b
```

**Internal consistency check (2026-09-18 audit):** At 2,918 MB/s (the implied bandwidth of the end-to-end fetch measurement), S/BW + H + M = 44.9 + 2.38 + 0.006 = 47.3 µs/tok, matching the measured full-pipeline fetch cost exactly.

### 3.3 Concurrency Model

On a single GPU, all resources — storage link, PCIe link, and GPU compute — are shared among N concurrent requests. The key measurement is:

```
Q(N) = N × T_service(1)     (slope 0.98–1.02)
```

Prefill serializes completely with no measurable batching benefit at production context lengths. This means L\* is **invariant in N** on a single GPU: both fetch and recompute scale by N, canceling in the crossover equation.

Across a node's GPUs, each GPU has its own compute pipeline and PCIe link, while the node's storage ceiling is shared. The multi-GPU fetch model is:

```
T_fetch_per_tok(G) = G × S/BW_node + H + M
```

### 3.4 Provenance System

All measured constants are recorded in `config/measured_constants.yaml` with mandatory fields: `value`, `units`, `source` (path to raw data), `script` (producing script), and `date`. A provenance test (`tests/test_provenance.py`) enforces that every non-null measurement has backing raw data, a script path, and a parseable date. This system was implemented after a stale-coefficient contamination incident (the retracted 27× claim) where code and record had decoupled.

### 3.5 Decision Gates

Methodological gates applied during the project (documented in `docs/GATES.md`):

1. **Seeded replication at n ≥ 5:** Simulation findings require replication across ≥5 seeds before reporting.
2. **Baseline comparison before claiming a mechanism:** Candidate findings must be compared against the simplest model that could produce them.
3. **No-episode controls:** Simulation runs with degradation must be compared against no-episode controls at the same load.
4. **Provenance coupling:** Scripts read constants from the YAML and write results back with provenance.
5. **Shared-resource audit before modeling concurrency:** Every shared resource must be explicitly divided among concurrent requests.

---

## 4. Implementation

### 4.1 Directory Structure

```
kv_staging/
├── config/
│   └── measured_constants.yaml    # Single source of truth for all measurements
├── scripts/                       # Measurement scripts (bash + python)
│   ├── hw_snapshot.sh             # Hardware state capture
│   ├── bw_sweep.sh                # Single-node bandwidth sweep (fio)
│   ├── bw_multinode.sh            # Multi-node bandwidth scaling
│   ├── bw_timeseries.sh           # Bandwidth stability over time
│   ├── bw_engine_compare.sh       # fio engine confound check (posixaio vs libaio)
│   ├── contention_experiment.sh   # Controlled contention with aggressor nodes
│   ├── prefill_sweep.py           # Prefill latency (HF transformers, superseded)
│   ├── prefill_sweep_vllm.py      # Prefill latency (vLLM, early)
│   ├── prefill_config_sweep.py    # vLLM configuration sweep
│   ├── prefill_loaded_sweep_v2.py # Authoritative loaded prefill sweep (vLLM)
│   ├── measure_fetch_overhead.py  # KV fetch pipeline v1 (superseded)
│   ├── measure_fetch_overhead_v2.py # Authoritative KV fetch pipeline (v2)
│   ├── diagnose_batching.py       # vLLM batching behavior diagnostic
│   ├── qn_extended_sweep.py       # Q(N) sweep to N=64
│   ├── qn_batch_confound.py       # Q(N) batch-budget confound check
│   ├── stability_seeded.py        # Saturation zone seed sweep
│   ├── overlap_sweep.py           # Transfer-compute overlap sweep
│   ├── contingent_sign_test.py    # Contingent sign controls (seeding + baseline)
│   ├── multigpu_prefill_scaling.py # Multi-GPU orchestrator
│   ├── multigpu_worker.py         # Single-GPU worker for multi-GPU experiment
│   ├── multigpu_analyze.py        # Multi-GPU results analysis
│   └── *.sbatch                   # Slurm batch scripts for each experiment
├── analysis/                      # Analysis and simulation code
│   ├── crossover.py               # Crossover point computation + plots
│   ├── stallopt.py                # ObjectCache policy reproduction
│   ├── queue_compute.py           # Fixed-N queue-aware compute window simulation
│   ├── queue_feedback.py          # Open-loop concurrency feedback simulation
│   ├── corrections.py             # Trace sensitivity / correction analysis
│   ├── bw_variance.py             # Bandwidth time series analysis
│   ├── multinode_scaling.py       # Multi-node scaling analysis
│   └── contention_response.py     # Contention experiment analysis
├── data/
│   ├── raw/                       # Raw measurement outputs (JSON, CSV, .out)
│   ├── hw/                        # Hardware snapshots
│   └── *.png                      # Generated plots
├── docs/
│   ├── STATUS.md                  # Current project status
│   ├── DECISIONS.md               # Decision log (detailed)
│   └── GATES.md                   # Decision gates
└── tests/
    └── test_provenance.py         # YAML provenance enforcement
```

### 4.2 Key Components

**`config/measured_constants.yaml`** — Central record of every measured number. Each leaf entry includes the value with explicit units, source data path, producing script, measurement date, and explanatory notes. Superseded values are preserved with `SUPERSEDED` or `WITHDRAWN` annotations and the reason.

**`scripts/prefill_loaded_sweep_v2.py`** — The authoritative prefill measurement script. Loads vLLM with specified configuration (`enforce_eager`, `enable_prefix_caching=False`), sweeps context lengths × concurrency levels × `max_num_batched_tokens` values, records per-request completion times (first/median/last), logs SM clocks at 200ms throughout, and performs crossover analysis reading constants from the YAML. Writes fitted N=1 coefficients back to the YAML for provenance coupling.

**`scripts/measure_fetch_overhead_v2.py`** — Measures the full KV cache fetch pipeline: `os.preadv` with `O_DIRECT` into a pinned CUDA host buffer, then `pinned.to('cuda', non_blocking=True)` with synchronize. Reports per-token costs at each stage and end-to-end, with implied bandwidth sanity checks against fio measurements.

**`scripts/qn_extended_sweep.py`** — Extended Q(N) measurement from N=1 to N=64. Uses both offline `LLM.generate()` and `AsyncLLMEngine` phases. Fits linear, quadratic, and power-law models to determine whether scaling is truly linear. Reports per-length slopes and cross-node reproducibility.

**`scripts/multigpu_worker.py`** / **`scripts/multigpu_prefill_scaling.sbatch`** — Multi-GPU experiment. The sbatch script orchestrates 1–4 independent vLLM processes on separate GPUs (pinned via `CUDA_VISIBLE_DEVICES`, no tensor parallelism). Each worker measures N=1 prefill service time. A pre-warm step runs a single-GPU pass first to populate the FlashInfer JIT sampling kernel cache, preventing race conditions during simultaneous builds. `scripts/multigpu_analyze.py` computes aggregate throughput, scaling efficiency, and the node-level fetch-vs-recompute crossover.

**`analysis/stallopt.py`** — Reproduces the ObjectCache bandwidth allocation policies (Equal, Stall-opt, Calibrated Stall-opt, KV-prop, BW-prop) and the layer-pipelined TTFT model from Zhu et al. (arXiv:2605.22850). Uses the paper's Table A8 request characterization data and compute windows.

**`analysis/queue_compute.py`** — Fixed-N simulation testing whether GPU queueing and storage bandwidth variance compound. Uses a four-cell experimental design: {isolated, queued} × {constant BW, bimodal BW}. The scheduler stays naive (allocates from r\*_iso); only the physics model uses the actual queued compute window. Uses the project's measured vLLM prefill coefficients, not the paper's A100 numbers.

**`analysis/queue_feedback.py`** — Open-loop Poisson arrival simulation where concurrency N is an output, not a parameter. Tests the concurrency-feedback channel: storage degrades → transfers stretch → occupancy rises → N rises → compute window stretches. Logs N(t) vs. bandwidth and measures metastability (whether the queue outlives a degradation episode).

### 4.3 Experiment Launch

All GPU experiments are launched via Slurm on the `h100` partition. Each experiment has a paired `.sbatch` file and a Python script. The sbatch files handle environment setup:

1. Activate the `m1` conda environment.
2. Set `PYTHONNOUSERSITE=1` to avoid conflicting user-site packages.
3. Set `CUDA_HOME` to the Spack-installed CUDA 12.9.1 (not conda's CUDA 13, which causes FlashInfer header incompatibility).
4. Set appropriate `LD_LIBRARY_PATH` and `LD_PRELOAD` for the conda environment's libraries.
5. Request sufficient memory (`--mem=200G` or `--mem=500G` for multi-GPU runs; omitting `--mem` defaults to `DefMemPerCPU × CPUs`, which caused OOM at only 8 GiB for 4-GPU runs).

Typical invocation:
```bash
sbatch scripts/prefill_loaded_v2.sbatch
```

---

## 5. Experimental Methodology

### 5.1 Storage Bandwidth Measurement

**Tool:** fio (user-installed), posixaio engine, 128K block size, `--direct=1` (bypass page cache), `iodepth=16`.

**Method:** Sequential read sweeps across 1–16 parallel jobs (`numjobs`), with pre-created 1 GB (initial sweep) or 8 GB (engine comparison) test files. Each configuration runs for 30 seconds time-based. Results parsed from fio JSON output.

**Aggregation:** Bandwidth is reported per operating regime (not averaged), because the storage system exhibits distinct states:

| Regime | How Identified | BW (MB/s) |
|--------|---------------|-----------|
| Degraded | Spontaneous event during multi-node run | 1,400 |
| Quiescent | 103-minute baseline (305 samples, CV=0.65%) | 2,588 |
| Quiet evening | During contention experiment (Saturday evening) | 3,806 |
| Peak single-stream | posixaio, 1 job, 8G files | 4,993 |
| Peak multi-stream | posixaio, 2 jobs, 8G files | 5,998 |

**Key finding:** Bandwidth is flat across 1–16 streams (4.4–4.8 GB/s in the initial sweep). This is a per-node ceiling, not a per-stream limit. This measurement was critical in correcting the loaded crossover model (which had incorrectly given each concurrent fetch the full node bandwidth).

### 5.2 Prefill Latency Measurement

**Authoritative measurement:** `scripts/prefill_loaded_sweep_v2.py` using vLLM 0.29.0 with FlashAttention v3.

**vLLM configuration:**
- `enforce_eager=True` (no CUDA graph compilation)
- `enable_prefix_caching=False`
- `tensor_parallel_size=1`
- `gpu_memory_utilization=0.90`
- `max_model_len=131072`
- `max_num_batched_tokens` swept: 16384, 32768, 65536

**Protocol:**
1. Load model, record vocab size and KV bytes/token.
2. Start nvidia-smi clock logger at 200ms intervals.
3. For each (length, concurrency, batch_token_config):
   - Generate random token IDs within vocab range (distinct per request).
   - Warmup: 1 iteration (discarded).
   - Timed: 3 repetitions, each recording wall time and per-request completion times from vLLM's `RequestOutput.metrics`.
4. Fit T = a·L + b·L² to N=1 wall times.
5. Write fitted coefficients back to `config/measured_constants.yaml`.

**Context lengths swept:** 2048, 4096, 8192, 16384, 32768, 65536 tokens.

**Clock state:** Sustained SM clocks of 960–1035 MHz confirmed by continuous 200ms nvidia-smi logging throughout the run. The max boost of 1785 MHz is a transient peak only; the H100 NVL power-limits to this range at 400W TDP.

### 5.3 KV Cache Fetch Pipeline Measurement

**Script:** `scripts/measure_fetch_overhead_v2.py`

**Pipeline measured:**
1. **T_read_into_pinned:** `os.preadv` with `O_DIRECT` flag directly into a pinned CUDA host tensor (`torch.empty(..., pin_memory=True)`). Uses 128 MiB chunks. `O_DIRECT` bypasses the kernel page cache, matching fio `--direct=1`.
2. **H_to_gpu:** `pinned_tensor.to('cuda', non_blocking=True)` followed by `torch.cuda.synchronize()`.
3. **Full pipeline:** Both steps measured end-to-end.

**Protocol:** For each context length (4096, 8192, 16384, 32768, 65536):
1. Create raw binary test file on NFS of appropriate size.
2. Before each repetition, evict page cache via `posix_fadvise(DONTNEED)`.
3. Measure 15 repetitions; report medians.

**Sanity checks:** Implied read bandwidth compared against fio-measured range (2.6–6.0 GB/s). Implied H2D bandwidth compared against expected PCIe Gen5 range (40–60 GB/s).

**Mechanism progression (quantifies loading-path impact):**

| Mechanism | Per-token cost | Relative to prefill `a` |
|-----------|---------------|------------------------|
| Naive (torch.load + pageable .cuda()) | 135.3 µs/tok | 3.9× |
| v1 (bytearray + pinned H2D) | 67.7 µs/tok | 2.0× |
| v2 (pinned + O_DIRECT, single pipeline) | 47.3 µs/tok | 1.35× |

The 2.9× reduction from naive to v2 demonstrates that the loading mechanism matters more than storage bandwidth for KV cache-sized files.

### 5.4 Q(N) Concurrency Scaling

**Goal:** Determine whether GPU prefill under concurrency exhibits batching benefits (slope < 1) or pure serialization (slope ≈ 1).

**Method:** Submit N identical-length prompts simultaneously via `LLM.generate()`. Measure wall time for all N to complete. Compute ratio = wall(N) / wall(1).

**Sweep parameters:**
- Context lengths: 4096, 16384, 32768
- Concurrency levels: 1, 2, 4, 8, 16, 24, 32, 48, 64
- `max_num_batched_tokens`: 8K, 16K, 32K, 65K, 131K (5 values)
- Nodes: rpg-93-1, rpg-93-3, rpg-93-5 (3 nodes, 2 clock states)
- Repetitions: 3 per configuration, 1 warmup

**Fit models:** Linear (ratio = 1 + slope·(N−1)), quadratic, and power law. Per-length slopes computed separately.

### 5.5 Multi-GPU Scaling

**Goal:** Measure whether prefill throughput scales linearly with the number of active GPUs on a node, and compute the node-level fetch-vs-recompute crossover.

**Method:** Independent vLLM processes on 1–4 GPUs (no tensor parallelism, `CUDA_VISIBLE_DEVICES` pinned). Each process measures N=1 prefill service time. GPU clocks and power logged at 100ms.

**Parameters:**
- Context lengths: 16384, 32768
- `max_model_len=33000`, `gpu_mem=0.85`, `max_num_batched_tokens=32768`
- 5 reps per GPU per length, 1 warmup
- 3 independent runs across 2 nodes (rpg-93-6 ×2, rpg-93-3 ×1): jobs 160492, 160494, 160495

**Pre-warm step:** A single-GPU run executes first to populate the FlashInfer JIT sampling kernel cache, preventing cache-race conditions from simultaneous ninja builds.

### 5.6 Metrics

| Metric | Definition | Units |
|--------|-----------|-------|
| T_prefill(L) | Wall-clock time for vLLM to complete prefill of L tokens | seconds |
| a, b | Coefficients in T = a·L + b·L² fit | s/token, s/token² |
| BW | fio sequential read throughput with direct I/O | MB/s |
| T_read_into_pinned | Median time to read KV cache from NFS into pinned host buffer | µs/token |
| H | Median time for pinned-to-GPU transfer | µs/token |
| T_fetch_pipeline | End-to-end read + H2D | µs/token |
| Q(N) slope | (wall(N)/wall(1) − 1) / (N − 1) | dimensionless |
| Scaling efficiency | Aggregate throughput at G GPUs / (G × single-GPU throughput) | percentage |
| L\* | Crossover context length where fetch = recompute | tokens |

---

## 6. Experiments and Results

### 6.1 Storage Bandwidth Characterization

**Purpose:** Establish the storage bandwidth available for KV cache fetching and characterize its variability.

**Experiments conducted:**
- Single-node fio sweep (1–16 streams): `scripts/bw_sweep.sh` → `data/raw/bw_shared_nfs_2026-09-12T11:15:22-05:00/`
- Engine comparison (posixaio, larger files): `scripts/bw_engine_compare.sh` → `data/raw/bw_posixaio_sweep_20260913T202840Z/`
- Multi-node scaling (1/2/4 nodes): `scripts/bw_multinode.sh` → `data/raw/bw_multinode_20260913T195637Z/`
- Time series (103 min): `scripts/bw_timeseries.sh` → `data/raw/bw_timeseries_20260913T211736Z/`
- Contention experiment (aggressor injection): `scripts/contention_experiment.sh` → `data/raw/contention_20260913T234348Z/`

**Results:**

| Measurement | Result |
|-------------|--------|
| Single-stream peak | 4,993 MB/s (posixaio, 8G files) |
| Multi-stream peak | 5,998 MB/s (2 jobs) |
| Parallelism scaling | Flat across 1–16 streams (per-node ceiling) |
| Quiescent stability | CV = 0.65% over 103 minutes (305 samples) |
| Multi-node scaling | ~90% efficient at 4 nodes |
| Spontaneous degradation | 2.3× drop to ~1,400 MB/s (observed during multi-node run, suspected external tenant load) |
| Aggressor injection | Read-only aggressor nodes did not reproduce the spontaneous degradation |

**Interpretation:** Hammerspace NFS provides 1.4–6.0 GB/s depending on the operating regime. The bandwidth is a per-node ceiling (no parallelism benefit), bimodal (discrete operating states rather than continuous variation), and episodic (spontaneous degradation events occur but are not reproducible on demand). The quiescent regime is highly stable.

### 6.2 Prefill Latency and the T(L) = a·L + b·L² Fit

**Purpose:** Measure GPU prefill cost as a function of context length to establish the recompute side of the crossover formula.

**Setup:** vLLM 0.29.0 + FlashAttention v3, enforce_eager, Llama-3.1-8B bfloat16, H100 NVL at 400W TDP. Clock-logged throughout.

**Results:**

| Coefficient | Value | Source |
|-------------|-------|--------|
| a (linear) | 3.5158 × 10⁻⁵ s/token | vLLM loaded sweep v2 (2026-09-16) |
| b (quadratic) | 5.8979 × 10⁻¹⁰ s/token² | vLLM loaded sweep v2 (2026-09-16) |
| SM clocks | 960–1035 MHz sustained | 200ms nvidia-smi log |

**Superseded coefficients:**

| Engine | a (s/tok) | b (s/tok²) | Note |
|--------|-----------|-----------|------|
| HF transformers (clock-locked 345 MHz) | 1.3062 × 10⁻⁴ | 3.7269 × 10⁻⁹ | Superseded: wrong clock state |
| HF transformers (boosted, no FlashAttn) | 3.4551 × 10⁻⁵ | 7.5003 × 10⁻¹⁰ | Superseded: no clock log, no FlashAttn v3 |
| **vLLM + FlashAttn v3 (authoritative)** | **3.5158 × 10⁻⁵** | **5.8979 × 10⁻¹⁰** | **Clock-logged, authoritative** |

The vLLM b coefficient is 21% lower than HF's, attributable to FlashAttention v3's more efficient attention scaling.

**Scheduler artifact check:** Throughput was invariant across `max_num_batched_tokens` values 16384/32768/65536 (<2% variation), confirming saturation is hardware-bound, not a scheduler configuration artifact.

### 6.3 KV Cache Fetch Pipeline

**Purpose:** Measure the end-to-end cost of getting KV cache data from NFS into GPU HBM, quantify each stage, and demonstrate how the loading mechanism affects the total cost.

**Setup:** Test files on Hammerspace NFS, O_DIRECT reads into pinned CUDA host buffers, 15-rep medians.

**Results (v2, authoritative, steady-state at L ≥ 8192):**

| Stage | Per-token cost | Implied bandwidth |
|-------|---------------|-------------------|
| T_read_into_pinned (NFS → host) | 44.9 µs/tok | 2.92 GB/s |
| H_to_gpu (host → GPU, pinned) | 2.38 µs/tok | 55.3 GB/s |
| Full pipeline | 47.3 µs/tok | 2.77 GB/s |
| Metadata (open/stat/close) | ~0.006 µs/tok | — (amortized, negligible) |
| Pin allocation (one-time) | 12 µs | — (amortized to zero with buffer pool) |

The read bandwidth (2.92 GB/s) falls within the fio-measured range (2.6–6.0 GB/s). The H2D bandwidth (55.3 GB/s) is consistent with PCIe Gen5 x16.

The v2 full pipeline cost (47.3 µs/tok) is **1.35× the vLLM prefill linear coefficient** (35.16 µs/tok), meaning that at the marginal level, fetching a single token's KV cache is only 35% more expensive than the linear component of computing it.

### 6.4 Crossover Points

**Purpose:** Determine at each bandwidth state the context length where fetching becomes more expensive than recomputing.

**Formula:** `L* = (S/BW + H + M − a) / b`, with all inputs measured.

**Results (N=1, single GPU):**

| BW state | BW (MB/s) | T_fetch (µs/tok) | L\* (tokens) | Status |
|----------|-----------|-------------------|-------------|--------|
| Degraded | 1,400 | 96.01 | 103,175 | Crossover within 128K context |
| Quiescent | 2,588 | 53.03 | 30,307 | Crossover within 128K context |
| Quiet evening | 3,806 | 36.82 | 2,826 | **Fragile** (margin 1.66 µs) |
| Peak 1-stream | 4,993 | 28.64 | — | Fetch always wins |

**H-sensitivity (row 3):** The quiet-evening crossover exists only because of the H2D overhead term (H = 2.38 µs/tok). The critical threshold is H = 0.71 µs/tok; measured H is 3.3× this threshold. Without H (payload-only formula), there is no crossover at 3,806 MB/s. The exact L\* is meaningful only to ±50 tokens given H variation.

**Worked example (defensible claim):** At L = 30K tokens:
- At 1,400 MB/s: T_recompute = 1,585.6 ms vs T_fetch = 2,880 ms → **recompute wins**
- At 3,806 MB/s: T_recompute = 1,585.6 ms vs T_fetch = 1,105 ms → **fetch wins** (1.44× advantage)

This holds under both the full formula and the payload-only variant.

### 6.5 Q(N) Prefill Serialization

**Purpose:** Determine whether GPU prefill batching provides any throughput benefit under concurrency, which affects the crossover under concurrent load and the accuracy of queueing simulations.

**Setup:** 3 nodes (rpg-93-1/3/5), 5 `max_num_batched_tokens` values (8K–131K), 3 context lengths (4K/16K/32K), N up to 64, 2 clock states (825–885 and 945–1005 MHz).

**Results:**

| Fit | Value |
|-----|-------|
| Global linear slope | 0.998 (RMSE: reported in results.json) |
| Per-length slopes | L=4K: 1.107, L=16K: 0.994, L=32K: 0.982 |
| Cross-node delta | < 0.7% |
| Batch-budget dependence | None (variation ~0.02, within noise) |
| Quadratic curvature | None significant |

**Interpretation:** Prefill serializes completely. Q(N) = N × T_service(1) with slope 0.98–1.02. There is no measurable batching benefit at production context lengths. The slightly super-linear slope at L=4K (1.107) reflects short-context per-step overhead. The relationship is linear to N=64 with no curvature (β = 0.998 in the power-law fit).

**Implication for the crossover:** On a single GPU, since both fetch time and recompute time scale by N under concurrency, the crossover L\* is invariant in N. The loaded crossover table reduces to the N=1 table.

**Correction:** The original Q(N) slope of 0.85 was a cold-start artifact from a single unreplicated measurement (`diagnose_batching.py`, job 155248) with no warmup. The N=1 baseline included JIT and memory allocation costs, inflating the denominator.

### 6.6 Multi-GPU Scaling

**Purpose:** Measure whether per-GPU prefill throughput degrades when multiple GPUs are active simultaneously, and compute the node-level fetch-vs-recompute crossover.

**Setup:** Independent vLLM processes on 1–4 GPUs. 3 independent runs across 2 nodes (rpg-93-6 ×2, rpg-93-3 ×1). Jobs 160492, 160494, 160495.

**Scaling efficiency (range across 3 runs):**

| G (GPUs active) | L=16K efficiency | L=32K efficiency |
|-----------------|-----------------|-----------------|
| 1 | 100% | 100% |
| 2 | 98.4–101.7% | 98.6–101.4% |
| 3 | 97.4–99.9% | 97.8–100.1% |
| 4 | 98.3–98.6% | 98.3–98.9% |

No per-GPU clock degradation or power throttling at G=4. Mean power per active GPU: 82–132W (well below 400W TDP). Host RSS: ~1.5 GiB per vLLM worker (model weights are GPU-resident).

**Node-level fetch-vs-recompute crossover (L=16,384):**

| BW state | G=1 | G=2 | G=3 | G=4 |
|----------|-----|-----|-----|-----|
| Degraded (1,400 MB/s) | RECOMP | RECOMP | RECOMP | RECOMP |
| Quiescent (2,588) | RECOMP | RECOMP | RECOMP | RECOMP |
| Quiet evening (3,806) | **FETCH** | RECOMP | RECOMP | RECOMP |
| Peak (4,993) | **FETCH** | RECOMP | RECOMP | RECOMP |

**Node-level fetch-vs-recompute crossover (L=32,768):**

| BW state | G=1 | G=2 | G=3 | G=4 |
|----------|-----|-----|-----|-----|
| Degraded (1,400 MB/s) | RECOMP | RECOMP | RECOMP | RECOMP |
| Quiescent (2,588) | MARGINAL (1.01×) | RECOMP | RECOMP | RECOMP |
| Quiet evening (3,806) | **FETCH** | RECOMP | RECOMP | RECOMP |
| Peak (4,993) | **FETCH** | MARGINAL (1.06×) | RECOMP | RECOMP |

**Interpretation:** With more GPUs active, the shared storage ceiling becomes the bottleneck and recompute gains. At quiet-evening and peak bandwidth — where single-GPU fetch wins — recompute wins at G ≥ 2 (L=16K) or G ≥ 2–3 (L=32K; peak L=32K G=2 is marginal at 1.06×).

**Storage share of total node KV-restoration capacity:**

| BW state | G=1 | G=2 | G=3 | G=4 |
|----------|-----|-----|-----|-----|
| Degraded | 31–35% | 18–21% | 13–16% | 10–12% |
| Quiescent | 44–50% | 29–33% | 22–25% | 17–21% |
| Quiet evening | 53–59% | 37–42% | 29–33% | 23–27% |
| Peak | 60–65% | 43–49% | 34–39% | 29–33% |

At degraded bandwidth with all 4 GPUs active, storage contributes only 10–12% of total restoration capacity.

### 6.7 Saturation Zone

**Purpose:** Determine the effective serving capacity under the measured Q(N) = N × T_service(1) serialization model, using simulation.

**Method:** 5-seed Poisson arrival-rate sweep (`scripts/stability_seeded.py`) at slope = 1.0, measuring queue stability at each load fraction of the compute-bound capacity estimate (~1.59 req/s).

**Results:**

| Load | Stability |
|------|-----------|
| 70% (1.11 req/s) | Only majority-stable load (3/5 stable, 0/5 unstable) |
| 71–83% | Transitional (mostly marginal) |
| 84–85% | 2/5 unstable seeds |

**Interpretation:** The effective capacity is materially below the 1.59 req/s compute-bound estimate. The saturation region is a zone (70–85%), not a sharp boundary. The slope correction from 0.85 to 1.0 moved the stability boundary from ~85% to ~70–76%, a qualitative shift. An assumed 15% prefill batching benefit that does not exist inflates apparent capacity by 12–24%.

### 6.8 Simulation Experiments (Withdrawn)

Four simulation-derived candidate findings were tested and subsequently withdrawn after controlled testing. They are documented here for completeness as negative results.

**6.8.1 Fixed-N Compute Window Interaction**

**Question:** Does GPU queueing compound with storage bandwidth variance (super-additivity)?

**Method:** Four-cell design {isolated, queued} × {constant BW, bimodal BW}, using ObjectCache's allocation policies. The interaction = cell4 − (cell2 + cell3 − cell1).

**Result:** Sub-additive interaction (effects partially offset, do not compound) at fixed N=4. However, this sub-additivity was produced entirely by the transfer-compute overlap model (crediting KV transfers with time spent queued behind other requests). An overlap sweep (0/31 to 31/31 layers) showed the interaction is effectively zero at overlap = 0, scaling linearly at ~14 ms per pipelined layer.

**Why withdrawn:** vLLM 0.29.0's KV connector scheduling starts reads **after** scheduler admission, not while queued. Queueing time does not provide transfer-compute overlap in the real system.

**6.8.2 Concurrency Feedback (Open-Loop)**

**Question:** When concurrency responds to bandwidth (N is an output), does the storage × compute interaction become super-additive?

**Result:** Super-additive interaction, confirmed by 30-seed bootstrap CIs excluding zero at all loads. However, a convexity baseline (fair-share queue with no layer pipeline, no overlap) reproduced and **exceeded** the interaction.

**Why withdrawn:** The super-additive feedback interaction is standard queueing convexity, not a KV-serving-specific coupling.

**6.8.3 Metastability**

**Question:** Does a degradation episode produce queue persistence that outlives the episode?

**Result:** 30-seed replication at 70% load showed median simulated/fluid drain ratio = 1.09 (indistinguishable from ordinary backlog recovery). The original 130s / 2.16× headline was seed 7 of 5 — the second-highest draw from a distribution with median 0.75×.

**Why withdrawn:** Cannot distinguish from ordinary backlog recovery. The pre-episode N → drain-time mechanism does not exist (r = −0.031).

**6.8.4 Contingent Sign of Coupling**

**Question:** Is the sign of storage × compute coupling contingent on whether concurrency can respond to bandwidth?

**Result:** Sub-additive at fixed N, super-additive under feedback — but each half reduced to a different baseline mechanism.

**Why withdrawn:** The super-additive half is queueing convexity; the sub-additive half depends on an overlap assumption that doesn't match vLLM's scheduler.

---

## 7. Key Findings

### 7.1 Measured Observations

1. **The crossover is bandwidth-state-dependent and spans the practical context range.** L\* ranges from ~103K tokens at degraded bandwidth (1,400 MB/s) to nonexistent at peak bandwidth (4,993+ MB/s), where fetch always wins. At the quiescent baseline (2,588 MB/s), L\* ≈ 30K tokens.

2. **The loading mechanism matters as much as storage bandwidth.** The fetch pipeline progression from naive (135.3 µs/tok) to optimized (47.3 µs/tok) represents a 2.9× reduction. Production systems using raw binary reads into pinned buffers with O_DIRECT achieve costs within the range of measured storage bandwidth.

3. **GPU prefill serializes completely at production context lengths.** Q(N) = N × T_service(1) with slope 0.98–1.02, measured across 3 nodes, 5 batch budgets, 3 lengths, N to 64. No batching benefit.

4. **Single-GPU concurrency does not move the crossover.** Storage, PCIe, and GPU compute are all shared among concurrent requests on a single GPU, so all scale by N and cancel.

5. **Multi-GPU scaling is near-ideal (98.3–101.7% at G=1–4).** Per-GPU service times are stable as more GPUs become active. No clock degradation or power throttling observed.

6. **Across a node's GPUs, recompute gains relative to fetch.** Each GPU has its own compute and PCIe, but the node's storage ceiling is shared. At quiet-evening and peak bandwidth — where single-GPU fetch wins — recompute wins at G ≥ 2.

7. **Hammerspace NFS bandwidth is bimodal and episodic.** Observed states range from 1,400 to ~6,000 MB/s. A spontaneous 2.3× degradation event was documented. The quiescent regime is highly stable (CV = 0.65%).

8. **The H2D overhead term (H = 2.38 µs/tok) is decisive at high bandwidth.** The quiet-evening crossover (row 3) exists only because H > 0.71 µs/tok. Without H, the payload-only formula gives no crossover at 3,806 MB/s.

9. **Effective serving capacity is 70–85% of the compute-bound estimate.** A non-existent prefill batching benefit inflates apparent capacity by 12–24%.

### 7.2 Interpretations

- The crossover formula `L* = (S/BW + H + M − a) / b` is internally consistent, audited, and composed entirely of measured inputs. It provides a direct way for serving systems to decide between fetch and recompute given their storage tier's bandwidth.

- The storage share of total node KV-restoration capacity (10–65%) quantifies how much a system can gain from a KV cache storage tier versus investing in more GPU compute. At degraded bandwidth with 4 GPUs active, storage contributes only 10–12% of capacity — the system is GPU-rich relative to its storage.

- Simulation experiments exploring storage × compute coupling did not produce findings specific to KV cache serving. All candidate interactions reduced to textbook queueing behavior or modeling assumptions once baselines or controls were applied. This is itself a finding: the interactions are real but not novel — they are standard queueing theory operating on KV-serving parameters.

### 7.3 Relationship to Research Hypothesis

The original hypothesis — that there exists a measurable crossover length L\* that depends on storage bandwidth — is confirmed. The crossover is well-characterized and varies dramatically with bandwidth state: from ~103K tokens (within the model's context limit) to non-existent (fetch always wins). The secondary investigation into whether queueing and storage variability compound produced negative results after controlled testing.

---

## 8. Reproducing the Experiments

### 8.1 Prerequisites

- Access to the REPACSS cluster `h100` partition (or equivalent H100 NVL nodes).
- Conda environment `m1` with vLLM 0.29.0, PyTorch, transformers, FlashAttention v3.
- `PYTHONNOUSERSITE=1` to avoid conflicting user-site packages.
- `CUDA_HOME` set to Spack-installed CUDA 12.9.1 (not conda's CUDA 13).
- fio installed (user-local at `~/opt/bin/fio` on REPACSS).
- Model weights at `/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B`.

### 8.2 Environment Setup

```bash
eval "$(/path/to/miniforge3/bin/conda shell.bash hook)"
conda activate m1
export PYTHONNOUSERSITE=1
export CUDA_HOME=/opt/apps/nfs/spack-1.1.0/opt/spack/linux-sapphirerapids/cuda-12.9.1-tio2hjc6xnw4bpsn37tz5fmwkz4dabp7
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

### 8.3 Core Experiments

**Hardware snapshot:**
```bash
salloc -p h100 -N1 --gpus=1 --time=00:30:00
bash scripts/hw_snapshot.sh
```
Output: `data/hw/<timestamp>_<jobid>_<hostname>/`

**Bandwidth sweep:**
```bash
salloc -p h100 -N1 --time=01:00:00
bash scripts/bw_sweep.sh /mnt/REPACSS shared_nfs
```
Output: `data/raw/bw_shared_nfs_<timestamp>/summary.csv`

**Prefill sweep (authoritative):**
```bash
sbatch scripts/prefill_loaded_v2.sbatch
```
Output: `data/raw/prefill_loaded_v2_<timestamp>.json`, GPU clock CSV

**Fetch overhead:**
```bash
sbatch scripts/fetch_overhead_v2.sbatch
```
Output: `data/raw/fetch_overhead_v2_<timestamp>.json`

**Q(N) extended sweep:**
```bash
sbatch scripts/qn_extended.sbatch
```
Output: `data/raw/qn_extended_<timestamp>/results.json`

**Q(N) batch-budget confound check:**
```bash
sbatch scripts/qn_batch_confound.sbatch
```

**Multi-GPU scaling:**
```bash
sbatch scripts/multigpu_prefill_scaling.sbatch
```
Output: `data/raw/multigpu_scaling_<timestamp>/` (per-GPU JSONs + analysis.json)

### 8.4 Analysis

**Crossover computation:**
```bash
python3 analysis/crossover.py data/raw/prefill_loaded_v2_*.json \
    data/raw/bw_shared_nfs_*/summary.csv
```
Output: `data/crossover.png`

**Multi-GPU analysis (if run separately from the sbatch):**
```bash
python3 scripts/multigpu_analyze.py --data-dir data/raw/multigpu_scaling_<timestamp>/
```

**Simulation experiments (withdrawn — for reproducibility only):**
```bash
python3 analysis/queue_compute.py          # Fixed-N four-cell
python3 analysis/queue_feedback.py         # Feedback simulation
python3 scripts/contingent_sign_test.py    # Seeded controls + baseline
python3 scripts/overlap_sweep.py           # Overlap sweep
python3 scripts/stability_seeded.py        # Saturation zone sweep
```

### 8.5 Tests

```bash
pytest tests/test_provenance.py
```
Verifies that all non-null entries in `config/measured_constants.yaml` have valid source data paths, script paths, and dates.

### 8.6 Where Results Are Stored

| Data | Location |
|------|----------|
| Raw measurements | `data/raw/<experiment>_<timestamp>/` |
| Hardware snapshots | `data/hw/<timestamp>_<jobid>_<hostname>/` |
| Generated plots | `data/*.png` |
| Measured constants | `config/measured_constants.yaml` |
| Simulation outputs | `data/raw/queue_compute_*/`, `data/raw/queue_feedback_*/` |

---

## 9. Repository Structure

```
kv_staging/
│
├── config/
│   └── measured_constants.yaml       # All measurements: value, units, source, script, date
│
├── scripts/
│   ├── hw_snapshot.sh                # GPU topology, NUMA, memory, mounts
│   ├── bw_sweep.sh                   # fio read BW vs parallelism (single node)
│   ├── bw_multinode.sh               # Synchronized fio across 1/2/4 nodes
│   ├── bw_timeseries.sh              # 103-min BW stability time series
│   ├── bw_engine_compare.sh          # posixaio vs libaio confound check
│   ├── contention_experiment.sh      # Victim + aggressor contention experiment
│   ├── contention_workload_sweep.sh  # Aggressor workload parameter sweep
│   ├── prefill_sweep.py              # HF transformers prefill (superseded)
│   ├── prefill_sweep_vllm.py         # vLLM prefill (early, superseded)
│   ├── prefill_config_sweep.py       # vLLM config sweep at fixed length
│   ├── prefill_loaded_sweep.py       # Loaded prefill v1 (superseded)
│   ├── prefill_loaded_sweep_v2.py    # ** Authoritative prefill measurement **
│   ├── measure_fetch_overhead.py     # Fetch pipeline v1 (superseded)
│   ├── measure_fetch_overhead_v2.py  # ** Authoritative fetch pipeline **
│   ├── diagnose_batching.py          # Batching diagnostic (produced superseded 0.85 slope)
│   ├── qn_extended_sweep.py          # Q(N) to N=64 (corrected measurement)
│   ├── qn_batch_confound.py          # Q(N) batch-budget confound check
│   ├── stability_seeded.py           # Saturation zone seed sweep
│   ├── stability_check.py            # Single-seed stability check (early)
│   ├── overlap_sweep.py              # Transfer-compute overlap parameter sweep
│   ├── contingent_sign_test.py       # 30-seed replication + convexity baseline
│   ├── metastability_deep.py         # 30-seed metastability replication
│   ├── rerun_experiments_bc.py       # Re-run simulations at corrected slope
│   ├── multigpu_prefill_scaling.py   # Multi-GPU orchestrator (early version)
│   ├── multigpu_worker.py            # Single-GPU vLLM worker
│   ├── multigpu_analyze.py           # Multi-GPU analysis
│   ├── multigpu_prefill_scaling.sbatch  # ** Multi-GPU Slurm script **
│   └── *.sbatch                      # Slurm job scripts for each experiment
│
├── analysis/
│   ├── crossover.py                  # Crossover computation + plotting
│   ├── stallopt.py                   # ObjectCache policy reproduction
│   ├── validate_stallopt.py          # Validation of stallopt reproduction
│   ├── queue_compute.py              # Fixed-N four-cell simulation
│   ├── queue_feedback.py             # Open-loop feedback simulation
│   ├── corrections.py                # Trace sensitivity analysis
│   ├── bw_variance.py                # BW time series analysis + plotting
│   ├── multinode_scaling.py          # Multi-node BW analysis + plotting
│   └── contention_response.py        # Contention experiment analysis
│
├── data/
│   ├── raw/                          # All raw measurement outputs
│   │   ├── bw_shared_nfs_*/          # fio bandwidth results
│   │   ├── bw_multinode_*/           # Multi-node BW results
│   │   ├── bw_timeseries_*/          # Time series BW data
│   │   ├── bw_posixaio_sweep_*/      # Engine comparison results
│   │   ├── contention_*/             # Contention experiment data
│   │   ├── prefill_loaded_v2_*.json  # Authoritative prefill data
│   │   ├── fetch_overhead_v2_*.json  # Authoritative fetch overhead data
│   │   ├── qn_extended_*/            # Q(N) extended sweep results
│   │   ├── qn_batch_confound_*/      # Q(N) confound check results
│   │   ├── multigpu_scaling_*/       # Multi-GPU scaling results
│   │   ├── queue_compute_*/          # Fixed-N simulation results
│   │   ├── queue_feedback_*/         # Feedback simulation results
│   │   └── corrections_*/            # Trace sensitivity results
│   ├── hw/                           # Hardware snapshots
│   └── *.png                         # Generated plots
│
├── docs/
│   ├── STATUS.md                     # Current project status
│   ├── DECISIONS.md                  # Detailed decision log with rationale
│   ├── GATES.md                      # Methodological decision gates
│   ├── archive/                      # Superseded planning documents
│   └── memos/                        # Research memos
│
├── tests/
│   └── test_provenance.py            # YAML provenance enforcement
│
├── README.md                         # Project overview
└── tests.md                          # Test catalog (historical)
```

---

## 10. Current Research Status

### 10.1 Completed and Demonstrated

| Item | Status |
|------|--------|
| Storage bandwidth characterization (5 operating regimes) | Measured |
| Prefill latency model (a, b coefficients, clock-logged) | Measured |
| KV fetch pipeline (3-step mechanism progression) | Measured |
| Crossover formula (audited, internally consistent) | Verified |
| Crossover points (4 bandwidth states) | Computed from measurements |
| Q(N) serialization (slope 1.0, N to 64) | Measured (3 nodes, 5 configs) |
| Multi-GPU scaling (G=1–4, 3 runs, 2 nodes) | Measured |
| Node-level fetch-vs-recompute table | Computed from measurements |
| Storage share of total node capacity | Computed from measurements |
| Saturation zone characterization | Simulated (5-seed sweep) |
| Simulation-derived findings tested with controls | All withdrawn (4/4) |
| Provenance system with automated tests | Implemented |

### 10.2 Remaining Planned Work (Identifiable from Repository)

| Item | Status | Notes |
|------|--------|-------|
| Local NVMe bandwidth | Pending | OOM on test file creation in prior attempt |
| Host DRAM (PCIe) bandwidth tier | Pending | Not attempted |
| Achieved TFLOPS at boosted clocks | Stale | Current 152 TFLOPS value measured at 345 MHz clock-lock |
| Reproduce storage degradation | Open | Controlled aggressor injection did not reproduce the spontaneous event |
| Live system validation | Open | All queue/feedback results are simulation; no live vLLM serving experiment |
| H100 SXM measurements | Open | SXM at 700W TDP would have ~1.5× higher sustained clocks |
