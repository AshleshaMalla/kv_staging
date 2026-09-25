# Project status

Last updated: 2026-09-24.

## Research question

At what context length does fetching a KV cache from storage become faster than
recomputing it on the GPU? Measured for Llama-3.1-8B on H100 NVL with
Hammerspace NFS.

## What is measured and stands

### Storage bandwidth (2026-09-13)

NFS bandwidth across observed operating regimes:

| Condition | Bandwidth (MB/s) |
|-----------|-------------------|
| Degraded (external tenant load) | 1,400 |
| Quiescent (103-min baseline, CV=0.65%) | 2,588 |
| Quiet evening | 3,806 |
| Peak single-stream | 4,993 |
| Peak multi-stream | 5,998 |

Bandwidth is flat across 1-16 streams (no parallelism benefit). Multi-node
scaling is ~90% efficient at 4 nodes. A spontaneous 2.3x degradation event was
observed and documented. (DECISIONS.md 2026-09-12)

### Prefill latency (2026-09-16)

T(L) = a\*L + b\*L^2 for vLLM 0.29.0 + FlashAttention v3 on H100 NVL at
sustained SM clocks of 960-1035 MHz (400W TDP power-limited):

- **a** = 3.5158e-5 s/token (linear term)
- **b** = 5.8979e-10 s/token^2 (quadratic term)

Clock-logged throughout the run. Earlier HF-based coefficients
(a=3.4551e-5, b=7.5003e-10) are superseded. (DECISIONS.md 2026-09-18)

### Fetch overhead (2026-09-15)

End-to-end KV cache fetch cost (NFS read into pinned CUDA buffer + H2D
transfer): **47.3 us/token** (implied 2.92 GB/s disk-to-host, 55.3 GB/s H2D).
This is 1.35x the prefill linear coefficient.

Mechanism progression quantifying how much the loading path matters:
- Naive (torch.load + pageable .cuda()): 135.3 us/tok
- v1 (bytearray + pinned H2D): 67.7 us/tok
- v2 (pinned + O\_DIRECT, single pipeline): 47.3 us/tok

(DECISIONS.md 2026-09-12)

### Crossover formula (audited 2026-09-18)

T\_fetch\_per\_tok = S/BW + H + M, where:
- S = 131,072 bytes/tok
- H = 2.38e-6 s/tok (PCIe H2D)
- M = 6.1e-9 s/tok (NFS metadata, amortized)

L\* = (T\_fetch\_per\_tok - a) / b

Internal consistency check: at 2,918 MB/s (implied BW of end-to-end
measurement), S/BW + H + M = 44.9 + 2.38 + 0.006 = 47.3 us/tok, matching
the measured pipeline exactly.

Crossover points:

| BW state | BW (MB/s) | T\_fetch (us/tok) | L\* (tokens) | Status |
|----------|-----------|-------------------|-------------|--------|
| Degraded | 1,400 | 96.01 | 103,175 | Crossover |
| Quiescent | 2,588 | 53.03 | 30,307 | Crossover |
| Quiet evening | 3,806 | 36.82 | 2,826 | Fragile (margin 1.66 us) |
| Peak 1-stream | 4,993 | 28.64 | -- | Fetch always wins |

Row 3 depends on H: the crossover vanishes if H < 0.71 us/tok. Measured H =
2.38 us/tok (3.3x the critical threshold), so the crossover exists but L\* is
meaningful only to +/-50 tokens. (DECISIONS.md 2026-09-18)

On a single GPU, concurrency does not move the crossover. Storage link, PCIe
link, and GPU compute are all shared among concurrent requests, so all three
scale by N and cancel: L\* is invariant in N to within ~10 tokens (the M
term). (DECISIONS.md 2026-09-24)

### Multi-GPU scaling (2026-09-24)

Across a node's GPUs, each GPU has its own compute and PCIe link, while the
node's storage ceiling is shared. With more GPUs busy, recompute gains relative
to fetch.

Measured with independent vLLM processes on 1-4 GPUs, 3 independent runs
across 2 nodes (rpg-93-6 x2, rpg-93-3 x1), 5 reps per GPU per length:

| GPUs active | L=16K efficiency | L=32K efficiency |
|-------------|-----------------|-----------------|
| 1 | 100% | 100% |
| 2 | 98.4-101.7% | 98.6-101.4% |
| 3 | 97.4-99.9% | 97.8-100.1% |
| 4 | 98.3-98.6% | 98.3-98.9% |

No per-GPU clock degradation or power throttling at G=4 (mean power 82-132W
per active GPU, well below the 400W TDP). Host RSS ~1.5 GiB per vLLM worker
process (model weights are GPU-resident).

**Fetch model.** T\_fetch\_per\_tok(G) = G \* S/BW\_node + H + M. Storage read
and H2D copy are treated as serial per token, consistent with the v2 fetch
measurement (47.3 us/tok). In a pipelined implementation, H2D would overlap
with the next chunk's storage read (each GPU has its own PCIe link; storage is
the slower stage by 19x), so pipelined throughput would be ~5% higher.

**Node-level crossover by BW state and context length.** Cells show
aggregate recompute tok/s vs node fetch tok/s. MARGINAL = within 10%.

L = 16,384:

| BW state | G=1 | G=2 | G=3 | G=4 |
|----------|-----|-----|-----|-----|
| Degraded (1,400) | 23.7K vs 10.4K **RECOMP** | 47.9K vs 10.5K **RECOMP** | 70.9K vs 10.6K **RECOMP** | 93.4K vs 10.6K **RECOMP** |
| Quiescent (2,588) | 23.7K vs 18.9K **RECOMP** | 47.9K vs 19.3K **RECOMP** | 70.9K vs 19.4K **RECOMP** | 93.4K vs 19.5K **RECOMP** |
| Quiet evening (3,806) | 23.7K vs 27.2K **FETCH** | 47.9K vs 28.1K **RECOMP** | 70.9K vs 28.4K **RECOMP** | 93.4K vs 28.5K **RECOMP** |
| Peak (4,993) | 23.7K vs 34.9K **FETCH** | 47.9K vs 36.4K **RECOMP** | 70.9K vs 37.0K **RECOMP** | 93.4K vs 37.2K **RECOMP** |

L = 32,768:

| BW state | G=1 | G=2 | G=3 | G=4 |
|----------|-----|-----|-----|-----|
| Degraded (1,400) | 19.1K vs 10.4K **RECOMP** | 38.7K vs 10.5K **RECOMP** | 57.0K vs 10.6K **RECOMP** | 75.5K vs 10.6K **RECOMP** |
| Quiescent (2,588) | 19.1K vs 18.9K **MARGINAL** (1.01x) | 38.7K vs 19.3K **RECOMP** | 57.0K vs 19.4K **RECOMP** | 75.5K vs 19.5K **RECOMP** |
| Quiet evening (3,806) | 19.1K vs 27.2K **FETCH** | 38.7K vs 28.1K **RECOMP** | 57.0K vs 28.4K **RECOMP** | 75.5K vs 28.5K **RECOMP** |
| Peak (4,993) | 19.1K vs 34.9K **FETCH** | 38.7K vs 36.4K **MARGINAL** (1.06x) | 57.0K vs 37.0K **RECOMP** | 75.5K vs 37.2K **RECOMP** |

**Storage share** (fraction of total node KV-restoration capacity provided by
the shared storage link):

| BW state | G=1 | G=2 | G=3 | G=4 |
|----------|-----|-----|-----|-----|
| Degraded | 31-35% | 18-21% | 13-16% | 10-12% |
| Quiescent | 44-50% | 29-33% | 22-25% | 17-21% |
| Quiet evening | 53-59% | 37-42% | 29-33% | 23-27% |
| Peak | 60-65% | 43-49% | 34-39% | 29-33% |

(DECISIONS.md 2026-09-24)

### Prefill serialization (2026-09-19)

Q(N) = N x T\_service(1), slope 0.98-1.02 (pure serialization, no measurable
batching benefit at production context lengths). Measured across 3 nodes
(rpg-93-1/3/5), 5 max\_num\_batched\_tokens values (8K-131K), 3 context lengths
(4K/16K/32K), N up to 64, 2 clock states (825-885 and 945-1005 MHz). Linear
with beta=0.998, no curvature. Cross-node delta <0.7%.

Per-length: 1.107 at L=4K (short-context overhead), 0.994 at L=16K, 0.982
at L=32K. (DECISIONS.md 2026-09-19)

### Saturation zone (2026-09-19)

5-seed arrival-rate sweep at slope 1.0: 70% is the only majority-stable load
(3/5 stable, 0/5 unstable). 71-83% is transitional (mostly marginal). 84-85%
has 2/5 unstable seeds. The effective capacity (1.11-1.35 req/s) is materially
below the 1.59 req/s compute-bound estimate.

An assumed 15% prefill batching benefit that does not exist inflates apparent
capacity by 12-24%. (DECISIONS.md 2026-09-19)

## What was withdrawn

- **27x boundary movement** (2026-09-18): Computed from superseded HF prefill
  coefficients that remained in a note after vLLM became authoritative.
- **0.85 Q(N) slope** (2026-09-19): Cold-start artifact from a single
  unreplicated diagnostic measurement. Corrected to 1.0.
- **Metastability (130s / 2.16x drain persistence)** (2026-09-19): 30-seed
  replication showed median 1.09x (indistinguishable from ordinary backlog
  recovery). The pre-episode-N -> drain-time mechanism does not exist (r=-0.031).
- **Contingent sign of storage x compute coupling** (2026-09-20): The
  super-additive feedback interaction is standard queueing convexity
  (reproduced by a textbook baseline). The sub-additive fixed-N interaction
  depends on an overlap assumption (transfer during queue wait) that does not
  match vLLM 0.29.0's actual KV connector scheduling (reads start after
  scheduler admission, not while queued).
- **~76% saturation point** (2026-09-19): Single-run artifact. Replaced by
  a 70-85% transitional zone.
- **Loaded crossover "N>=4 fetch always wins"** (2026-09-24): Gave each
  concurrent fetch the full node storage bandwidth and unshared PCIe. With all
  shared resources correctly divided among N concurrent requests, L\* is
  invariant in N. The single-GPU crossover table reduces to the N=1 table.

All four simulation-derived candidate findings reduced to textbook queueing
behavior or modeling assumptions once a baseline or control was applied. The
loaded crossover was a modeling error in the crossover computation itself.

## Open questions

- **vLLM WAITING\_FOR\_REMOTE\_KVS under storage degradation:** vLLM 0.29.0's
  async KV connector places requests in WAITING\_FOR\_REMOTE\_KVS until external
  tokens arrive. What happens to scheduling fairness and tail latency when
  storage degrades and these requests accumulate? Not testable without a
  working storage-disturbance method.
- **No working storage-disturbance method:** The observed 2.3x degradation was
  spontaneous (suspected external tenant load). Controlled read-only aggressor
  injection did not reproduce it. Without a reproducible trigger, experiments
  requiring bandwidth variance cannot move beyond simulation.
- **Local NVMe and host DRAM bandwidth:** Both unmeasured (NVMe OOM'd on test
  file creation; DRAM not attempted).
- **Achieved TFLOPS at boosted clocks:** Not re-measured since the 345 MHz
  clock-lock era (152 TFLOPS is stale).

## Candidate next directions

1. **Reproduce storage degradation:** Partner with REPACSS admins to identify
   the tenant-load pattern or use QoS controls to inject controlled bandwidth
   reduction.
2. **Live system validation:** Run the crossover analysis against a real vLLM
   serving deployment with KV cache transfer enabled, rather than simulation.
3. **NVMe/DRAM tiers:** Complete the pending bandwidth measurements to compute
   crossover points for faster storage tiers.
4. **SXM form factor:** The H100 NVL's 400W TDP power-limits SM clocks to
   960-1035 MHz. An H100 SXM (700W TDP) would have ~1.5x higher sustained
   clocks, shifting crossovers outward. If accessible, re-measure.
