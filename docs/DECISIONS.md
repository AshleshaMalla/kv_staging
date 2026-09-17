# Decision Log

## 2026-09-12 — Repository created

**Research question.** For LLM inference, at what context length does fetching a
KV cache from a given storage tier become faster than recomputing it on the GPU?
The crossover is L* = (KV_bytes_per_token / BW - a) / b, where BW is tier
bandwidth and T_prefill(L) = a*L + b*L^2. Every term is measurable. This
weekend's job is to measure them.

**Open items:**

- CacheGen fallback semantics: per-chunk vs sticky — how should a cache miss at
  chunk N be handled? Refetch from the top, or fall back to recompute for just
  that chunk? The answer changes the effective fetch cost.
- The unexplained 1.5 GB/s NFS single-stream figure against ~50 GB/s of
  installed ConnectX-7 bandwidth. Need to determine if this is a client tuning
  issue, NFS protocol overhead, or Hammerspace-specific.
- RAPL readability — it is unclear whether non-root users on this cluster can
  read intel-rapl energy counters. hw_snapshot.sh will probe this.
- GPU-to-NUMA topology — which GPUs are local to which NUMA nodes, and does
  pinning matter for the DMA path from NVMe or NIC to GPU?

## 2026-09-12 — SM clocks locked at minimum; prefill coefficients invalidated

**Discovery.** All 4 H100 NVL GPUs on this node have SM clocks pinned at
345 MHz — the minimum supported frequency — while max boost is 1785 MHz.
This was discovered after prefill benchmarks (both HF transformers and vLLM)
consistently achieved only ~150 TFLOP/s (~18% of the commonly cited 835 TFLOP/s
peak). A pure bf16 GEMM benchmark confirmed the ceiling: 178.5 TFLOP/s at
345 MHz, which is 95.7% of the correct theoretical peak at that clock (186.5
TFLOP/s for 132 SMs at 4096 flops/SM/clock). Clock was confirmed pinned at
345 MHz via 100ms nvidia-smi sampling during sustained compute. No throttle
reasons are active. Power limits are set to 400W (above 310W default), so power
is not the constraint. The user does not have permission to change clocks
(`nvidia-smi -rgc` fails). Evidence captured in
`data/hw/clock_lock_evidence_2026-09-12T16:07:09-05:00.txt`.

**Impact.** The measured prefill coefficients (a, b in T = a*L + b*L^2) reflect
performance at 1/5 of the GPU's clock capability. They are a lower bound on
serving-grade prefill speed. The crossover analysis that concluded "fetching
from NFS always beats recomputing" is not valid — at full clocks, prefill would
be much faster (by a large but unquantified factor; must be re-measured, not
extrapolated). The crossover analysis is blocked pending clock resolution.

**Correction.** The "835 TFLOP/s" figure used in earlier analysis was wrong for
our specific H100 NVL (132 SMs). The correct peak at 1785 MHz is 965 TFLOP/s,
derived from the H100 SXM spec (989.5 TFLOP/s at 1830 MHz, same 132 SMs).

**Action required.** Request REPACSS admins to unlock SM clocks to default boost
behavior. After clocks are resolved, re-run the full prefill sweep and crossover
analysis from scratch.

## 2026-09-17 — Queue-aware compute window: superadditivity tested at fixed N, NOT supported

**SCOPE CAVEAT — READ FIRST.** This experiment holds the concurrency N fixed at
4 (the workload size). It therefore tests only ONE of the two links in the
causal chain we care about:

  - TESTED: "compute queue deepens → per-layer compute window c stretches."
  - NOT TESTED: "storage degrades → transfers stretch → more requests in flight
    at once." Here N does not respond to bandwidth; it is an input, not an
    output.

Every result below is a fixed-N result. The concurrency-feedback channel — where
a degradation episode inflates N and N feeds back into c — is exactly where
compounding could still appear, and it is deferred to a follow-on discrete-event
simulation. Do not read the sub-additivity finding as "the two effects don't
compound." Read it as "through the compute-window channel alone, at fixed N,
they don't compound; the feedback channel is untested."

**Setup.** We break ObjectCache's assumption (arXiv:2605.22850) that the
per-layer compute window c_i is a fixed property of request i. On our hardware
prefill throughput saturates at concurrency 1, so N concurrent requests serialize
with measured queueing Q(N) ≈ 0.85·(N−1)·service (job 155248; N=4 wall-time
ratio 3.45x/3.35x). Two compute models, using OUR measured vLLM prefill
coefficients (a=3.5158e-5 s/tok, b=5.8979e-10 s/tok², SM 960–1035 MHz sustained
under a 400W cap, FlashAttn v3 — NOT the paper's A100 Table A8 numbers):

  - c_iso(req)       = T_prefill(uncached)/L,  uncached = ctx·(1−hit)
  - c_queued(N,req)  = c_iso · (1 + 0.85·(N−1)),  = 3.55·c_iso at N=4

The scheduler stays naive: every policy allocates bandwidth from r*_iso = s/c_iso
in every cell (ObjectCache never sees queueing). Only the physics (the stall
model) uses c_actual. Added TTFT is measured against the fixed isolated
infinite-BW baseline L·c_iso, so both the queueing wait and the bandwidth stall
surface, and the wait cancels cleanly in the interaction term.
Script: `analysis/queue_compute.py`. Data:
`data/raw/queue_compute_20260917T035604Z/`.

**Four-cell aggregate added TTFT (ms):**

| Policy         | (1) iso+const | (2) iso+bimod | (3) queued+const | (4) queued+bimod |
|----------------|--------------:|--------------:|-----------------:|-----------------:|
| **Workload A (cap 80 Gbps)** |        |               |                  |                  |
| Equal          |        3534.4 |        4772.4 |           8855.7 |           9505.6 |
| Stall-opt      |        2839.4 |        3875.3 |           7896.2 |           8491.7 |
| Cal. Stall-opt |        3057.7 |        3936.0 |           8070.2 |           8542.1 |
| KV-prop        |        3768.5 |        4871.2 |           8379.2 |           9272.7 |
| BW-prop        |       11108.6 |       12950.4 |          11309.0 |          13150.9 |
| **Workload B (cap 50 Gbps)** |        |               |                  |                  |
| Equal          |        7012.5 |        8557.1 |          11150.8 |          12127.0 |
| Stall-opt      |        5906.4 |        7336.4 |           9927.2 |          11081.2 |
| Cal. Stall-opt |        5906.4 |        7369.1 |           9927.2 |          11108.4 |
| KV-prop        |        7012.5 |        8680.9 |          11064.4 |          12294.0 |
| BW-prop        |       19235.6 |       21501.1 |          19436.0 |          21701.5 |

**Superadditivity hypothesis: TESTED AND NOT SUPPORTED (at fixed N).** The
independent-effects prediction of cell (4) is (2)+(3)−(1); interaction = (4) −
that. On Workload A: Equal −588, Stall-opt −441, Cal. Stall-opt −406 ms — all
sub-additive (the wait cancels, so this is a pure stall interaction). Workload B
is the same sign (Equal −568, Stall-opt −276, Cal. Stall-opt −282). The effects
partially OFFSET, they do not compound.

**Mechanism (the explanation, not just the number).** Queueing stretches the
per-layer compute window 3.55x at N=4. A wider compute window gives the layer
pipeline more shadow to hide KV transfers under, so the *incremental* damage from
a bimodal bandwidth drop is SMALLER when compute is queued. Concretely, for
Stall-opt on Workload A, bimodal variance adds 1036 ms on top of isolated-c
(3875 − 2839) but only 595 ms on top of queued-c (8492 − 7896). That gap is the
offset.

**BW-prop is the degenerate control that makes the mechanism credible.** BW-prop
is so transfer-bound that max(X,c)=X always — the compute window never binds — so
its interaction is exactly 0.0 in both workloads (11309+12950−11109 = 13151 =
cell 4 to the decimal). If the sub-additivity were an artifact it would not
vanish precisely where the compute window stops mattering. It does. That is why
the mechanism is the explanation and not an assertion.

**How wrong r*_i gets under queueing.** r*_iso / r*_queued = 1 + 0.85·(N−1):
1.85 / 3.55 / 6.95 / 13.75 at N = 2 / 4 / 8 / 16. ObjectCache allocates r*_iso,
so it over-provisions bandwidth to a queued request by up to 13.75x at N=16 —
bandwidth that request cannot usefully absorb (its compute has not started) and
that could serve others. This is a real misallocation finding independent of the
TTFT interaction; under batching all N share the GPU equally, so position in the
queue is degenerate and every request in the wave sees the same factor.

**Inversion status (Stall-opt degrades more than Equal under bimodal).** Holds in
both bimodal cells, but the effect of queueing on it is directionally MIXED and
small: Workload A weaker (SO−EQ degradation gap +0.015 → +0.002), Workload B
slightly stronger (+0.022 → +0.029). We do NOT claim queueing amplifies the
inversion.

## 2026-09-17 — Two corrections to the time-varying bandwidth findings

Both use the paper's Table A8 compute window (t_total/L), unchanged — they
re-report the existing Step 2a findings more honestly. Script:
`analysis/corrections.py`. Data: `data/raw/corrections_20260917T035721Z/`.

**Correction 1 — the 14.5x uniform result is bounded and mostly tail.** The
unfloored uniform[0, 2·mean] is the literature's assumed model (Cake §5.6 samples
bandwidth ~ Uniform(0, 25 Gbps)), NOT a distribution any storage system produces;
it is kept as reference only. Flooring the distribution at 10/25/50% of mean
(mean preserved) drops Stall-opt's degradation from 14.5x → 11.3x → 8.5x → 5.4x
(Workload A). Attribution on identical draws: ~23% of the effect comes from
samples below 25% of mean, ~47% from below 50%. The 14.5x is real Jensen's-
inequality behavior driven by the near-zero tail, and about a third of it lives
in the sub-25%-of-mean region that real storage never visits.

**Correction 2 — bimodal is a (duration, gap) surface; depth is the only
measured axis.** Depth R=2.3 (B_high/B_low) is MEASURED — we observed exactly one
degradation episode. Low-episode DURATION and inter-episode GAP are SWEPT
assumptions (we saw one episode, so we cannot claim their values). Holding depth
and mean fixed, the Stall-opt/Equal inversion holds in 39/42 swept cells
(Workload A) and 42/42 (Workload B). It is strongest for long low-episodes with
short gaps (sustained degradation) and only disappears in 3 Workload-A cells with
long duration + short gap (e.g. 120s/30s → ratio 0.94), where the trace is
mostly-low and Equal's incidental headroom erodes too.

**Follow-on (planned).** Discrete-event simulation where N is an OUTPUT: Poisson
open-loop arrivals, slot occupancy that lengthens as received bandwidth drops, c
recomputed from instantaneous N. That tests the concurrency-feedback channel this
fixed-N work does not, and looks for metastability (a degradation episode whose
queue outlives it).
