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

## 2026-09-17 — The sign of storage×compute coupling is contingent on whether concurrency can respond to bandwidth

**HEADLINE.** Break ObjectCache's assumption (arXiv:2605.22850) that the
per-layer compute window c_i is a fixed property of request i, independent of
concurrent load — on our hardware prefill throughput saturates at concurrency 1,
so N requests serialize with measured queueing Q(N) ≈ 0.85·(N−1)·service (job
155248; N=4 wall-time ratio 3.45x/3.35x). The question was whether storage
degradation (bandwidth variance) and compute contention (queueing) *compound*.
The answer is: **the SIGN of the coupling is determined by whether concurrency
can respond to bandwidth.**

  - **Fixed N (control): SUB-additive.** With N pinned at the workload size,
    interaction = −441 ms for Stall-opt (Workload A). Queueing stretches the
    compute window, which *shields* against bandwidth stalls.
  - **Feedback (N is an output): SUPER-additive.** When N responds to bandwidth,
    interaction = +600.9 ms for Stall-opt at 70% load. Concurrency growth
    *compounds* the two penalties.

These two are a PAIR, not a hypothesis and its refutation. Both ObjectCache and
Cake evaluate with fixed request sets, so the fixed-N result is the CONTROL that
shows what their modeling assumption produces — sub-additivity — and the feedback
result shows what actually happens once offered load is open-loop. Neither system
models either half. The mechanism is symmetric: fixed N → queueing stretches the
compute window 3.55x (N=4), giving the layer pipeline more shadow to hide
transfers under; feedback → bandwidth drops stretch transfers, occupancy climbs
(peak N=41 vs mean 12.6; mean N 13.3 in low-BW samples vs 11.8 in high-BW), the
per-request compute window compresses in useful terms, and r*_i estimates degrade
further.

Both halves are pure simulation driven by our measured vLLM prefill coefficients
(a=3.5158e-5 s/tok, b=5.8979e-10 s/tok²; SM 960–1035 MHz sustained under a 400W
cap, FlashAttn v3 — NOT the paper's A100 Table A8 numbers), measured
Q(N)=0.85·(N−1)·service, and measured storage traces. Feedback half uses
open-loop Poisson arrivals. Not validated against a live serving system.

---

### Part 1 — Fixed N (the control)

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

**Stall-opt vs Equal — both metrics (fixed N).** Two different comparisons that
need not agree, reported together (see Correction 2 below): by ABSOLUTE added
TTFT, Stall-opt is lower (better) than Equal in every cell (WL-A cell 4: 8491.7
vs 9505.6). By degradation RATIO (bimodal/constant), Stall-opt is HIGHER (more
variance-sensitive) — the "inversion." So Stall-opt is simultaneously better on
average and more sensitive to variance. The effect of queueing on the ratio
inversion is directionally MIXED and small (WL-A SO−EQ gap +0.015 → +0.002; WL-B
+0.022 → +0.029); we do NOT claim queueing amplifies it.

### Part 2 — Feedback (N is an output)

**Setup.** Discrete-event, time-stepped (`analysis/queue_feedback.py`, data
`data/raw/queue_feedback_20260917T042116Z/`). Open-loop Poisson arrivals — a
closed-loop client would throttle offered load when latency rises and hide the
queue growth we are hunting. Each request holds a concurrency slot until BOTH its
KV fetch (rate = policy allocation from r*_iso, the same naive scheduler) and its
prefill compute (rate = 1/(1+0.85·(N−1)) of real time) finish; slower transfers
lengthen occupancy, raising N, which feeds back into the compute window. N(t) is
logged. Same four cells (iso vs queued × constant vs bimodal BW), same
interaction form. Metric is mean added TTFT over completed requests. Arrival rate
swept as a fraction of the compute-bound capacity estimate (~1.59 req/s).

**N responds to bandwidth — the loop closes.** Representative run (85% load,
bimodal, Stall-opt, queued): N tracks bandwidth inversely, ramping through every
low-BW episode and draining on recovery — peak N=41, mean 12.6, mean N 13.3 in
low-BW samples vs 11.8 in high-BW. Plot: `N_vs_bw.png`.

**Superadditivity (interaction = (4) − [(2)+(3)−(1)], ms):**

| Load | Equal | Stall-opt | Cal.SO | KV-prop | BW-prop | censoring |
|------|------:|----------:|-------:|--------:|--------:|-----------|
| 30%  | +14.6 |    +15.9  |  −11.0 |  +42.1  |   −6.1  | 0.0% |
| 50%  | +93.5 |    +79.4  |  +57.5 | +191.8  |  −38.9  | 0.0% |
| 70%  | +545.8|   +600.9  | +578.3 | +731.2  |  −23.0  | 0.0% |
| 85%  | +944.3|  +1140.8  |+1110.8 |+2059.5  | +227.4  | 0.0% |
| ~~95%~~ | ~~+415~~ | ~~+5537~~ | ~~+5696~~ | ~~+4506~~ | ~~+584~~ | **CENSORED** |

The interaction is positive and grows steeply with load. It is already clean and
uncensored at 50–70% — a stable regime with 0.0% unfinished requests — so the
finding does not depend on the blow-up. All claims are based on the 50–85% range.

**Correction 1 — the 95% row is censored and excluded.** At 95% offered load,
requests that arrive in the eligible window but never finish by the horizon are
dropped from the mean, and the censoring is ASYMMETRIC across policies: Equal
4.4% (1155/1208 completed), KV-prop 2.8%, Stall-opt 1.7%, BW-prop 0%. Equal's
slow tail is exactly what gets cut, which deflates its cell-4 mean and produces
the misleading Equal +415 vs Stall-opt +5537 gap. The 95% row is unreliable and
excluded from every claim. Loads 30–85% have 0.0% censoring and are the basis for
all conclusions.

**BW-prop is the control in both regimes.** Transfer-bound, so max(X,c)=X and the
compute window never binds: interaction ≈ 0 (slightly negative) at every
non-saturated load, exactly as in the fixed-N case where it was 0.0 to the
decimal. It only turns positive once saturation drags everything up. The
mechanism is credible because it vanishes precisely where the compute window
stops mattering.

**METASTABILITY — the strongest single finding.** One injected 60s degradation
episode on otherwise-constant bandwidth, then full recovery at t=360s. The queue
built during the episode OUTLIVES its cause: at 70% load N drains back in **94s
(1.57× the 60s episode)**, at 85% in **155s (2.58×)**. The 95% case is excluded —
its pre-episode window is already at N≈28 and climbing (N 22→35), i.e. intrinsic
instability where offered load exceeds effective capacity, not episode-induced
persistence.

**Why metastability matters — it is CacheGen estimator blindness from the other
side.** CacheGen estimates path bandwidth from the previous chunk's throughput.
During the 94–155s drain window, bandwidth has fully recovered but the system has
not: the queue is still draining. A throughput-based estimator sees a healthy
path and resumes normal behavior while the system remains degraded for another
95–155s, for a reason it structurally cannot observe (occupancy, not bandwidth).
The estimator blindness we identified in CacheGen's text-fallback mode and this
metastability are the same problem viewed from two directions: throughput is not
a sufficient statistic for the state of a queueing system.

**Stall-opt vs Equal — both metrics (feedback).** Same two comparisons as fixed
N, non-censored loads: by ABSOLUTE added TTFT Stall-opt is lower (SO/EQ 0.93 →
0.81 from 30% → 85%); by degradation ratio (cell4/cell3) Stall-opt is HIGHER
(1.19–1.31 vs Equal 1.12–1.19), so the ratio inversion HOLDS and does not reverse
under feedback. The two metrics disagree because they measure different things —
average performance vs variance sensitivity — and the pattern is identical in
both regimes: Stall-opt is better on average and more variance-sensitive.

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

**Follow-on (DONE).** The concurrency-feedback simulation is implemented
(`analysis/queue_feedback.py`) and is written up in the contingent-sign entry
above — it found super-additivity and the metastability result (queue outlives a
60s episode by 1.5–2.6×).
