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

## 2026-09-18 — Crossover audit: 27x claim retracted, formula verified, process fix

An external review checked our crossover table against our stated constants and
could not reproduce rows 1 and 3 of 4. The investigation found no arithmetic
error and no unit error. The formula is correct and defensible. Two issues:

### Issue 1 — Stale-coefficient contamination (the 27x claim)

The "27x boundary movement" figure was L\*(1400)/L\*(3806) = 81,941/3,031 =
27.0x, computed from the **superseded HF** prefill coefficients (a=3.4551e-5,
b=7.5003e-10). These coefficients remained in YAML notes after the vLLM fit
(a=3.5158e-5, b=5.8979e-10) became authoritative. With the correct vLLM
coefficients, the ratio would be 103,174/2,825 = 36.5x, but the denominator is
fragile (see below), so the ratio itself is not a defensible claim. The 27x
figure is **RETRACTED**.

The stale "1.37x" ratio in the `v2_full_pipeline` note (47.3/34.55, referencing
the HF coefficient) is corrected to 1.35x (47.3/35.16, vLLM coefficient).

**Root cause:** `scripts/prefill_loaded_sweep_v2.py` hardcoded S, H, and M
rather than reading `config/measured_constants.yaml`. Code and record were never
coupled, so the superseded coefficient survived into a headline claim. Fixed:
the script now reads all constants from the YAML and writes fitted coefficients
back with date and source provenance.

### Issue 2 — Row 3 is fragile (the reviewer's qualitative disagreement)

Our fetch cost formula:

    T_fetch_per_tok = S/BW + H + M

where all terms are MEASURED:
- S = 131,072 bytes/tok (`model.kv_bytes_per_token`)
- H = 2.38e-6 s/tok (`fetch_overhead.v2_h_to_gpu_us_per_token`) — PCIe H2D
- M = 6.1e-9 s/tok (`fetch_overhead.metadata_amortized_s_per_token`) — NFS metadata
- BW in bytes/s (`bandwidth.hammerspace_*`)

The reviewer used `T_fetch_per_tok = S/BW` (payload only, no H or M). This
agrees with our formula at low BW (where H is <3% of total) but disagrees at
3,806 MB/s: payload-only gives 34.44 us/tok < a=35.16 us/tok (no crossover),
while our formula gives 36.82 us/tok > a (crossover at 2,825 tokens). The
entire row 3 crossover exists because of the 2.38 us/tok H2D term.

**Internal consistency check.** At 2,918 MB/s (implied BW of the end-to-end
measurement), S/BW + H + M = 44.9 + 2.38 + 0.006 = 47.3 us/tok, matching the
measured `v2_full_pipeline_us_per_token` exactly. The decomposition is sound.

**H-sensitivity at 3,806 MB/s.** The crossover vanishes if H < 0.71 us/tok
(the critical threshold). Measured H across steady-state lengths (L≥32K,
15-rep medians): 2.365 and 2.393 us/tok (mean 2.379, stdev 0.020). The
measurement is 3.3x the critical threshold, so the crossover exists with high
confidence, but L\* is meaningful only to ±50 tokens from H variation:

| H (us/tok) | L\* (tokens) | Note |
|------------|-------------|------|
| 0.71       | 0           | Critical threshold — crossover vanishes |
| 2.00       | 2,181       | |
| 2.37       | 2,800       | Low end of measured range |
| 2.38       | 2,825       | Reported value |
| 2.39       | 2,847       | High end of measured range |
| 3.00       | 3,877       | |

### Corrected crossover table (N=1)

Formula: `L* = (S/BW + H + M − a) / b`. All inputs MEASURED.

| BW state | BW (MB/s) | T\_fetch (us/tok) | L\* (tokens) | Status |
|----------|-----------|-------------------|-------------|--------|
| Degraded | 1,400 | 96.01 | 103,174 | Crossover |
| Quiescent | 2,588 | 53.03 | 30,306 | Crossover |
| Quiet evening | 3,806 | 36.82 | 2,825 | **FRAGILE** — margin is 1.66 us |
| Peak 1-stream | 4,993 | 28.64 | — | Fetch always wins |

Payload-only comparison (reviewer formula: `L* = (S/BW − a) / b`):

| BW state | BW (MB/s) | T\_fetch (us/tok) | L\* (tokens) | Status |
|----------|-----------|-------------------|-------------|--------|
| Degraded | 1,400 | 93.62 | 99,128 | Crossover |
| Quiescent | 2,588 | 50.65 | 26,260 | Crossover |
| Quiet evening | 3,806 | 34.44 | — | **No crossover** |
| Peak 1-stream | 4,993 | 26.25 | — | Fetch always wins |

Rows 1, 2, and 4 agree qualitatively. Row 3 is the sole disagreement.

### The 34.6 vs 35.2 us/tok discrepancy

Not a discrepancy. Two different fits from two different engines:
- 34.6 us/tok = HF `coeff_a_hf_boosted` = 3.4551e-5 (2026-09-15, no FlashAttn)
- 35.2 us/tok = vLLM `coeff_a_vllm` = 3.5158e-5 (2026-09-16, FlashAttn v3, clock-logged)

The vLLM coefficient is authoritative. The 34.6 appeared in the
`v2_full_pipeline` note, which was written when HF was the only boosted
measurement and never updated. Corrected.

### Defensible replacement claim

**Retracted:** "27x boundary movement" (from superseded HF coefficients).

**Replacement (worked example):** A 30K-token context prefers recompute at
1,400 MB/s (T\_recompute = 1,585.6 ms vs T\_fetch = 2,880 ms) and prefers
fetch at 3,806 MB/s (T\_recompute = 1,585.6 ms vs T\_fetch = 1,105 ms). This
holds under both formula variants (payload-only: 2,809/1,033 ms). Stated as a
concrete worked example at a specific length, not as a ratio of crossover
points, because the high-BW crossover point is fragile.

### Process fix

The root cause was decoupling between code and record.
`scripts/prefill_loaded_sweep_v2.py` now reads S, H, M from
`config/measured_constants.yaml` and writes fitted a/b back after each run. M
(6.1e-9 s/tok, previously inline-only) is now a full YAML entry with
provenance. This ensures that when a coefficient is superseded, the crossover
computation automatically uses the updated value.

## 2026-09-19 — Q(N) slope corrected, metastability withdrawn, three single-observation findings retracted

### Q(N) slope: 0.85 → 1.0

The Q(N) = 0.85·(N−1)·service relationship used in Experiments B and C was a
**cold-start measurement artifact**. Job 155248 (`diagnose_batching.py`) made a
single unreplicated N=4 measurement at L=4096 with no warmup on rpg-93-5. The
N=1 baseline (0.1546s) included one-time costs (JIT, memory allocation) that
inflated the denominator, producing ratio=3.45x (slope 0.82). The loaded sweep
(`prefill_loaded_sweep_v2.py`) measured the same configuration with warmup and
3 reps and found ratio=3.978 (slope 0.993) — but the diagnostic result was
adopted because it seemed more targeted and its result was more dramatic. The
correcting data was available at the time.

**Corrected value: slope ≈ 1.0** (pure serialization, no batching benefit at
production context lengths). Evidence:
- Extended sweep (`qn_extended_sweep.py`): 3 context lengths × 9 N-values to
  N=64, linear with beta=0.998, no curvature
- Batch-budget confound (`qn_batch_confound.py`): 5 `max_num_batched_tokens`
  values (8K–131K) × 2 lengths × 7 N-values on rpg-93-3. Slope does not depend
  on batch budget (variation ~0.02, within noise)
- Cross-node: rpg-93-1 (825–885 MHz) and rpg-93-3 (945–1005 MHz) agree within
  0.7%. Clock-invariant as expected for a dimensionless ratio
- Per-length: 1.107 at L=4K (short-context per-step overhead), 0.994 at L=16K,
  0.982 at L=32K

Simulation code updated: `Q_SLOPE = 1.0` in `analysis/queue_compute.py`.

### Experiments B and C re-run at slope=1.0

**Experiment B (fixed N=4):** c\_wall goes from 3.55x to 4.0x c\_iso.
Sub-additivity preserved for all policies of interest: Stall-opt −441 → −431
(WL-A), −276 → −313 (WL-B). BW-prop WL-A flips from exactly 0.0 to weakly
super-additive (+199 ms) because at a 4.0x window it is no longer fully
transfer-bound — the degenerate control becomes non-degenerate. Does not affect
any claim about Stall-opt or Equal.

**Experiment C (feedback):** Superadditivity preserved and strengthened at every
load above 30%: 70% +601 → +742 ms, 85% +1141 → +4614 ms.

**The contingent-sign result is robust.** Fixed-N sub-additive and feedback
super-additive at every load, under both slope 0.85 and 1.0. The sign survived
an 18% change in a load-bearing constant. If it had flipped, the finding was an
artifact of the constant.

### Metastability claim WITHDRAWN

**The 130s/2.16x drain-time headline is retracted.** 30-seed replication at 70%
load, slope=1.0, against the reviewer's fluid-queue baseline
t\_drain = (n\_end − preN) / (μ − λ), μ=1.59, λ=1.11:

- Simulated/fluid ratio: mean 1.53, **median 1.09**, std 1.58, range [0.00, 7.24]
- 15/28 seeds above 1.0, 13/28 below
- Drain persistence: mean 1.09x, **median 0.75x**, std 1.26x
- Only 9/27 draining seeds exceed 1.0x persistence
- 3 seeds had the queue shrink during the episode (Poisson thinning)

The distribution is centered on the fluid prediction. We cannot distinguish
simulated drain from ordinary backlog recovery. The mean (1.53) is pulled by two
outliers (7.24, 5.02); the median (1.09) is the better central tendency.

The preN→drain mechanism does not exist: r = −0.031 at n=30. The apparent
pattern in the 5-seed sample was Poisson noise. What does predict drain time is
delta\_N (r = 0.536) — how much backlog accumulated during the episode — which
is ordinary queue behavior, not a metastability signature.

**85% load excluded as intrinsic instability.** No-episode control at slope=1.0
trends upward (window means 19.7 → 35.6 with no degradation injected). Same
exclusion logic previously applied to 95% at slope=0.85.

**70% no-episode control is stationary.** 3/5 seeds stationary, 2/5 marginal,
0/5 trending. This is a positive control added in response to external review.

### Saturation is a zone, not a point

5-seed arrival-rate sweep at slope=1.0 with no episode: 70% is the only
majority-stable load (3/5 stable, 0/5 unstable). 71–83% is transitional (mostly
marginal seeds). 84–85% has 2/5 unstable seeds. There is no sharp boundary.

Drop the "~76%" figure — it was a single-run artifact. Effective capacity is
somewhere in the 70–76% range (1.11–1.21 req/s) depending on how "stable" is
defined, substantially below the 1.59 req/s compute-bound estimate.

**Operational implication:** the slope correction alone moved the stability
boundary from ~85% to ~70–76%. An assumed 15% prefill batching benefit that
does not exist inflates apparent capacity by roughly 12–24%. A deployment sized
on that assumption over-provisions its headroom, and discovers the error during
a degradation episode — exactly when the margin matters. This finding is
independent of the simulation and stands on the Q(N) measurement alone.

### Extrapolation caveat: RESOLVED

Q(N) is now measured linear to N=64 with beta=0.998 and no curvature. This
covers the peak N=48 reached in the 70% metastability runs (within range) and
would have covered the peak N=68 at 85% had that load not been excluded. Drain
times are no longer extrapolated. The concern that motivated Priority 1 of the
Q(N) extension — sub-linear scaling at high N shortening drain times — was
resolved: the scaling is linear and if anything super-linear at short contexts.

### Process note: three single-observation findings retracted

1. **0.85 Q(N) slope** — cold-start artifact from a single unreplicated
   diagnostic measurement. Correcting data (loaded sweep, slope 0.993) was
   available at the time.
2. **27x crossover boundary ratio** — computed from superseded HF coefficients
   that remained in a note after vLLM became authoritative.
3. **2.16x metastability persistence** — seed 7 of 5, the second-highest draw
   from a distribution with median 0.75x.

All three were caught by external review, not internal process. In each case the
finding was adopted because it was more dramatic than the alternative, and in
each case the correcting data or methodology was available at the time.

### What survives (as of 2026-09-19)

- **Q(N) = N × T\_service(1):** Measured 3 nodes, 5 batch budgets, 3 context
  lengths, N to 64, 2 clock states. Prefill serializes; no measurable batching
  benefit at production context lengths.
- **Storage bandwidth:** 1.4–6.0 GB/s on Hammerspace NFS, bimodal and episodic,
  with a documented spontaneous 2.3x degradation event.
- **Fetch mechanism progression:** 135.3 → 67.7 → 47.3 us/tok. The loading
  path matters as much as storage bandwidth.
- **Crossover formula:** T\_fetch = S/BW + H + M. H-sensitivity: H\_critical =
  0.71 us/tok, measured H = 2.38, 3.3x threshold.
- **Saturation zone:** 70–85% transitional. Effective capacity materially below
  the compute-bound estimate due to serialization overhead.

## 2026-09-20 — Contingent-sign result withdrawn; simulation findings reduced to measurements + textbook

### Bug: overlap-sweep condition was inverted

The conditional-overlap sweep in `scripts/contingent_sign_test.py` used
`layer < prefetch_layers`, which meant P=0 gave full overlap (the condition
was never true, so the `else` branch always ran) and P=32 gave no overlap
(always true, so sequential execution). The numbers in the first run were
correct; the labels were backwards. Caught before recording. Fixed to use
`overlap_layers` directly in `scripts/overlap_sweep.py`.

### Super-additivity (feedback regime) = queueing convexity

Seeded, 30 seeds, bootstrap 95% CIs exclude zero at all loads: 30% \[+25, +48\],
50% \[+127, +209\], 70% \[+710, +1054\] ms. The interaction is real and
reproducible.

A convexity baseline — fair-share queue with no layer pipeline, no overlap, no
bandwidth allocation policy, same compute model — reproduces it and **exceeds**
it at 50% and 70%. Sim − baseline: −22 ms \[−34, −11\] at 50%, −147 ms \[−246,
−50\] at 70%. CIs exclude zero on the negative side. Our simulator's layer
pipeline actually *dampens* the convexity-driven interaction.

**The super-additive feedback interaction is standard queueing theory**, not a
storage×compute coupling specific to KV cache serving. Near saturation, queueing
delay is convex in utilization. Any two independent capacity reductions (BW
variance + compute contention) compound along that convex curve. The steep
growth with load (+36/+167/+874 ms) is exactly the shape convexity produces.

### Sub-additivity (fixed-N regime) is produced entirely by the overlap model

Corrected overlap sweep (0/31 to 31/31 layers with pipeline overlap):

| Overlap layers | Stall-opt WL-A interaction | Stall-opt WL-B |
|---------------|---------------------------|----------------|
| 0/31 (none) | +3.5 ms | +3.1 ms |
| 1/31 | −10.2 ms | −5.7 ms |
| 4/31 | −51.7 ms | −32.9 ms |
| 8/31 | −105.5 ms | −70.7 ms |
| 31/31 (full, original) | −430.8 ms | −313.0 ms |

At overlap=0, the interaction is effectively zero for all policies. Equal at
0/31: +2.1 ms (WL-A), −0.4 ms (WL-B) — confirming no other channel. The sub-
additivity scales linearly with overlap coverage at \~14 ms per pipelined layer.

Note: 31/31 with one-layer lookahead IS ObjectCache's Equation 3 pipeline model,
so it is the realistic model for that system, not an extreme assumption. The
operative question is not how many layers pipeline — it is whether KV transfer
may overlap time spent **queued behind other requests** (c\_wall) rather than
only the request's own compute time (c\_useful).

### vLLM 0.29.0 KV connector scheduling: reads start AFTER admission, not while queued

Source-code audit of vLLM 0.29.0 (`v1/core/sched/scheduler.py`,
`v1/worker/gpu/kv_connector.py`, `v1/worker/kv_connector_model_runner_mixin.py`,
`distributed/kv_transfer/kv_connector/v1/base.py`,
`distributed/kv_transfer/kv_connector/v1/lmcache_connector.py`):

- The scheduler queries `get_num_new_matched_tokens()` during scheduling to
  check if external KV tokens are available, but does not initiate reads.
- **Sync KV loads:** started in `pre_forward()` after the scheduler admits the
  request to a batch. The request is in the running set; reads precede the
  forward pass but NOT the scheduling decision.
- **Async KV loads:** the request enters `WAITING_FOR_REMOTE_KVS` state and
  stays out of the running batch. Loads start via `build_connector_meta()` at
  the end of `schedule()`, then `post_forward()` in subsequent steps.

In neither path are KV reads issued while a request waits in the prefill queue
for GPU admission. **Queueing time does not provide transfer-compute overlap in
the real system.** The c\_wall overlap model — where queueing delay stretches
the window for hiding transfers — does not match vLLM's actual scheduling.
Transfers can only overlap with the request's own forward-pass compute
(c\_useful), not with time spent waiting behind other requests.

This means the realistic overlap regime is closer to overlap=0/31 (interaction
\~0) than overlap=31/31 (interaction −431 ms). The sub-additive finding is an
artifact of the overlap model, not a property of the real system.

### Contingent-sign result WITHDRAWN

The contingent sign (sub-additive at fixed N, super-additive under feedback) is
withdrawn as a finding about KV cache serving:
- The super-additive half is queueing convexity, reproduced by a textbook
  baseline.
- The sub-additive half depends on an overlap assumption that does not match the
  real scheduler's behavior.

### Process note: fourth simulation-derived finding withdrawn after controls

The pattern is now consistent: all four simulation-derived candidate findings
(metastability, super-additivity-as-coupling, sub-additivity-as-mechanism, and
the contingent sign) reduced to textbook queueing or modeling assumptions once
a baseline or control was applied. No simulation finding from Experiments B or C
survived as a KV-serving-specific result.

### What survives (updated 2026-09-20)

**Hardware measurements (not dependent on the simulation):**
- **Q(N) = N × T\_service(1):** Prefill serializes completely. Measured 3 nodes,
  5 batch budgets, 3 lengths, N to 64. The prior 0.85 slope was a cold-start
  artifact.
- **Storage bandwidth:** 1.4–6.0 GB/s, bimodal/episodic, documented 2.3x
  degradation. Not contested.
- **Fetch mechanism progression:** 135.3 → 67.7 → 47.3 us/tok. The loading
  path matters as much as storage bandwidth. Not contested.
- **Crossover formula:** T\_fetch = S/BW + H + M. Audited, internally
  consistent. Not contested.
- **Saturation zone:** 70–85% transitional at slope=1.0. Effective capacity
  materially below the 1.59 req/s compute-bound estimate. Operational
  implication: a non-existent 15% batching benefit inflates apparent capacity.

**Simulation results that survived controls:**
- None. Every interaction-magnitude finding reduced to a baseline or modeling
  assumption. The simulations were useful for identifying questions (do effects
  compound? does the queue outlive the episode?) but did not produce answers
  that survived controlled testing.

## 2026-09-24 — Loaded crossover withdrawn: single-GPU L* is invariant in N

### The error

The loaded crossover computation (`scripts/prefill_loaded_sweep_v2.py`, line
383) modeled recompute with queueing — T\_wall(N) = N × service — but gave each
concurrent fetch the full node storage bandwidth independently:

    T_fetch_per_tok = S/BW + H + M          (same for all N)
    T_recompute_per_tok(N, L) = N*(a + b*L)  (scales by N)

This produced "N=2 crossover drops to 21K" and "N>=4 fetch always wins at any
BW." Both are artifacts of the unshared-bandwidth model.

The contradiction was available in the project's own data: single-node NFS read
bandwidth is flat at 4.4–4.8 GB/s across 1–16 parallel fio streams (a per-node
ceiling, measured 2026-09-12). N concurrent fetches on one node share this
ceiling. Similarly, N concurrent requests on one GPU share one PCIe link for
H2D transfers.

### The correction

With all shared resources correctly divided:

    T_fetch_per_tok(N) = N*(S/BW + H) + M
    T_recompute_per_tok(N, L) = N*(a + b*L)

Both sides scale by N. Setting equal:

    L* = (S/BW + H - a)/b + M/(N*b)

The base term (S/BW + H - a)/b is identical to the N=1 crossover. The only
N-dependent piece is M/(N×b) ≈ 10/N tokens — negligible. **The single-GPU
crossover is invariant in N.** The loaded crossover table reduces to the N=1
table.

### What is withdrawn

- **"N=2, degraded BW: crossover drops to ~21K"** — artifact.
- **"N>=4, any BW: fetch always wins"** — artifact.
- **"Under concurrent fallback load (N>=2), fetch dominates; GPU serialization
  is the decisive factor"** — the asymmetry does not exist on a single GPU.
  Storage, PCIe, and compute are all shared among concurrent requests.

### What replaces it

**On a single GPU, concurrency does not move the crossover.** The crossover
depends on bandwidth and the prefill coefficients, not on the number of
concurrent requests. The N=1 table stands as the complete single-GPU answer.

**DERIVED (not measured):** Across a node's GPUs, each GPU has its own compute
pipeline and its own PCIe link, while the node's storage ceiling is shared by
all GPUs. With more GPUs fetching simultaneously, per-GPU storage bandwidth
shrinks while per-GPU compute is unaffected. So with more GPUs busy, recompute
gains relative to fetch. This follows from two measured inputs (flat storage BW
across streams; independent GPU compute) but has not been measured directly.

### Process note: fifth finding withdrawn after resource-sharing check

The external reviewer raised the concern ("both storage and GPU compute appear
capped") and it was not propagated at the time. The pattern is the same as
prior withdrawals: the loaded crossover was adopted because its conclusion was
more dramatic (fetch dominates under concurrency), and the contradicting
evidence (flat bandwidth across streams) was already in the project's own
measurements. This is the fifth finding caught by external review rather than
internal process.

## 2026-09-24 — Multi-GPU prefill scaling measured; DERIVED claim promoted to MEASURED

### What was measured

Independent vLLM 0.29.0 processes (CUDA\_VISIBLE\_DEVICES pinned, no tensor
parallelism) on 1-4 GPUs of the same node simultaneously. L = 16384 and 32768,
enforce\_eager, FlashAttention v3, max\_model\_len=33000, 5 reps after warmup,
per-GPU SM clock and power logged at 100ms.

3 independent runs across 2 nodes (rpg-93-6 ×2, rpg-93-3 ×1): jobs 160492,
160494, 160495.

### Results

Scaling efficiency vs G × single-GPU throughput (range across 3 runs):

| G | L=16K | L=32K |
|---|-------|-------|
| 1 | 100% | 100% |
| 2 | 98.4-101.7% | 98.6-101.4% |
| 3 | 97.4-99.9% | 97.8-100.1% |
| 4 | 98.3-98.6% | 98.3-98.9% |

Per-GPU SM clocks: no degradation at G=4 vs G=1. Mean power per active GPU
82-132W, well below 400W TDP. No thermal or power throttling observed.

Host RSS: ~1.5 GiB per vLLM worker process (model weights are GPU-resident;
host cost is Python interpreter + metadata). The initial G=4 OOM (job 160364)
was a Slurm cgroup limit: without `--mem`, Slurm allocated 8 GiB
(1 CPU × DefMemPerCPU=8054 MB). The node has 503 GiB physical RAM. Fixed
with `--mem=500G`.

### Node-level fetch-vs-recompute crossover (per context length)

Fetch model: T\_fetch\_per\_tok(G) = G × S/BW\_node + H + M. Storage read and
H2D copy are treated as serial per token, consistent with the v2 fetch
measurement (47.3 us/tok). In a pipelined implementation, H2D would overlap
with the next chunk's storage read (each GPU has its own PCIe link; storage
is 19x slower: 44.9 vs 2.38 us/tok), giving ~5% higher throughput. All values
use the conservative serial model.

MARGINAL = recompute/fetch ratio within 0.9-1.1.

L = 16,384: recompute wins at all G and BW states except single-GPU at quiet
evening (0.87x) and peak (0.68x). No MARGINAL cells.

L = 32,768: two MARGINAL cells:
- **Quiescent, G=1**: ratio 1.01 (recompute barely wins; this is close to the
  single-GPU crossover at ~30K tokens).
- **Peak, G=2**: ratio 1.06 (recompute barely wins; two GPUs just overtake the
  storage ceiling at the fastest measured BW).

### Capacity-share framing

Storage share = node fetch tok/s / (node fetch + aggregate recompute tok/s).
This is the fraction of total KV-restoration capacity provided by the shared
storage link when a system can fetch and recompute simultaneously.

| BW state | G=1 | G=2 | G=3 | G=4 |
|----------|-----|-----|-----|-----|
| Degraded | 31-35% | 18-21% | 13-16% | 10-12% |
| Quiescent | 44-50% | 29-33% | 22-25% | 17-21% |
| Quiet evening | 53-59% | 37-42% | 29-33% | 23-27% |
| Peak | 60-65% | 43-49% | 34-39% | 29-33% |

Ranges span L=16K and L=32K. At degraded BW with all 4 GPUs active, storage
contributes only 10-12% of total restoration capacity.

### Process note

The vLLM environment required CUDA\_HOME pointing to the system's Spack-installed
CUDA 12.9.1 (the same path used in all prior working vLLM runs), not the conda
env's CUDA 13. The mismatch was diagnosed by diffing environment variables from
the job 155219 output header against the failing run. FlashInfer's JIT sampling
kernel cache was pre-warmed with a single-GPU run before launching parallel
workers, preventing cache-race conditions from simultaneous ninja builds.
