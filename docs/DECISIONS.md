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
