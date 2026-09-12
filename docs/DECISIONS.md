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
