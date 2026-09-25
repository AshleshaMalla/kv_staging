#!/usr/bin/env python3
"""Break the compute-window assumption: queue-aware c_i.

ObjectCache's zero-stall rate is r*_i = s_i / c_i, where c_i is the per-layer
compute window, treated (their footnote 1 + derivation) as a fixed property of
request i, independent of concurrent load.  We measured that failing on our
hardware: prefill throughput saturates at concurrency 1, so N concurrent
requests serialize with queueing delay Q(N) ~= 0.85 * (N-1) * service
(measured, job 155248; wall-time ratio 3.45x/3.35x at N=4).

THIS SCRIPT IS ABOUT OUR HARDWARE, NOT A RE-DERIVATION OF THEIRS.
Compute windows use OUR measured vLLM prefill coefficients
(a=3.5158e-5 s/tok, b=5.8979e-10 s/tok^2; SM 960-1035 MHz sustained under a
400W cap, FlashAttn v3) — NOT the paper's A100 Table A8 numbers.

Two compute models:
  c_iso(req)        = T_prefill(uncached_tokens) / L      (paper's assumption)
  c_queued(N, req)  = c_iso * (1 + 0.85*(N-1))            (queue-aware)
where uncached_tokens = ctx*(1-hit): cached KV is LOADED, not recomputed, so
the compute window is the prefill of the uncached suffix (higher hit ratio ->
smaller compute window, matching the paper's Table A8 structure).

The SCHEDULER stays naive: every policy allocates bandwidth from r*_iso =
s/c_iso in every cell (ObjectCache never sees queueing).  Only the PHYSICS
(the stall model) uses c_actual.  Added TTFT is measured against the fixed
isolated infinite-BW baseline L*c_iso, so both the queueing wait and the
bandwidth stall surface, and the (policy/BW-independent) wait cancels cleanly
in the superadditivity interaction term.

Four cells:
  (1) c_iso    + constant BW   <- paper's assumptions, reference
  (2) c_iso    + bimodal  BW   <- the variance finding we already have
  (3) c_queued + constant BW   <- the compute assumption alone
  (4) c_queued + bimodal  BW   <- both together

Pure simulation.  Deterministic.  No cluster, no GPU.
"""

import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stallopt import (
    L, BYTES_PER_TOKEN_PER_LAYER, POLICIES, POLICY_ORDER,
    Bps_to_gbps, gbps_to_Bps, _transfer_time,
    make_constant_trace, make_bimodal_trace,
)

# ── OUR measured vLLM prefill coefficients (config/measured_constants.yaml) ──
A_VLLM = 3.5158e-05   # s/token       (rounds to 3.52e-5)
B_VLLM = 5.8979e-10   # s/token^2      (rounds to 5.90e-10)

# ── Measured queueing model ──
# SUPERSEDED: Q_SLOPE = 0.85 (cold-start artifact from job 155248's
# diagnose_batching.py — single unreplicated measurement, no warmup.
# Contradicted by prefill_loaded_sweep_v2.py's own data at the same config.)
# Corrected 2026-09-19: slope = 1.0 across 3 nodes, 5 batch budgets,
# 3 context lengths, N up to 64, linear with beta=0.998.
Q_SLOPE = 1.0         # Q(N) = 1.0 * (N-1) * service ; T_wall(N) ≈ N * T_service(1)

# Workload: the 4 concurrent requests of paper Workloads A/B.
WORKLOAD_SPEC = [
    (16384, 0.500),
    (16384, 0.875),
    (65536, 0.500),
    (65536, 0.875),
]
N_CONCURRENT = len(WORKLOAD_SPEC)   # N = 4 requests share the GPU

WORKLOADS = {"A": 80, "B": 50}
EPOCH = 1.0
N_STARTS = 200
BIMODAL_DEPTH = 2.3
BIMODAL_EPISODE = 30.0   # representative; inside the inversion region (Corr. 2)


def t_prefill(tokens):
    """Our vLLM prefill compute time (s) for a forward pass over `tokens`."""
    return A_VLLM * tokens + B_VLLM * tokens * tokens


def queue_factor(n, slope=Q_SLOPE):
    """c_queued / c_iso = r*_iso / r*_queued = 1 + slope*(n-1)."""
    return 1.0 + slope * (n - 1)


@dataclass
class QReq:
    """Request with an OVERRIDABLE per-layer compute window (unlike stallopt's
    Request, whose c is locked to Table A8).  Exposes the attributes the
    allocate_* policies need: .s, .cached_tokens, .r_star."""
    context_tokens: int
    hit_ratio: float
    c_iso: float          # isolated per-layer compute window (s)

    @property
    def cached_tokens(self):
        return round(self.context_tokens * self.hit_ratio)

    @property
    def uncached_tokens(self):
        return self.context_tokens - self.cached_tokens

    @property
    def s(self):
        return self.cached_tokens * BYTES_PER_TOKEN_PER_LAYER

    @property
    def c(self):
        # Policies read r_star = s/c ; the naive scheduler always uses c_iso.
        return self.c_iso

    @property
    def r_star(self):
        return self.s / self.c_iso

    @property
    def label(self):
        return f"{self.context_tokens // 1024}K/{self.hit_ratio:.1%}"


def build_workload():
    reqs = []
    for ctx, hit in WORKLOAD_SPEC:
        uncached = ctx - round(ctx * hit)
        c_iso = t_prefill(uncached) / L
        reqs.append(QReq(ctx, hit, c_iso))
    return reqs


# ── Physics: pipeline TTFT with c_actual decoupled from the allocation c ──

def sim_one(alloc_reqs, idx, c_actual, policy_fn, trace, epoch, t0):
    """TTFT (s) for one request: layer pipeline with per-layer compute window
    c_actual, bandwidth allocated per-epoch by policy_fn from the naive
    (c_iso) requests."""
    s = alloc_reqs[idx].s
    t = t0
    t += _transfer_time(s, idx, alloc_reqs, policy_fn, trace, epoch, t0, t)
    for _ in range(1, L):
        dt = _transfer_time(s, idx, alloc_reqs, policy_fn, trace, epoch, t0, t)
        t += max(dt, c_actual)
    t += c_actual
    return t - t0


def run_cell(alloc_reqs, c_actual_list, policy_name, trace, epoch=EPOCH,
             n_starts=N_STARTS):
    """Aggregate added TTFT (ms) over the workload, averaged over start times.
    Added TTFT is measured vs the fixed isolated infinite-BW baseline L*c_iso
    (same reference in every cell)."""
    policy_fn = POLICIES[policy_name]
    margin = 60.0
    max_start = trace.duration - margin
    if max_start <= 0:
        starts = [trace.times[0]]
    else:
        starts = [trace.times[0] + max_start * i / (n_starts - 1)
                  for i in range(n_starts)]
    total = 0.0
    for t0 in starts:
        agg = 0.0
        for idx, req in enumerate(alloc_reqs):
            ttft_s = sim_one(alloc_reqs, idx, c_actual_list[idx],
                             policy_fn, trace, epoch, t0)
            agg += (ttft_s - L * req.c_iso) * 1000.0   # baseline: L*c_iso
        total += agg
    return total / len(starts)


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/raw") / f"queue_compute_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = []
    p = out.append

    p("=" * 80)
    p("QUEUE-AWARE COMPUTE WINDOW — four-cell experiment")
    p(f"Timestamp: {ts}")
    p("=" * 80)
    p("")
    p("OUR HARDWARE (H100 NVL, 400W, SM 960-1035 MHz, vLLM+FlashAttn v3).")
    p(f"vLLM prefill coeffs:  a = {A_VLLM:.4e} s/tok   b = {B_VLLM:.4e} s/tok^2")
    p(f"Queueing model:       Q(N) = {Q_SLOPE}*(N-1)*service  (measured, job 155248)")
    p(f"Concurrency:          N = {N_CONCURRENT} (the 4 workload requests share the GPU)")
    p(f"c_queued/c_iso factor at N={N_CONCURRENT}:  {queue_factor(N_CONCURRENT):.3f}")
    p("Compute window: c = T_prefill(uncached)/L, uncached = ctx*(1-hit).")
    p("Scheduler is naive (allocates from r*_iso in all cells); physics uses c_actual.")
    p("Added TTFT measured vs fixed baseline L*c_iso.")
    p("")

    reqs = build_workload()

    # r*_iso / r*_queued table (deliverable)
    p("-" * 80)
    p("HOW WRONG r*_i GETS UNDER QUEUEING:  r*_iso / r*_queued = 1 + 0.85*(N-1)")
    p("-" * 80)
    p(f"  {'N':>4s}  {'r*_iso/r*_queued':>18s}   interpretation")
    for n in (1, 2, 4, 8, 16):
        f = queue_factor(n)
        p(f"  {n:>4d}  {f:>18.3f}   r*_iso overestimates useful BW by {f:.2f}x")
    p("")
    p("  (Under measured batching all N share the GPU equally, so every request")
    p("   in the wave sees the same factor — position in the queue is degenerate.)")
    p("")

    # Per-request compute windows
    p("-" * 80)
    p("PER-REQUEST COMPUTE WINDOWS (our vLLM coeffs)")
    p("-" * 80)
    p(f"  {'Request':>12s}  {'cached':>7s}  {'uncached':>8s}  "
      f"{'c_iso ms':>9s}  {'c_queued ms':>11s}  {'r*_iso Gbps':>12s}")
    qf = queue_factor(N_CONCURRENT)
    for r in reqs:
        p(f"  {r.label:>12s}  {r.cached_tokens:>7d}  {r.uncached_tokens:>8d}  "
          f"{r.c_iso*1000:>9.2f}  {r.c_iso*qf*1000:>11.2f}  "
          f"{Bps_to_gbps(r.r_star):>12.2f}")
    p("")

    c_iso_list = [r.c_iso for r in reqs]
    c_q_list = [r.c_iso * qf for r in reqs]

    results = {}   # results[wl][cell][policy] = added TTFT ms
    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        const = make_constant_trace(cap_Bps)
        bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                     episode_sec=BIMODAL_EPISODE)

        cells = {
            "(1) c_iso+const":  (c_iso_list, const),
            "(2) c_iso+bimod":  (c_iso_list, bimodal),
            "(3) c_queued+const": (c_q_list, const),
            "(4) c_queued+bimod": (c_q_list, bimodal),
        }
        cell_names = list(cells.keys())

        res = {cell: {} for cell in cell_names}
        for cell, (c_list, trace) in cells.items():
            for pol in POLICY_ORDER:
                res[cell][pol] = run_cell(reqs, c_list, pol, trace)
        results[wl] = res

        p("=" * 80)
        p(f"WORKLOAD {wl}  (cap = {cap_gbps} Gbps)   —  Aggregate ADDED TTFT (ms)")
        p("=" * 80)
        hdr = f"{'Policy':<16s}" + "".join(f"  {c:>19s}" for c in cell_names)
        p(hdr)
        p("-" * len(hdr))
        for pol in POLICY_ORDER:
            row = f"{pol:<16s}"
            for cell in cell_names:
                row += f"  {res[cell][pol]:>19.1f}"
            p(row)
        p("")

        # Superadditivity check per policy:
        #   independent prediction of cell4 = cell2 + cell3 - cell1
        #   interaction = cell4 - (cell2 + cell3 - cell1)
        #               = cell4 - cell2 - cell3 + cell1
        c1, c2, c3, c4 = cell_names
        p("SUPERADDITIVITY CHECK  (added TTFT over the cell-1 reference)")
        p("  independent-effects prediction of (4):  pred = (2)+(3)-(1)")
        p("  interaction = (4) - pred ;  >0 SUPERadditive (compound), "
          "<0 SUBadditive (offset)")
        p(f"  {'Policy':<16s}  {'(4) actual':>11s}  {'pred indep':>11s}  "
          f"{'interaction':>12s}  {'(4)/pred':>9s}   verdict")
        for pol in POLICY_ORDER:
            a1, a2, a3, a4 = (res[c1][pol], res[c2][pol],
                              res[c3][pol], res[c4][pol])
            pred = a2 + a3 - a1
            inter = a4 - pred
            ratio = a4 / pred if pred else float('nan')
            verdict = "SUPER-additive" if inter > 1e-6 else (
                      "sub-additive" if inter < -1e-6 else "additive")
            p(f"  {pol:<16s}  {a4:>11.1f}  {pred:>11.1f}  {inter:>+12.1f}  "
              f"{ratio:>9.3f}   {verdict}")
        p("")

        # Inversion: Stall-opt vs Equal, at each compute assumption.
        p("STALL-OPT / EQUAL INVERSION under bimodal:")
        so2_deg = res[c2]["Stall-opt"] / res[c1]["Stall-opt"]
        eq2_deg = res[c2]["Equal"]     / res[c1]["Equal"]
        so4_deg = res[c4]["Stall-opt"] / res[c3]["Stall-opt"]
        eq4_deg = res[c4]["Equal"]     / res[c3]["Equal"]
        p(f"  cell (2) isolated c:  Stall-opt deg {so2_deg:.3f}x  "
          f"Equal deg {eq2_deg:.3f}x  -> "
          f"{'inversion HOLDS' if so2_deg > eq2_deg else 'no inversion'}")
        p(f"  cell (4) queued  c:  Stall-opt deg {so4_deg:.3f}x  "
          f"Equal deg {eq4_deg:.3f}x  -> "
          f"{'inversion HOLDS' if so4_deg > eq4_deg else 'no inversion'}")
        gap2 = so2_deg - eq2_deg
        gap4 = so4_deg - eq4_deg
        if abs(gap4) > abs(gap2) + 1e-6:
            change = "STRONGER"
        elif abs(gap4) < abs(gap2) - 1e-6:
            change = "WEAKER"
        else:
            change = "UNCHANGED"
        p(f"  Inversion gap (SO-EQ deg):  cell2 {gap2:+.3f}  cell4 {gap4:+.3f}"
          f"  -> inversion is {change} under queueing")
        p("")

    text = "\n".join(out) + "\n"
    print(text)
    path = out_dir / "results.txt"
    path.write_text(text)
    print(f"Output saved to {path}")


if __name__ == "__main__":
    main()
