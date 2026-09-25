#!/usr/bin/env python3
"""Conditional-overlap sweep: corrected implementation.

The original sim_one_conditional had inverted labels: P=0 gave full overlap
and P=32 gave no overlap. This script uses explicit parameters and sweeps
all five policies across both workloads.

Model:
  T = X_0 + sum_{l=1}^{L-1} step(l) + C_{final}

  With overlap (pipeline): step(l) = max(X_l, C_l)
  Without overlap (sequential): step(l) = X_l + C_l

  overlap_layers = number of layers (from the END) that get overlap.
  Layers 1..L-1-overlap_layers are sequential.
  Layers L-1-overlap_layers+1..L-1 get max(X, C).
  overlap_layers=0: fully sequential (no overlap at all)
  overlap_layers=L-1=31: original model (full unconditional overlap)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import queue_compute
queue_compute.Q_SLOPE = 1.0

from queue_compute import (
    queue_factor, build_workload, N_CONCURRENT, WORKLOADS,
    EPOCH, N_STARTS, BIMODAL_DEPTH, BIMODAL_EPISODE,
)
from stallopt import (
    L, POLICIES, POLICY_ORDER, gbps_to_Bps, _transfer_time,
    make_constant_trace, make_bimodal_trace,
)


def sim_one_overlap(alloc_reqs, idx, c_actual, policy_fn, trace, epoch, t0,
                    overlap_layers):
    """TTFT with parameterized overlap.
    overlap_layers: how many of the 31 inter-layer steps use max(X, C).
    The rest use X + C (sequential).
    overlap_layers=0: no overlap anywhere, fully sequential.
    overlap_layers=31: full overlap everywhere (original model).
    """
    s = alloc_reqs[idx].s
    t = t0

    # Layer 0: always just transfer
    t += _transfer_time(s, idx, alloc_reqs, policy_fn, trace, epoch, t0, t)

    n_sequential = (L - 1) - overlap_layers
    for layer_idx in range(1, L):
        dt = _transfer_time(s, idx, alloc_reqs, policy_fn, trace, epoch, t0, t)
        if layer_idx <= n_sequential:
            t += dt + c_actual
        else:
            t += max(dt, c_actual)

    t += c_actual
    return t - t0


def run_cell_overlap(alloc_reqs, c_actual_list, policy_name, trace,
                     overlap_layers, epoch=EPOCH, n_starts=N_STARTS):
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
            ttft_s = sim_one_overlap(alloc_reqs, idx, c_actual_list[idx],
                                     policy_fn, trace, epoch, t0, overlap_layers)
            agg += (ttft_s - L * req.c_iso) * 1000.0
        total += agg
    return total / len(starts)


def main():
    print("=" * 80)
    print("CONDITIONAL-OVERLAP SWEEP — corrected (all policies, both workloads)")
    print("=" * 80)
    print(f"  overlap_layers: 0 = fully sequential, 31 = full overlap (original model)")
    print(f"  L = {L} layers, N = {N_CONCURRENT}")
    print()

    qf = queue_factor(N_CONCURRENT)
    reqs = build_workload()
    c_iso_list = [r.c_iso for r in reqs]
    c_q_list = [r.c_iso * qf for r in reqs]

    overlap_values = [0, 1, 2, 4, 8, 16, 24, 31]

    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        const = make_constant_trace(cap_Bps)
        bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                     episode_sec=BIMODAL_EPISODE)

        print(f"  Workload {wl} (cap = {cap_gbps} Gbps)")
        print(f"  {'Policy':<16s}  {'Overlap':>7s}  {'c1':>10s}  {'c2':>10s}  "
              f"{'c3':>10s}  {'c4':>10s}  {'Interaction':>12s}  {'Sign':>12s}")
        print("  " + "-" * 95)

        for pol in POLICY_ORDER:
            for ov in overlap_values:
                c1 = run_cell_overlap(reqs, c_iso_list, pol, const, ov)
                c2 = run_cell_overlap(reqs, c_iso_list, pol, bimodal, ov)
                c3 = run_cell_overlap(reqs, c_q_list, pol, const, ov)
                c4 = run_cell_overlap(reqs, c_q_list, pol, bimodal, ov)
                inter = c4 - (c2 + c3 - c1)
                sign = "sub" if inter < -1 else ("SUPER" if inter > 1 else "~0")
                label = f"{ov}/31"
                print(f"  {pol:<16s}  {label:>7s}  {c1:>10.1f}  {c2:>10.1f}  "
                      f"{c3:>10.1f}  {c4:>10.1f}  {inter:>+12.1f}  {sign:>12s}")
            print()
        print()

    # Focused diagnostic: Equal at overlap=0 (no pipeline at all)
    print("=" * 80)
    print("DIAGNOSTIC: Equal at overlap=0 (no pipeline, no overlap)")
    print("=" * 80)
    print("  If Equal's interaction is 0, the sub-additivity in other policies")
    print("  comes from c-dependent allocations. If nonzero, there is another channel.")
    print()
    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        const = make_constant_trace(cap_Bps)
        bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                     episode_sec=BIMODAL_EPISODE)
        c1 = run_cell_overlap(reqs, c_iso_list, "Equal", const, 0)
        c2 = run_cell_overlap(reqs, c_iso_list, "Equal", bimodal, 0)
        c3 = run_cell_overlap(reqs, c_q_list, "Equal", const, 0)
        c4 = run_cell_overlap(reqs, c_q_list, "Equal", bimodal, 0)
        inter = c4 - (c2 + c3 - c1)
        print(f"  WL-{wl} Equal overlap=0:  c1={c1:.1f}  c2={c2:.1f}  c3={c3:.1f}  c4={c4:.1f}")
        print(f"    interaction = {inter:+.1f} ms")
        if abs(inter) < 1:
            print(f"    → ZERO: sub-additivity in other policies is from c-dependent allocation")
        else:
            print(f"    → NONZERO: another channel exists")

        # Also trace: does c affect transfer time at overlap=0?
        # At overlap=0, step = dt + c_actual. dt comes from _transfer_time which
        # uses alloc_reqs[idx].s and the policy allocation. For Equal, allocation
        # = cap/N, independent of c. So dt should be identical across c_iso and
        # c_queued cells. The only difference is c_actual itself.
        # T_sequential = X_0 + sum(dt_l + c_actual) + c_actual
        #              = X_0 + (L-1)*dt_avg + (L-1)*c_actual + c_actual
        #              = X_0 + (L-1)*dt_avg + L*c_actual
        # Added TTFT = T - L*c_iso = X_0 + (L-1)*dt_avg + L*(c_actual - c_iso)
        # For cells 1,2: c_actual = c_iso, so added = X_0 + (L-1)*dt_avg
        # For cells 3,4: c_actual = c_q = qf*c_iso, so added = X_0 + (L-1)*dt_avg + L*(qf-1)*c_iso
        # Interaction = c4 - (c2 + c3 - c1) = [X0+dt_sum+L*(qf-1)*c] - ([X0+dt_sum] + [X0+dt_sum+L*(qf-1)*c] - [X0+dt_sum])
        # = 0 if dt is the same across cells.
        # BUT: dt depends on bandwidth via _transfer_time, which uses the trace.
        # The trace is the same for (1,3) [constant] and (2,4) [bimodal].
        # For Equal, allocation is cap/N independent of anything request-specific.
        # So dt IS the same for matched cells (1 vs 3, 2 vs 4).
        # Therefore interaction SHOULD be exactly 0 for Equal at overlap=0.
        print()


if __name__ == "__main__":
    main()
