#!/usr/bin/env python3
"""Re-run Experiments B and C with corrected Q_SLOPE=1.0, report old vs new.

No GPU needed — pure simulation. Runs both experiments at the old slope (0.85)
and the new slope (1.0) and prints side-by-side comparisons.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from stallopt import (
    L, BYTES_PER_TOKEN_PER_LAYER, POLICIES, POLICY_ORDER,
    Bps_to_gbps, gbps_to_Bps,
    make_constant_trace, make_bimodal_trace, BandwidthTrace,
)
from queue_compute import (
    A_VLLM, B_VLLM, WORKLOAD_SPEC, t_prefill,
    QReq, build_workload, sim_one, run_cell,
    queue_factor, N_CONCURRENT, WORKLOADS, EPOCH, N_STARTS,
    BIMODAL_DEPTH, BIMODAL_EPISODE,
)
from queue_feedback import (
    simulate, CAPACITY, mean_service, CAP_GBPS,
    _single_episode_trace, _meanN_window, _drain_time, _mean,
)


def run_experiment_b(slope):
    """Run fixed-N four-cell experiment at given slope. Returns results dict."""
    qf = 1.0 + slope * (N_CONCURRENT - 1)
    reqs = build_workload()
    c_iso_list = [r.c_iso for r in reqs]
    c_q_list = [r.c_iso * qf for r in reqs]

    results = {}
    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        const = make_constant_trace(cap_Bps)
        bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                     episode_sec=BIMODAL_EPISODE)
        cells = {
            "c1": (c_iso_list, const),
            "c2": (c_iso_list, bimodal),
            "c3": (c_q_list, const),
            "c4": (c_q_list, bimodal),
        }
        res = {}
        for cell, (c_list, trace) in cells.items():
            res[cell] = {}
            for pol in POLICY_ORDER:
                res[cell][pol] = run_cell(reqs, c_list, pol, trace)
        results[wl] = res
    return results, qf


def run_experiment_c(slope):
    """Run feedback simulation at given slope. Returns per-load results."""
    import queue_compute
    import queue_feedback
    old_qc = queue_compute.Q_SLOPE
    old_qf = queue_feedback.Q_SLOPE
    queue_compute.Q_SLOPE = slope
    queue_feedback.Q_SLOPE = slope
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    const = make_constant_trace(cap_Bps, duration=100000.0)
    bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                 episode_sec=BIMODAL_EPISODE, n_cycles=200)
    HORIZON = 900.0
    SEED = 7
    load_fracs = [0.30, 0.50, 0.70, 0.85]

    results = {}
    for frac in load_fracs:
        rate = frac * CAPACITY
        row = {}
        for label, trace_obj, queued in [
            ("c1", const, False), ("c2", bimodal, False),
            ("c3", const, True), ("c4", bimodal, True),
        ]:
            cell_results = {}
            for pol in POLICY_ORDER:
                added, _, counts = simulate(rate, trace_obj, pol, HORIZON, SEED,
                                            queued=queued)
                cell_results[pol] = _mean(added)
            row[label] = cell_results
        results[frac] = row

    # Metastability
    meta = {}
    EPISODE_DUR = 60.0
    for frac in [0.70, 0.85]:
        rate = frac * CAPACITY
        ep_trace = _single_episode_trace(cap_Bps, t_start=300.0, dur=EPISODE_DUR,
                                         depth=BIMODAL_DEPTH, horizon=HORIZON)
        _, nlog, _ = simulate(rate, ep_trace, "Stall-opt", HORIZON, SEED,
                              queued=True, log_N=True)
        before = _meanN_window(nlog, 200, 300)
        during = _meanN_window(nlog, 300, 360)
        peak = max((n for t, n, _ in nlog if 300 <= t < 420), default=0)
        drain = _drain_time(nlog, 360.0, before)
        meta[frac] = {"preN": before, "duringN": during, "peakN": peak,
                       "drain_s": drain, "persistence": drain / EPISODE_DUR if drain else None}

    queue_compute.Q_SLOPE = old_qc
    queue_feedback.Q_SLOPE = old_qf
    return results, meta


def main():
    print("=" * 80)
    print("EXPERIMENTS B & C: OLD (slope=0.85) vs NEW (slope=1.0)")
    print("=" * 80)
    print()

    # ── Experiment B ──
    print("EXPERIMENT B — Fixed N=4, four-cell")
    print("-" * 80)
    res_old, qf_old = run_experiment_b(0.85)
    res_new, qf_new = run_experiment_b(1.0)

    print(f"  c_wall factor: old = {qf_old:.2f}x c_iso, new = {qf_new:.2f}x c_iso")
    print()

    for wl in ["A", "B"]:
        print(f"  Workload {wl}:")
        print(f"    {'Policy':<16s}  {'OLD interaction':>15s}  {'NEW interaction':>15s}  {'Change':>8s}  {'Sign holds?':>12s}")
        for pol in POLICY_ORDER:
            for label, res in [("OLD", res_old), ("NEW", res_new)]:
                pass
            old_inter = (res_old[wl]["c4"][pol] - (res_old[wl]["c2"][pol] +
                         res_old[wl]["c3"][pol] - res_old[wl]["c1"][pol]))
            new_inter = (res_new[wl]["c4"][pol] - (res_new[wl]["c2"][pol] +
                         res_new[wl]["c3"][pol] - res_new[wl]["c1"][pol]))
            change = new_inter - old_inter
            old_sign = "sub" if old_inter < -1 else ("super" if old_inter > 1 else "~0")
            new_sign = "sub" if new_inter < -1 else ("super" if new_inter > 1 else "~0")
            holds = "YES" if old_sign == new_sign else "CHANGED"
            print(f"    {pol:<16s}  {old_inter:>+15.1f}  {new_inter:>+15.1f}  {change:>+8.1f}  {holds:>12s}")
        print()

        # Also show absolute cell values for Stall-opt
        print(f"  Workload {wl} — Stall-opt absolute added TTFT (ms):")
        print(f"    {'Cell':<20s}  {'OLD':>10s}  {'NEW':>10s}  {'Delta':>10s}")
        for cell in ["c1", "c2", "c3", "c4"]:
            ov = res_old[wl][cell]["Stall-opt"]
            nv = res_new[wl][cell]["Stall-opt"]
            print(f"    {cell:<20s}  {ov:>10.1f}  {nv:>10.1f}  {nv-ov:>+10.1f}")
        print()

    # ── Experiment C ──
    print()
    print("EXPERIMENT C — Feedback (N is output)")
    print("-" * 80)

    print("  Running old slope...")
    c_old, meta_old = run_experiment_c(0.85)
    print("  Running new slope...")
    c_new, meta_new = run_experiment_c(1.0)

    print()
    print("  Superadditivity interaction (Stall-opt, ms):")
    print(f"    {'Load':>6s}  {'OLD':>10s}  {'NEW':>10s}  {'Change':>10s}  {'Sign holds?':>12s}")
    for frac in [0.30, 0.50, 0.70, 0.85]:
        for label, res in [("old", c_old), ("new", c_new)]:
            pass
        old_inter = (c_old[frac]["c4"]["Stall-opt"] -
                     (c_old[frac]["c2"]["Stall-opt"] + c_old[frac]["c3"]["Stall-opt"] -
                      c_old[frac]["c1"]["Stall-opt"]))
        new_inter = (c_new[frac]["c4"]["Stall-opt"] -
                     (c_new[frac]["c2"]["Stall-opt"] + c_new[frac]["c3"]["Stall-opt"] -
                      c_new[frac]["c1"]["Stall-opt"]))
        change = new_inter - old_inter
        old_sign = "super" if old_inter > 1 else ("sub" if old_inter < -1 else "~0")
        new_sign = "super" if new_inter > 1 else ("sub" if new_inter < -1 else "~0")
        holds = "YES" if old_sign == new_sign else "CHANGED"
        print(f"    {frac:>5.0%}  {old_inter:>+10.1f}  {new_inter:>+10.1f}  {change:>+10.1f}  {holds:>12s}")

    print()
    print("  All policies at 70% load (interaction, ms):")
    print(f"    {'Policy':<16s}  {'OLD':>10s}  {'NEW':>10s}  {'Change':>10s}")
    for pol in POLICY_ORDER:
        old_i = (c_old[0.70]["c4"][pol] - (c_old[0.70]["c2"][pol] +
                 c_old[0.70]["c3"][pol] - c_old[0.70]["c1"][pol]))
        new_i = (c_new[0.70]["c4"][pol] - (c_new[0.70]["c2"][pol] +
                 c_new[0.70]["c3"][pol] - c_new[0.70]["c1"][pol]))
        print(f"    {pol:<16s}  {old_i:>+10.1f}  {new_i:>+10.1f}  {new_i-old_i:>+10.1f}")

    print()
    print("  Metastability drain times (Stall-opt, queued):")
    print(f"    {'Load':>6s}  {'OLD drain(s)':>12s}  {'OLD persist':>12s}  {'NEW drain(s)':>12s}  {'NEW persist':>12s}  {'Direction':>10s}")
    for frac in [0.70, 0.85]:
        od = meta_old[frac]["drain_s"]
        op = meta_old[frac]["persistence"]
        nd = meta_new[frac]["drain_s"]
        np_ = meta_new[frac]["persistence"]
        od_s = f"{od:.0f}" if od else "N/A"
        op_s = f"{op:.2f}x" if op else "N/A"
        nd_s = f"{nd:.0f}" if nd else "N/A"
        np_s = f"{np_:.2f}x" if np_ else "N/A"
        if nd and od:
            direction = "STRONGER" if nd > od else "WEAKER"
        elif nd and not od:
            direction = "STRONGER"
        elif not nd and od:
            direction = "WEAKER"
        else:
            direction = "SAME"
        print(f"    {frac:>5.0%}  {od_s:>12s}  {op_s:>12s}  {nd_s:>12s}  {np_s:>12s}  {direction:>10s}")

    print()
    print("  Metastability detail:")
    for frac in [0.70, 0.85]:
        print(f"    {frac:.0%}: OLD preN={meta_old[frac]['preN']:.1f} duringN={meta_old[frac]['duringN']:.1f} peakN={meta_old[frac]['peakN']}")
        print(f"    {frac:.0%}: NEW preN={meta_new[frac]['preN']:.1f} duringN={meta_new[frac]['duringN']:.1f} peakN={meta_new[frac]['peakN']}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("  Experiment B: sub-additivity sign preserved? Check above.")
    print("  Experiment C: superadditivity sign preserved? Check above.")
    print("  Metastability: drain times moved which direction? Check above.")
    print("  Expectation was: both findings STRENGTHEN with slope 1.0.")


if __name__ == "__main__":
    main()
