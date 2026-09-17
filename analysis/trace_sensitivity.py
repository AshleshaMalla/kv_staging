#!/usr/bin/env python3
"""Step 2a: Time-varying bandwidth simulation experiment.

Runs 4 trace types x 5 policies x 2 workloads, then sweeps epoch length
on the bimodal trace.  Reports degradation ratios and epoch sensitivity.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stallopt import (
    POLICY_ORDER, Bps_to_gbps, gbps_to_Bps, get_workload_requests,
    make_bimodal_trace, make_constant_trace, make_quiescent_trace,
    make_uniform_trace, run_trace_experiment,
)

WORKLOADS = {"A": 80, "B": 50}
TRACE_NAMES = ["Constant", "Quiescent", "Bimodal", "Uniform"]
DEFAULT_EPOCH = 1.0
EPOCH_SWEEP = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0]


def make_traces(cap_Bps):
    return {
        "Constant":  make_constant_trace(cap_Bps),
        "Quiescent": make_quiescent_trace(cap_Bps),
        "Bimodal":   make_bimodal_trace(cap_Bps),
        "Uniform":   make_uniform_trace(cap_Bps),
    }


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/raw") / f"trace_sensitivity_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    requests = get_workload_requests()

    lines = []

    def p(s=""):
        lines.append(s)
        print(s)

    p("=" * 80)
    p("Step 2a: Time-Varying Bandwidth Simulation")
    p(f"Timestamp: {ts}")
    p(f"Default epoch: {DEFAULT_EPOCH} s, start-time averaging: 200 starts")
    p("=" * 80)

    all_results = {}

    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        traces = make_traces(cap_Bps)

        p()
        p(f"WORKLOAD {wl}  (cap = {cap_gbps} Gbps)")
        p("-" * 72)

        p("Trace statistics:")
        for tn in TRACE_NAMES:
            tr = traces[tn]
            p(f"  {tn:<12s}  mean = {Bps_to_gbps(tr.mean):6.2f} Gbps  "
              f"CV = {tr.cv * 100:5.2f}%  dur = {tr.duration:.0f}s")
        p()

        p(f"Aggregate delta TTFT (ms) — epoch = {DEFAULT_EPOCH} s:")
        hdr = f"{'Policy':<18s}" + "".join(f"  {tn:>12s}" for tn in TRACE_NAMES)
        p(hdr)
        p("-" * len(hdr))

        wl_results = {}
        for policy in POLICY_ORDER:
            wl_results[policy] = {}
            for tn in TRACE_NAMES:
                val = run_trace_experiment(requests, policy,
                                          traces[tn], DEFAULT_EPOCH)
                wl_results[policy][tn] = val
            row = f"{policy:<18s}"
            row += "".join(f"  {wl_results[policy][tn]:>12.1f}"
                           for tn in TRACE_NAMES)
            p(row)

        p()
        p("Degradation ratio  (trace / Constant):")
        hdr2 = f"{'Policy':<18s}" + "".join(
            f"  {tn:>12s}" for tn in TRACE_NAMES[1:])
        p(hdr2)
        p("-" * len(hdr2))
        for policy in POLICY_ORDER:
            base = wl_results[policy]["Constant"]
            row = f"{policy:<18s}"
            for tn in TRACE_NAMES[1:]:
                ratio = wl_results[policy][tn] / base if base > 0 else 0
                row += f"  {ratio:>12.6f}"
            p(row)
        p()

        all_results[wl] = wl_results

    # -- Epoch sensitivity (bimodal only) --
    p()
    p("=" * 80)
    p("EPOCH SENSITIVITY — Bimodal trace")
    p("=" * 80)

    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        bimodal = make_bimodal_trace(cap_Bps)

        p()
        p(f"WORKLOAD {wl}  (cap = {cap_gbps} Gbps)")
        hdr = f"{'Epoch (s)':<12s}" + "".join(
            f"  {pol:>14s}" for pol in POLICY_ORDER)
        p(hdr)
        p("-" * len(hdr))

        for epoch in EPOCH_SWEEP:
            row = f"{epoch:<12.2f}"
            for policy in POLICY_ORDER:
                val = run_trace_experiment(requests, policy, bimodal, epoch)
                row += f"  {val:>14.1f}"
            p(row)
        p()

    # -- Key findings --
    p()
    p("=" * 80)
    p("KEY FINDINGS")
    p("=" * 80)
    p()

    for wl in WORKLOADS:
        r = all_results[wl]
        eq_base = r["Equal"]["Constant"]
        so_base = r["Stall-opt"]["Constant"]
        cso_base = r["Cal. Stall-opt"]["Constant"]

        eq_bi = r["Equal"]["Bimodal"] / eq_base if eq_base else 0
        so_bi = r["Stall-opt"]["Bimodal"] / so_base if so_base else 0
        cso_bi = r["Cal. Stall-opt"]["Bimodal"] / cso_base if cso_base else 0

        eq_uni = r["Equal"]["Uniform"] / eq_base if eq_base else 0
        so_uni = r["Stall-opt"]["Uniform"] / so_base if so_base else 0

        p(f"Workload {wl}:")
        p(f"  Bimodal degradation:  Equal {eq_bi:.4f}x  "
          f"Stall-opt {so_bi:.4f}x  Cal.Stall-opt {cso_bi:.4f}x")
        more = "MORE" if so_bi > eq_bi else "LESS"
        p(f"  Stall-opt degrades {more} than Equal under bimodal "
          f"({so_bi:.4f} vs {eq_bi:.4f})")
        p(f"  Uniform degradation:  Equal {eq_uni:.4f}x  "
          f"Stall-opt {so_uni:.4f}x")
        gap = abs(so_bi - so_uni)
        p(f"  Bimodal vs Uniform gap for Stall-opt: "
          f"{gap:.4f} (sustained drops vs rapid fluctuation)")
        p()

    out_path = out_dir / "results.txt"
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    p(f"Output saved to {out_path}")


if __name__ == "__main__":
    main()
