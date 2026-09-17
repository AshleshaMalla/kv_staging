#!/usr/bin/env python3
"""Corrections 1 and 2 to the time-varying bandwidth findings.

Both corrections use the PAPER's compute window (Request.c = t_total/L from
Table A8), unchanged — they re-report the existing Step 2a findings more
honestly.  The queue-aware compute model (our vLLM coefficients) lives in
queue_compute.py, a separate task.

Correction 1 — Bound the uniform-trace result.
  The 14.5x Stall-opt degradation under uniform[0, 2*mean] is dominated by
  near-zero samples.  Re-run with floors at 10/25/50% of mean, and decompose
  what fraction of the unfloored effect comes from samples below each floor.

Correction 2 — Label bimodal parameters honestly.
  Depth (R=2.3) is measured; episode duration and inter-episode gap are swept.
  Re-report bimodal as a sensitivity surface over (low_duration, high_gap) and
  locate where the Stall-opt/Equal inversion holds vs disappears.
"""

import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stallopt import (
    BandwidthTrace, POLICY_ORDER, Bps_to_gbps, gbps_to_Bps,
    get_workload_requests, make_constant_trace, run_trace_experiment,
)

WORKLOADS = {"A": 80, "B": 50}
EPOCH = 1.0
N_STARTS = 200
MEASURED_DEPTH = 2.3   # R = B_high/B_low; the one measured degradation episode


# ─────────────────────────────────────────────────────────────────────────
# Correction 1: floored uniform traces
# ─────────────────────────────────────────────────────────────────────────

def make_uniform_floored(mean_Bps, floor_frac, interval_sec=0.1,
                         duration_sec=600.0, seed=42):
    """Uniform bandwidth in [floor_frac*mean, (2-floor_frac)*mean].

    Symmetric truncation preserves the time-mean at `mean_Bps` for every
    floor_frac.  floor_frac=0 reproduces the literature model uniform[0, 2mean]
    (Cake §5.6 samples bandwidth uniformly 0-25 Gbps)."""
    lo = floor_frac * mean_Bps
    hi = (2.0 - floor_frac) * mean_Bps
    rng = random.Random(seed)
    times, bws = [], []
    t = 0.0
    while t < duration_sec:
        times.append(t)
        bws.append(max(1.0, rng.uniform(lo, hi)))
        t += interval_sec
    return BandwidthTrace(times, bws)


def make_uniform_clamped(mean_Bps, floor_frac, interval_sec=0.1,
                         duration_sec=600.0, seed=42):
    """The SAME random draws as unfloored uniform[0, 2mean], but every sample
    below floor_frac*mean is raised to the floor.  Used to attribute the effect:
    the drop in degradation from unfloored to clamped is exactly the
    contribution of sub-floor samples (the mean shifts up slightly — that is
    the point: those low samples were carrying the tail)."""
    floor = floor_frac * mean_Bps
    rng = random.Random(seed)
    times, bws = [], []
    t = 0.0
    while t < duration_sec:
        times.append(t)
        raw = max(1.0, rng.uniform(0.0, 2.0 * mean_Bps))
        bws.append(max(raw, floor))
        t += interval_sec
    return BandwidthTrace(times, bws)


def correction_1(out):
    p = out.append
    p("=" * 80)
    p("CORRECTION 1 — Bounding the uniform-trace result")
    p("=" * 80)
    p("")
    p("The unfloored number (floor=0%, uniform[0,2*mean]) is the literature's")
    p("assumed model (Cake §5.6: bandwidth ~ Uniform(0, 25 Gbps)), NOT a")
    p("distribution any storage system produces.  It is kept as reference only.")
    p("")

    floors = [0.0, 0.10, 0.25, 0.50]
    requests = get_workload_requests()

    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        const = make_constant_trace(cap_Bps)

        # Constant-BW baselines per policy
        base = {pol: run_trace_experiment(requests, pol, const, EPOCH, N_STARTS)
                for pol in POLICY_ORDER}

        p(f"WORKLOAD {wl}  (cap = {cap_gbps} Gbps)")
        p("-" * 72)

        # (a) Truncated-uniform floors — degradation ratio (trace/constant)
        p("(a) Truncated uniform[floor*mean, (2-floor)*mean], mean preserved:")
        p("    Degradation ratio = delta-TTFT(trace) / delta-TTFT(constant)")
        hdr = f"    {'Floor':>7s}" + "".join(f"  {pol:>14s}" for pol in POLICY_ORDER)
        p(hdr)
        p("    " + "-" * (len(hdr) - 4))
        floored_ratio = {}
        for f in floors:
            tr = make_uniform_floored(cap_Bps, f)
            row = f"    {f*100:>5.0f}% "
            floored_ratio[f] = {}
            for pol in POLICY_ORDER:
                val = run_trace_experiment(requests, pol, tr, EPOCH, N_STARTS)
                ratio = val / base[pol] if base[pol] else 0.0
                floored_ratio[f][pol] = ratio
                row += f"  {ratio:>14.3f}"
            p(row)
        p("")

        # (b) Attribution: clamp sub-floor samples up to the floor on the SAME
        #     draws as unfloored.  Fraction of the unfloored EXCESS (ratio-1)
        #     removed by clamping = fraction of effect from samples below floor.
        p("(b) Fraction of the unfloored effect (ratio-1) contributed by")
        p("    samples BELOW each floor (clamp-and-measure on identical draws):")
        hdr2 = f"    {'Floor':>7s}" + "".join(f"  {pol:>14s}" for pol in POLICY_ORDER)
        p(hdr2)
        p("    " + "-" * (len(hdr2) - 4))
        excess0 = {pol: floored_ratio[0.0][pol] - 1.0 for pol in POLICY_ORDER}
        for f in floors[1:]:
            tr = make_uniform_clamped(cap_Bps, f)
            row = f"    <{f*100:>4.0f}% "
            for pol in POLICY_ORDER:
                val = run_trace_experiment(requests, pol, tr, EPOCH, N_STARTS)
                excess_clamped = (val / base[pol] if base[pol] else 0.0) - 1.0
                if excess0[pol] > 1e-9:
                    frac = (excess0[pol] - excess_clamped) / excess0[pol]
                else:
                    frac = 0.0
                row += f"  {frac*100:>13.1f}%"
            p(row)
        p("")
        p(f"    Stall-opt unfloored ratio: {floored_ratio[0.0]['Stall-opt']:.2f}x"
          f"  ->  at 10% floor: {floored_ratio[0.10]['Stall-opt']:.2f}x"
          f"  ->  at 25%: {floored_ratio[0.25]['Stall-opt']:.2f}x"
          f"  ->  at 50%: {floored_ratio[0.50]['Stall-opt']:.2f}x")
        p("")


# ─────────────────────────────────────────────────────────────────────────
# Correction 2: bimodal sensitivity surface over (duration, gap)
# ─────────────────────────────────────────────────────────────────────────

def make_bimodal_asym(mean_Bps, low_dur, high_gap, depth=MEASURED_DEPTH,
                      min_total=1800.0):
    """Bimodal trace with independent low-episode duration and high gap.

    Preserves BOTH the measured depth (R = B_high/B_low = depth) and the
    time-mean (= mean_Bps), by solving for B_low/B_high given the duty cycle:
        duty = low_dur / (low_dur + high_gap)
        mean = duty*B_low + (1-duty)*B_high,   B_high = depth*B_low
      => B_low = mean / (duty + (1-duty)*depth)

    n_cycles chosen so total duration >= min_total (for start-averaging)."""
    duty = low_dur / (low_dur + high_gap)
    B_low = mean_Bps / (duty + (1.0 - duty) * depth)
    B_high = depth * B_low
    period = low_dur + high_gap
    n_cycles = max(8, int(min_total / period) + 1)

    times, bws = [], []
    t = 0.0
    for _ in range(n_cycles):
        times.append(t); bws.append(B_high)   # gap (high) first
        t += high_gap
        times.append(t); bws.append(B_low)     # then low episode
        t += low_dur
    times.append(t); bws.append(B_high)
    return BandwidthTrace(times, bws)


def correction_2(out):
    p = out.append
    p("=" * 80)
    p("CORRECTION 2 — Bimodal as a (duration, gap) sensitivity surface")
    p("=" * 80)
    p("")
    p(f"Depth R = {MEASURED_DEPTH} (B_high/B_low) is MEASURED (one observed")
    p("degradation episode, ~2.3x).  Low-episode DURATION and inter-episode")
    p("GAP are SWEPT — we saw exactly one episode, so their values are")
    p("assumptions.  Mean is held at the cap for every (duration, gap).")
    p("")
    p("Cell = Stall-opt degradation ratio / Equal degradation ratio.")
    p("  > 1  => INVERSION HOLDS  (Stall-opt degrades MORE than Equal)")
    p("  < 1  => inversion GONE   (Equal degrades more)")
    p("")

    low_durs = [2, 5, 15, 30, 60, 120]        # seconds, low-BW episode length
    gaps = [5, 15, 30, 60, 120, 300, 600]     # seconds, high-BW inter-episode

    requests = get_workload_requests()

    for wl, cap_gbps in WORKLOADS.items():
        cap_Bps = gbps_to_Bps(cap_gbps)
        const = make_constant_trace(cap_Bps)
        eq_base = run_trace_experiment(requests, "Equal", const, EPOCH, N_STARTS)
        so_base = run_trace_experiment(requests, "Stall-opt", const, EPOCH, N_STARTS)

        p(f"WORKLOAD {wl}  (cap = {cap_gbps} Gbps)")
        p("-" * 72)
        dg = "dur\\gap"
        hdr = f"  {dg:>8s}" + "".join(f"  {g:>6d}s" for g in gaps)
        p(hdr)
        p("  " + "-" * (len(hdr) - 2))
        surface = {}
        for d in low_durs:
            row = f"  {d:>7d}s"
            surface[d] = {}
            for g in gaps:
                tr = make_bimodal_asym(cap_Bps, d, g)
                eq = run_trace_experiment(requests, "Equal", tr, EPOCH, N_STARTS)
                so = run_trace_experiment(requests, "Stall-opt", tr, EPOCH, N_STARTS)
                eq_r = eq / eq_base if eq_base else 0.0
                so_r = so / so_base if so_base else 0.0
                ratio = so_r / eq_r if eq_r else 0.0
                surface[d][g] = (ratio, so_r, eq_r)
                mark = "*" if ratio > 1.0 else " "
                row += f"  {ratio:>5.2f}{mark}"
            p(row)
        p("")
        p("  ('*' marks cells where the Stall-opt/Equal inversion holds)")
        p("")

        # Locate the boundary
        holds = [(d, g) for d in low_durs for g in gaps
                 if surface[d][g][0] > 1.0]
        gone = [(d, g) for d in low_durs for g in gaps
                if surface[d][g][0] <= 1.0]
        p(f"  Inversion HOLDS in {len(holds)}/{len(low_durs)*len(gaps)} cells.")
        if gone:
            worst_gone = min(gone, key=lambda k: surface[k[0]][k[1]][0])
            p(f"  Inversion strongest at long low-episodes / short gaps "
              f"(sustained degradation).")
            p(f"  Inversion weakest / gone at: dur={worst_gone[0]}s gap={worst_gone[1]}s "
              f"(ratio {surface[worst_gone[0]][worst_gone[1]][0]:.2f}) — "
              f"brief, rare dips.")
        else:
            p("  Inversion holds across the ENTIRE swept surface.")
        p("")


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/raw") / f"corrections_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = []
    out.append("Corrections 1 & 2 to time-varying bandwidth findings")
    out.append(f"Timestamp: {ts}")
    out.append(f"Compute window: PAPER's Table A8 (t_total/L).  epoch={EPOCH}s, "
               f"{N_STARTS} starts.")
    out.append("")
    correction_1(out)
    correction_2(out)

    text = "\n".join(out) + "\n"
    print(text)
    path = out_dir / "results.txt"
    path.write_text(text)
    print(f"Output saved to {path}")


if __name__ == "__main__":
    main()
