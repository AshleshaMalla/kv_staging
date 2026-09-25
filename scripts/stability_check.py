#!/usr/bin/env python3
"""No-episode stability check: is N stationary at each load level?

Runs the feedback simulation with constant bandwidth (no degradation episode)
at slope=1.0, and reports whether N is stationary or trending upward.
Also sweeps arrival rate to find the effective saturation point.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import queue_compute
import queue_feedback
from queue_feedback import simulate, CAPACITY, mean_service, _mean, _meanN_window
from stallopt import gbps_to_Bps, make_constant_trace, Bps_to_gbps

queue_compute.Q_SLOPE = 1.0
queue_feedback.Q_SLOPE = 1.0

CAP_GBPS = 80
HORIZON = 900.0
SEED = 7


def check_stationarity(frac, label=""):
    """Run at given load with constant BW, no episode. Report N trajectory."""
    rate = frac * CAPACITY
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    const = make_constant_trace(cap_Bps, duration=100000.0)

    added, nlog, counts = simulate(rate, const, "Stall-opt", HORIZON, SEED,
                                    queued=True, log_N=True)

    ns = [n for _, n, _ in nlog]
    mean_n = _mean(ns)
    max_n = max(ns)

    # Split into windows to check for trend
    w1 = _meanN_window(nlog, 100, 300)
    w2 = _meanN_window(nlog, 300, 500)
    w3 = _meanN_window(nlog, 500, 700)
    w4 = _meanN_window(nlog, 700, 880)

    trending = w4 > w1 * 1.3
    stationary = abs(w4 - w1) / max(w1, 0.1) < 0.15

    if stationary:
        verdict = "STATIONARY"
    elif trending:
        verdict = "TRENDING UP — intrinsic instability"
    else:
        verdict = "AMBIGUOUS"

    return {
        "frac": frac,
        "rate": rate,
        "mean_N": mean_n,
        "max_N": max_n,
        "w1": w1, "w2": w2, "w3": w3, "w4": w4,
        "trending": trending,
        "stationary": stationary,
        "verdict": verdict,
        "censored_frac": counts["censored_frac"],
    }


def find_saturation():
    """Sweep arrival rate to find where mean N stops draining (effective capacity)."""
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    const = make_constant_trace(cap_Bps, duration=100000.0)

    print("  Arrival rate sweep (constant BW, no episode, queued, Stall-opt):")
    print(f"    {'Load':>6s}  {'Rate':>7s}  {'meanN':>6s}  {'maxN':>5s}  {'w1':>5s}  {'w4':>5s}  {'Trend':>8s}  {'Cens':>5s}  {'Verdict':>20s}")

    last_stable = None
    first_unstable = None

    for pct in range(50, 100, 2):
        frac = pct / 100.0
        rate = frac * CAPACITY
        added, nlog, counts = simulate(rate, const, "Stall-opt", HORIZON, SEED,
                                        queued=True, log_N=True)
        ns = [n for _, n, _ in nlog]
        mean_n = _mean(ns)
        max_n = max(ns)
        w1 = _meanN_window(nlog, 100, 300)
        w4 = _meanN_window(nlog, 700, 880)

        trending = w4 > w1 * 1.3
        stationary = abs(w4 - w1) / max(w1, 0.1) < 0.15

        if stationary:
            verdict = "stable"
            last_stable = frac
        elif trending:
            verdict = "UNSTABLE"
            if first_unstable is None:
                first_unstable = frac
        else:
            verdict = "marginal"

        print(f"    {frac:>5.0%}  {rate:>7.3f}  {mean_n:>6.1f}  {max_n:>5d}  {w1:>5.1f}  {w4:>5.1f}  {'UP' if trending else 'flat':>8s}  {counts['censored_frac']*100:>4.1f}%  {verdict:>20s}")

    return last_stable, first_unstable


def main():
    print("=" * 80)
    print("NO-EPISODE STABILITY CHECK (slope=1.0)")
    print("=" * 80)
    print(f"Q_SLOPE = {queue_feedback.Q_SLOPE}")
    print(f"Capacity = {CAPACITY:.3f} req/s (mean service {mean_service():.3f}s)")
    print()

    # Step 1 & 2: Check 70% and 85%
    print("STEP 1-2: No-episode N trajectory at 70% and 85% load")
    print("-" * 80)
    for frac in [0.70, 0.85]:
        r = check_stationarity(frac)
        print(f"  Load {frac:.0%} ({r['rate']:.3f} req/s):")
        print(f"    mean N = {r['mean_N']:.1f}, max N = {r['max_N']}")
        print(f"    Windows: [100-300s]={r['w1']:.1f}  [300-500s]={r['w2']:.1f}  "
              f"[500-700s]={r['w3']:.1f}  [700-880s]={r['w4']:.1f}")
        print(f"    Censoring: {r['censored_frac']*100:.1f}%")
        print(f"    → {r['verdict']}")
        print()

    # Step 3: Find effective saturation
    print("STEP 3: Arrival rate sweep — find effective saturation")
    print("-" * 80)
    last_stable, first_unstable = find_saturation()
    print()
    if last_stable and first_unstable:
        print(f"  Last stable load: {last_stable:.0%}")
        print(f"  First unstable load: {first_unstable:.0%}")
        mid = (last_stable + first_unstable) / 2
        print(f"  Effective saturation point: ~{mid:.0%} of {CAPACITY:.3f} req/s "
              f"= ~{mid * CAPACITY:.3f} req/s")
    elif last_stable:
        print(f"  Last stable load: {last_stable:.0%}")
        print(f"  No instability found in sweep range.")
    else:
        print(f"  No stable load found (all trending).")


if __name__ == "__main__":
    main()
