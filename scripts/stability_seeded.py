#!/usr/bin/env python3
"""Seeded replication of saturation boundary and 70% metastability.

1. Arrival-rate sweep at 65-85% with 5 seeds per load — resolve the saturation
   boundary as a range, not a point estimate from one Poisson realization.
2. 70% metastability: 5 seeds for episode and no-episode runs. Report drain
   time distribution and confirm the control is stationary across seeds.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import queue_compute
import queue_feedback
from queue_feedback import (
    simulate, CAPACITY, mean_service, _mean, _meanN_window, _drain_time,
    _single_episode_trace,
)
from stallopt import gbps_to_Bps, make_constant_trace

queue_compute.Q_SLOPE = 1.0
queue_feedback.Q_SLOPE = 1.0

CAP_GBPS = 80
HORIZON = 900.0
BIMODAL_DEPTH = 2.3
EPISODE_DUR = 60.0
SEEDS = [7, 13, 42, 97, 131]


def check_stability(frac, seed):
    rate = frac * CAPACITY
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    const = make_constant_trace(cap_Bps, duration=100000.0)
    added, nlog, counts = simulate(rate, const, "Stall-opt", HORIZON, seed,
                                    queued=True, log_N=True)
    ns = [n for _, n, _ in nlog]
    w1 = _meanN_window(nlog, 100, 300)
    w4 = _meanN_window(nlog, 700, 880)
    trending = w4 > w1 * 1.3
    stationary = abs(w4 - w1) / max(w1, 0.1) < 0.15
    return {
        "mean_N": _mean(ns), "max_N": max(ns),
        "w1": w1, "w4": w4,
        "trending": trending, "stationary": stationary,
        "censored_frac": counts["censored_frac"],
    }


def check_metastability(frac, seed):
    rate = frac * CAPACITY
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    ep_trace = _single_episode_trace(cap_Bps, t_start=300.0, dur=EPISODE_DUR,
                                     depth=BIMODAL_DEPTH, horizon=HORIZON)
    _, nlog, _ = simulate(rate, ep_trace, "Stall-opt", HORIZON, seed,
                          queued=True, log_N=True)
    before = _meanN_window(nlog, 200, 300)
    during = _meanN_window(nlog, 300, 360)
    peak = max((n for t, n, _ in nlog if 300 <= t < 420), default=0)
    drain = _drain_time(nlog, 360.0, before)
    return {
        "preN": before, "duringN": during, "peakN": peak,
        "drain_s": drain,
        "persistence": drain / EPISODE_DUR if drain else None,
    }


def main():
    print("=" * 80)
    print("SEEDED REPLICATION (slope=1.0, 5 seeds)")
    print("=" * 80)
    print(f"Q_SLOPE = {queue_feedback.Q_SLOPE}")
    print(f"Capacity = {CAPACITY:.3f} req/s")
    print(f"Seeds: {SEEDS}")
    print()

    # ── Part 1: Saturation boundary sweep ──
    print("PART 1: Saturation boundary (65-85%, 5 seeds, no episode)")
    print("-" * 80)
    print(f"  {'Load':>6s}  {'Seed':>6s}  {'meanN':>6s}  {'maxN':>5s}  {'w1':>5s}  {'w4':>5s}  {'Cens':>5s}  {'Verdict':>12s}")

    boundary_data = {}
    for pct in range(65, 86):
        frac = pct / 100.0
        results = []
        for seed in SEEDS:
            r = check_stability(frac, seed)
            results.append(r)
            v = "stable" if r["stationary"] else ("UNSTABLE" if r["trending"] else "marginal")
            print(f"  {frac:>5.0%}  {seed:>6d}  {r['mean_N']:>6.1f}  {r['max_N']:>5d}  {r['w1']:>5.1f}  {r['w4']:>5.1f}  {r['censored_frac']*100:>4.1f}%  {v:>12s}")

        n_stable = sum(1 for r in results if r["stationary"])
        n_unstable = sum(1 for r in results if r["trending"])
        n_marginal = len(results) - n_stable - n_unstable
        boundary_data[frac] = {
            "n_stable": n_stable, "n_unstable": n_unstable, "n_marginal": n_marginal,
            "mean_meanN": _mean([r["mean_N"] for r in results]),
            "max_maxN": max(r["max_N"] for r in results),
        }
        print(f"  {frac:>5.0%}  {'ALL':>6s}  → {n_stable}/5 stable, {n_unstable}/5 unstable, {n_marginal}/5 marginal")
        print()

    print()
    print("  SATURATION SUMMARY:")
    print(f"  {'Load':>6s}  {'Stable':>7s}  {'Unstable':>9s}  {'Marginal':>9s}  {'meanN':>6s}  {'Classification':>20s}")
    last_stable = None
    first_unstable = None
    for pct in range(65, 86):
        frac = pct / 100.0
        d = boundary_data[frac]
        if d["n_stable"] >= 3:
            classification = "STABLE"
            last_stable = frac
        elif d["n_unstable"] >= 3:
            classification = "UNSTABLE"
            if first_unstable is None:
                first_unstable = frac
        else:
            classification = "transitional"
        print(f"  {frac:>5.0%}  {d['n_stable']:>5d}/5  {d['n_unstable']:>7d}/5  {d['n_marginal']:>7d}/5  {d['mean_meanN']:>6.1f}  {classification:>20s}")

    print()
    if last_stable and first_unstable:
        print(f"  Last majority-stable load: {last_stable:.0%}")
        print(f"  First majority-unstable load: {first_unstable:.0%}")
        print(f"  Saturation range: {last_stable:.0%} - {first_unstable:.0%} of {CAPACITY:.3f} req/s")
        print(f"    = {last_stable*CAPACITY:.3f} - {first_unstable*CAPACITY:.3f} req/s")
    print()

    # ── Part 2: 70% metastability replicated ──
    print("PART 2: 70% metastability (5 seeds, episode + no-episode control)")
    print("-" * 80)

    print("  No-episode control (constant BW):")
    print(f"  {'Seed':>6s}  {'meanN':>6s}  {'w1':>5s}  {'w4':>5s}  {'Verdict':>12s}")
    controls = []
    for seed in SEEDS:
        r = check_stability(0.70, seed)
        controls.append(r)
        v = "STATIONARY" if r["stationary"] else ("TRENDING" if r["trending"] else "marginal")
        print(f"  {seed:>6d}  {r['mean_N']:>6.1f}  {r['w1']:>5.1f}  {r['w4']:>5.1f}  {v:>12s}")
    n_stat = sum(1 for r in controls if r["stationary"])
    print(f"  → {n_stat}/5 stationary")
    print()

    print("  Episode injection (single 60s degradation at t=300):")
    print(f"  {'Seed':>6s}  {'preN':>5s}  {'durN':>5s}  {'peakN':>6s}  {'drain(s)':>9s}  {'persist':>8s}")
    episodes = []
    for seed in SEEDS:
        r = check_metastability(0.70, seed)
        episodes.append(r)
        d_s = f"{r['drain_s']:.0f}" if r["drain_s"] else "N/A"
        p_s = f"{r['persistence']:.2f}x" if r["persistence"] else "N/A"
        print(f"  {seed:>6d}  {r['preN']:>5.1f}  {r['duringN']:>5.1f}  {r['peakN']:>6d}  {d_s:>9s}  {p_s:>8s}")

    drains = [r["drain_s"] for r in episodes if r["drain_s"] is not None]
    peaks = [r["peakN"] for r in episodes]
    if drains:
        import statistics
        mean_drain = statistics.mean(drains)
        if len(drains) >= 2:
            std_drain = statistics.stdev(drains)
        else:
            std_drain = 0
        min_drain = min(drains)
        max_drain = max(drains)
        print(f"\n  Drain time: mean={mean_drain:.0f}s, std={std_drain:.0f}s, "
              f"range=[{min_drain:.0f}, {max_drain:.0f}]s")
        print(f"  Persistence: mean={mean_drain/EPISODE_DUR:.2f}x, "
              f"range=[{min_drain/EPISODE_DUR:.2f}x, {max_drain/EPISODE_DUR:.2f}x]")
        print(f"  Peak N: range=[{min(peaks)}, {max(peaks)}], all {'within' if max(peaks) <= 64 else 'EXCEEDS'} measured N<=64")
        print(f"  Drained: {len(drains)}/{len(episodes)} seeds")
    print()


if __name__ == "__main__":
    main()
