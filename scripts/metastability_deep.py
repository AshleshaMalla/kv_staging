#!/usr/bin/env python3
"""Deep metastability analysis: predictability + fluid-queue baseline.

Task 1: 30 seeds at 70% with episode. Regress drain time on pre-episode state.
Task 2: Fluid-queue baseline for each seed. Is drain slower than backlog recovery?
"""

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import queue_compute
import queue_feedback
from queue_feedback import (
    simulate, CAPACITY, mean_service, _mean, _meanN_window, _drain_time,
    _single_episode_trace, ActiveReq, WORKLOAD_SPEC,
)
from queue_compute import t_prefill
from stallopt import gbps_to_Bps, make_constant_trace, Bps_to_gbps

queue_compute.Q_SLOPE = 1.0
queue_feedback.Q_SLOPE = 1.0

CAP_GBPS = 80
HORIZON = 1200.0
EPISODE_DUR = 60.0
BIMODAL_DEPTH = 2.3
SEEDS = list(range(1, 31))


def run_episode_detailed(frac, seed):
    """Run episode injection and return detailed state."""
    rate = frac * CAPACITY
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    ep_trace = _single_episode_trace(cap_Bps, t_start=300.0, dur=EPISODE_DUR,
                                     depth=BIMODAL_DEPTH, horizon=HORIZON)
    _, nlog, counts = simulate(rate, ep_trace, "Stall-opt", HORIZON, seed,
                                queued=True, log_N=True)

    pre_n = _meanN_window(nlog, 200, 300)
    during_n = _meanN_window(nlog, 300, 360)
    post_n = _meanN_window(nlog, 360, 420)
    peak_n = max((n for t, n, _ in nlog if 300 <= t < 420), default=0)

    n_at_ep_start = next((n for t, n, _ in nlog if t >= 299.5), 0)
    n_at_ep_end = next((n for t, n, _ in nlog if t >= 359.5), 0)

    drain = _drain_time(nlog, 360.0, pre_n)

    n_at_300 = [n for t, n, _ in nlog if 299 <= t <= 301]
    n_at_360 = [n for t, n, _ in nlog if 359 <= t <= 361]
    n_at_start = _mean(n_at_300) if n_at_300 else 0
    n_at_end = _mean(n_at_360) if n_at_360 else 0

    delta_n = n_at_end - n_at_start

    return {
        "seed": seed,
        "preN": pre_n,
        "duringN": during_n,
        "postN": post_n,
        "peakN": peak_n,
        "n_at_start": n_at_start,
        "n_at_end": n_at_end,
        "delta_n": delta_n,
        "drain_s": drain,
        "persistence": drain / EPISODE_DUR if drain else None,
        "censored": counts["censored_frac"],
    }


def fluid_baseline_drain(rate, pre_n, n_at_end, episode_dur):
    """Fluid-queue prediction: how long SHOULD the drain take?

    During the episode, arrival rate lambda stays constant but effective
    service rate mu_degraded = mu / degradation_factor. The queue builds.
    After recovery, service rate returns to mu. Drain time = backlog / (mu - lambda).

    But we don't model service-rate degradation directly — the episode
    reduces BANDWIDTH, not compute. The compute queue builds because
    transfers take longer, extending occupancy.

    Simpler model: treat the queue at episode end (n_at_end) as backlog.
    Each request takes mean_service to complete. The server drains at rate
    mu = 1/mean_service when not receiving new work, but new work arrives
    at rate lambda. Net drain rate = mu - lambda.

    t_drain_fluid = (n_at_end - pre_n) / (mu - lambda)
    where mu = 1/mean_service (effective, accounting for serialization)
    and lambda = rate.

    At slope=1.0 (pure serialization), effective throughput with N in
    system is 1/mean_service regardless of N (compute-bound, no batching).
    So mu = 1/mean_service = CAPACITY.
    """
    mu = CAPACITY
    if mu <= rate:
        return float('inf')
    excess = n_at_end - pre_n
    if excess <= 0:
        return 0.0
    return excess / (mu - rate)


def main():
    print("=" * 80)
    print("METASTABILITY DEEP ANALYSIS (slope=1.0, 30 seeds)")
    print("=" * 80)
    print(f"Q_SLOPE = {queue_feedback.Q_SLOPE}")
    print(f"Capacity = {CAPACITY:.3f} req/s, mean_service = {mean_service():.3f}s")
    print(f"Load = 70% = {0.70 * CAPACITY:.3f} req/s")
    print(f"Episode: 60s degradation (depth {BIMODAL_DEPTH}x) at t=300s")
    print(f"Seeds: {SEEDS[0]}-{SEEDS[-1]} ({len(SEEDS)} total)")
    print()

    # ── Run all seeds ──
    results = []
    print("TASK 1: Episode injection, 30 seeds")
    print("-" * 80)
    print(f"  {'Seed':>4s}  {'preN':>5s}  {'n@300':>5s}  {'n@360':>5s}  {'dN':>5s}  {'peakN':>5s}  "
          f"{'drain':>6s}  {'persist':>8s}")

    for seed in SEEDS:
        r = run_episode_detailed(0.70, seed)
        results.append(r)
        d_s = f"{r['drain_s']:.0f}" if r['drain_s'] else "N/A"
        p_s = f"{r['persistence']:.2f}x" if r['persistence'] else "N/A"
        print(f"  {seed:>4d}  {r['preN']:>5.1f}  {r['n_at_start']:>5.1f}  {r['n_at_end']:>5.1f}  "
              f"{r['delta_n']:>+5.1f}  {r['peakN']:>5d}  {d_s:>6s}  {p_s:>8s}")

    drains = [r['drain_s'] for r in results if r['drain_s'] is not None]
    persists = [r['persistence'] for r in results if r['persistence'] is not None]
    pre_ns = [r['preN'] for r in results]
    delta_ns = [r['delta_n'] for r in results]

    print(f"\n  Drain time: mean={statistics.mean(drains):.0f}s, "
          f"median={statistics.median(drains):.0f}s, "
          f"std={statistics.stdev(drains):.0f}s")
    print(f"  Persistence: mean={statistics.mean(persists):.2f}x, "
          f"median={statistics.median(persists):.2f}x, "
          f"std={statistics.stdev(persists):.2f}x")
    print(f"  Range: [{min(drains):.0f}s, {max(drains):.0f}s] = "
          f"[{min(persists):.2f}x, {max(persists):.2f}x]")
    n_above_1 = sum(1 for p in persists if p > 1.0)
    print(f"  Seeds with persistence > 1.0: {n_above_1}/{len(persists)}")
    print(f"  Seeds with persistence < 1.0: {len(persists) - n_above_1}/{len(persists)}")

    # ── Regression: drain on preN ──
    print()
    print("REGRESSION: drain_time vs preN")
    print("-" * 80)

    valid = [(r['preN'], r['drain_s']) for r in results if r['drain_s'] is not None]
    xs = [v[0] for v in valid]
    ys = [v[1] for v in valid]

    n = len(xs)
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    ss_xx = sum((x - mean_x)**2 for x in xs)
    ss_yy = sum((y - mean_y)**2 for y in ys)
    ss_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))

    r_val = ss_xy / math.sqrt(ss_xx * ss_yy) if ss_xx > 0 and ss_yy > 0 else 0
    slope = ss_xy / ss_xx if ss_xx > 0 else 0
    intercept = mean_y - slope * mean_x

    print(f"  n = {n}")
    print(f"  Pearson r = {r_val:.3f}")
    print(f"  drain_s = {slope:.1f} * preN + {intercept:.1f}")
    print(f"  Direction: {'NEGATIVE (higher preN → shorter drain)' if slope < 0 else 'POSITIVE (higher preN → longer drain)'}")

    if abs(r_val) > 0.5:
        print(f"  → MODERATE-STRONG correlation: drain IS predictable from pre-episode state")
    elif abs(r_val) > 0.3:
        print(f"  → WEAK correlation")
    else:
        print(f"  → NO meaningful correlation: drain is not predictable from preN")

    # Also check delta_n vs drain
    valid_dn = [(r['delta_n'], r['drain_s']) for r in results if r['drain_s'] is not None]
    xs_dn = [v[0] for v in valid_dn]
    mean_dn = sum(xs_dn) / n
    ss_dn = sum((x - mean_dn)**2 for x in xs_dn)
    ss_dn_y = sum((x - mean_dn) * (y - mean_y) for x, y in zip(xs_dn, ys))
    r_dn = ss_dn_y / math.sqrt(ss_dn * ss_yy) if ss_dn > 0 and ss_yy > 0 else 0
    print(f"\n  delta_N (queue built during episode) vs drain:")
    print(f"  Pearson r = {r_dn:.3f}")
    print(f"  Mean delta_N = {statistics.mean(xs_dn):.1f}, std = {statistics.stdev(xs_dn):.1f}")
    if abs(r_dn) > 0.5:
        print(f"  → Delta_N IS predictive of drain time")
    else:
        print(f"  → Delta_N is NOT strongly predictive")

    # ── TASK 2: Fluid-queue baseline ──
    print()
    print("TASK 2: Fluid-queue baseline comparison")
    print("-" * 80)
    rate = 0.70 * CAPACITY
    mu = CAPACITY
    print(f"  lambda = {rate:.3f} req/s, mu = {mu:.3f} req/s")
    print(f"  Spare capacity = mu - lambda = {mu - rate:.3f} req/s")
    print()
    print(f"  {'Seed':>4s}  {'preN':>5s}  {'n@360':>5s}  {'dN':>5s}  {'sim_drain':>10s}  "
          f"{'fluid_drain':>12s}  {'ratio':>7s}")

    ratios = []
    for r in results:
        if r['drain_s'] is None:
            continue
        fluid = fluid_baseline_drain(rate, r['preN'], r['n_at_end'], EPISODE_DUR)
        ratio = r['drain_s'] / fluid if fluid > 0 else float('inf')
        ratios.append(ratio)
        f_s = f"{fluid:.0f}" if fluid < 1e6 else "inf"
        print(f"  {r['seed']:>4d}  {r['preN']:>5.1f}  {r['n_at_end']:>5.1f}  "
              f"{r['delta_n']:>+5.1f}  {r['drain_s']:>9.0f}s  {f_s:>11s}s  {ratio:>7.2f}")

    finite_ratios = [r for r in ratios if r < 1e6]
    if finite_ratios:
        print(f"\n  Simulated / fluid baseline ratio:")
        print(f"    mean = {statistics.mean(finite_ratios):.2f}")
        print(f"    median = {statistics.median(finite_ratios):.2f}")
        print(f"    std = {statistics.stdev(finite_ratios):.2f}")
        print(f"    range = [{min(finite_ratios):.2f}, {max(finite_ratios):.2f}]")
        n_above = sum(1 for r in finite_ratios if r > 1.0)
        print(f"    Seeds with ratio > 1.0 (drain SLOWER than fluid): {n_above}/{len(finite_ratios)}")
        print(f"    Seeds with ratio < 1.0 (drain FASTER than fluid): {len(finite_ratios) - n_above}/{len(finite_ratios)}")
        print()
        if statistics.mean(finite_ratios) > 1.2 and n_above > len(finite_ratios) * 0.6:
            print("  → METASTABILITY: drain is systematically slower than fluid prediction.")
            print("    The queue does not merely recover backlog — something slows recovery.")
        elif statistics.mean(finite_ratios) < 0.8:
            print("  → NO METASTABILITY: drain is faster than fluid prediction.")
            print("    This is ordinary backlog recovery.")
        else:
            print("  → AMBIGUOUS: simulated drain is close to fluid baseline.")
            print("    Cannot distinguish metastability from ordinary backlog recovery")
            print("    at this sample size and load level.")


if __name__ == "__main__":
    main()
