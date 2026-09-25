#!/usr/bin/env python3
"""Test the contingent-sign result: seeding, convexity baseline, conditional overlap.

Task 1: Seed the feedback four-cell interaction at 30/50/70% with 30 seeds.
Task 2: Convexity baseline — no layer pipeline, no overlap, just queueing delay.
Task 3: Conditional-overlap sweep for the fixed-N regime.
"""

import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

import queue_compute
import queue_feedback
from queue_feedback import (
    simulate, CAPACITY, mean_service, _mean,
    ActiveReq, WORKLOAD_SPEC, EPOCH as FB_EPOCH,
    poisson_arrivals,
)
from queue_compute import (
    A_VLLM, B_VLLM, Q_SLOPE, t_prefill, WORKLOAD_SPEC as QC_WORKLOAD_SPEC,
    QReq, build_workload, run_cell, queue_factor,
    N_CONCURRENT, WORKLOADS, EPOCH as QC_EPOCH, N_STARTS,
    BIMODAL_DEPTH, BIMODAL_EPISODE,
)
from stallopt import (
    L, BYTES_PER_TOKEN_PER_LAYER, POLICIES, POLICY_ORDER,
    Bps_to_gbps, gbps_to_Bps, _transfer_time,
    make_constant_trace, make_bimodal_trace,
)

queue_compute.Q_SLOPE = 1.0
queue_feedback.Q_SLOPE = 1.0

CAP_GBPS = 80
HORIZON = 900.0
SEEDS = list(range(1, 31))


# ═══════════════════════════════════════════════════════════════════════════
# TASK 1 — Seed the feedback result
# ═══════════════════════════════════════════════════════════════════════════

def run_feedback_cell(frac, seed, trace, queued):
    rate = frac * CAPACITY
    added, _, counts = simulate(rate, trace, "Stall-opt", HORIZON, seed,
                                 queued=queued)
    return _mean(added), counts["censored_frac"]


def task1_seed_feedback():
    print("=" * 80)
    print("TASK 1: Seed the feedback four-cell interaction (30 seeds)")
    print("=" * 80)
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    const = make_constant_trace(cap_Bps, duration=100000.0)
    bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                 episode_sec=BIMODAL_EPISODE, n_cycles=200)

    print(f"  Experiment B (fixed N=4): DETERMINISTIC — no seeding needed.")
    print(f"  Uses 200 start-time offsets over a deterministic bimodal trace.")
    print()

    for frac in [0.30, 0.50, 0.70]:
        print(f"  Load {frac:.0%} ({frac * CAPACITY:.3f} req/s):")
        interactions = []
        cells_by_seed = []
        for seed in SEEDS:
            c1, _ = run_feedback_cell(frac, seed, const, queued=False)
            c2, _ = run_feedback_cell(frac, seed, bimodal, queued=False)
            c3, _ = run_feedback_cell(frac, seed, const, queued=True)
            c4, cens = run_feedback_cell(frac, seed, bimodal, queued=True)
            inter = c4 - (c2 + c3 - c1)
            interactions.append(inter)
            cells_by_seed.append((seed, c1, c2, c3, c4, inter))

        mean_i = statistics.mean(interactions)
        median_i = statistics.median(interactions)
        std_i = statistics.stdev(interactions)
        n_pos = sum(1 for i in interactions if i > 0)
        n_neg = sum(1 for i in interactions if i < 0)

        # Bootstrap 95% CI
        import random
        rng = random.Random(999)
        n_boot = 10000
        boot_means = []
        for _ in range(n_boot):
            sample = [rng.choice(interactions) for _ in range(len(interactions))]
            boot_means.append(statistics.mean(sample))
        boot_means.sort()
        ci_lo = boot_means[int(n_boot * 0.025)]
        ci_hi = boot_means[int(n_boot * 0.975)]

        print(f"    Interaction (Stall-opt): mean={mean_i:+.1f}, median={median_i:+.1f}, "
              f"std={std_i:.1f}")
        print(f"    Bootstrap 95% CI: [{ci_lo:+.1f}, {ci_hi:+.1f}]")
        print(f"    Sign: {n_pos}/30 positive (super-additive), {n_neg}/30 negative")
        ci_excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        sign_str = "SUPER-ADDITIVE" if ci_lo > 0 else ("SUB-ADDITIVE" if ci_hi < 0 else "STRADDLES ZERO")
        print(f"    CI excludes zero: {'YES' if ci_excludes_zero else 'NO'} → {sign_str}")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# TASK 2 — Convexity baseline (no layer pipeline, no overlap)
# ═══════════════════════════════════════════════════════════════════════════

def simulate_convexity_baseline(rate, trace, horizon, seed, queued=True):
    """Simple queue: each request has a service time, degraded BW extends it.
    No layer pipeline, no overlap, no bandwidth allocation policy.
    Just: service_time_effective = t_compute + t_fetch.
    t_fetch = total_bytes / BW(t).
    t_compute = t_prefill_iso * cfactor(N).
    Request finishes when both are done (same as queue_feedback).

    The KEY difference from our simulator: there is no per-layer overlap model
    and no bandwidth allocation policy. Each request just sees the raw BW.
    This isolates the queueing-convexity effect from the KV-serving pipeline.
    """
    COOLDOWN = 60.0
    WARMUP = 60.0
    arrivals = poisson_arrivals(rate, horizon, seed)
    ai = 0
    active = []
    completed = []
    completed_arrivals = set()
    dt = FB_EPOCH

    t = 0.0
    n_steps = int(math.ceil(horizon / dt))
    for _ in range(n_steps):
        while ai < len(arrivals) and arrivals[ai][0] <= t:
            ctx, hit = arrivals[ai][1]
            active.append(_BaselineReq(ctx, hit, arrivals[ai][0]))
            ai += 1

        N = len(active)
        if N == 0:
            t += dt
            continue

        B = trace.bw_at(t)
        cfactor = (1.0 + Q_SLOPE * (N - 1)) if queued else 1.0

        finished_idx = []
        for i, req in enumerate(active):
            if req.bytes_remaining > 0 and B > 0:
                # Each request gets 1/N of the bandwidth (simple fair share)
                req.bytes_remaining -= (B / N) * dt
            req.compute_remaining -= dt / cfactor
            if req.bytes_remaining <= 1e-6 and req.compute_remaining <= 1e-9:
                req.ttft = (t + dt) - req.arrival
                added = (req.ttft - req.t_compute_iso) * 1000.0
                if req.arrival >= WARMUP:
                    completed.append(added)
                    completed_arrivals.add(round(req.arrival, 6))
                finished_idx.append(i)
        for i in reversed(finished_idx):
            active.pop(i)
        t += dt

    eligible = [a for a in arrivals if WARMUP <= a[0] <= horizon - COOLDOWN]
    n_eligible = len(eligible)
    n_done = sum(1 for a in eligible if round(a[0], 6) in completed_arrivals)
    cens = (n_eligible - n_done) / n_eligible if n_eligible else 0.0
    return _mean(completed) if completed else 0.0, cens


class _BaselineReq:
    """Minimal request for the convexity baseline."""
    __slots__ = ("context_tokens", "hit_ratio", "arrival",
                 "bytes_remaining", "compute_remaining", "t_compute_iso", "ttft")

    def __init__(self, context_tokens, hit_ratio, arrival):
        self.context_tokens = context_tokens
        self.hit_ratio = hit_ratio
        uncached = context_tokens - round(context_tokens * hit_ratio)
        self.t_compute_iso = t_prefill(uncached)
        self.arrival = arrival
        cached = round(context_tokens * hit_ratio)
        self.bytes_remaining = cached * BYTES_PER_TOKEN_PER_LAYER * L
        self.compute_remaining = self.t_compute_iso
        self.ttft = None


def task2_convexity_baseline():
    print("=" * 80)
    print("TASK 2: Convexity baseline (no layer pipeline, no overlap)")
    print("=" * 80)
    print("  Baseline: simple queue with fair-share BW, same compute model,")
    print("  no per-layer pipeline, no bandwidth allocation policy.")
    print()

    cap_Bps = gbps_to_Bps(CAP_GBPS)
    const = make_constant_trace(cap_Bps, duration=100000.0)
    bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                 episode_sec=BIMODAL_EPISODE, n_cycles=200)

    for frac in [0.30, 0.50, 0.70]:
        rate = frac * CAPACITY
        print(f"  Load {frac:.0%}:")

        sim_interactions = []
        base_interactions = []
        for seed in SEEDS:
            # Our simulator
            s1, _, _ = simulate(rate, const, "Stall-opt", HORIZON, seed, queued=False)
            s2, _, _ = simulate(rate, bimodal, "Stall-opt", HORIZON, seed, queued=False)
            s3, _, _ = simulate(rate, const, "Stall-opt", HORIZON, seed, queued=True)
            s4, _, _ = simulate(rate, bimodal, "Stall-opt", HORIZON, seed, queued=True)
            sim_inter = _mean(s4) - (_mean(s2) + _mean(s3) - _mean(s1))
            sim_interactions.append(sim_inter)

            # Convexity baseline
            b1, _ = simulate_convexity_baseline(rate, const, HORIZON, seed, queued=False)
            b2, _ = simulate_convexity_baseline(rate, bimodal, HORIZON, seed, queued=False)
            b3, _ = simulate_convexity_baseline(rate, const, HORIZON, seed, queued=True)
            b4, _ = simulate_convexity_baseline(rate, bimodal, HORIZON, seed, queued=True)
            base_inter = b4 - (b2 + b3 - b1)
            base_interactions.append(base_inter)

        sim_mean = statistics.mean(sim_interactions)
        sim_std = statistics.stdev(sim_interactions)
        base_mean = statistics.mean(base_interactions)
        base_std = statistics.stdev(base_interactions)

        # Bootstrap CIs
        import random
        rng = random.Random(999)
        n_boot = 10000

        def bootstrap_ci(vals):
            boots = []
            for _ in range(n_boot):
                s = [rng.choice(vals) for _ in range(len(vals))]
                boots.append(statistics.mean(s))
            boots.sort()
            return boots[int(n_boot * 0.025)], boots[int(n_boot * 0.975)]

        sim_ci = bootstrap_ci(sim_interactions)
        base_ci = bootstrap_ci(base_interactions)

        # Difference
        diffs = [s - b for s, b in zip(sim_interactions, base_interactions)]
        diff_mean = statistics.mean(diffs)
        diff_ci = bootstrap_ci(diffs)

        print(f"    Simulator:  mean={sim_mean:+.1f} ms, std={sim_std:.1f}, "
              f"CI=[{sim_ci[0]:+.1f}, {sim_ci[1]:+.1f}]")
        print(f"    Baseline:   mean={base_mean:+.1f} ms, std={base_std:.1f}, "
              f"CI=[{base_ci[0]:+.1f}, {base_ci[1]:+.1f}]")
        print(f"    Sim-Base:   mean={diff_mean:+.1f} ms, "
              f"CI=[{diff_ci[0]:+.1f}, {diff_ci[1]:+.1f}]")

        if base_ci[0] > 0:
            print(f"    → Baseline is SUPER-ADDITIVE (CI excludes zero)")
        elif base_ci[1] < 0:
            print(f"    → Baseline is SUB-ADDITIVE")
        else:
            print(f"    → Baseline straddles zero")

        if diff_ci[0] > 0:
            print(f"    → Simulator EXCEEDS baseline: KV-specific contribution")
        elif diff_ci[1] < 0:
            print(f"    → Simulator BELOW baseline: overlap absorbs some convexity")
        else:
            print(f"    → Sim and baseline not significantly different")
        print()


# ═══════════════════════════════════════════════════════════════════════════
# TASK 3 — Conditional-overlap sweep for fixed-N
# ═══════════════════════════════════════════════════════════════════════════

def sim_one_conditional(alloc_reqs, idx, c_actual, policy_fn, trace, epoch, t0,
                        prefetch_layers):
    """Like sim_one but with conditional overlap.
    A layer's transfer can only overlap with compute if its read was issued
    `prefetch_layers` layers in advance. If prefetch_layers=0, no overlap
    (reads are synchronous). If prefetch_layers >= L, full overlap
    (equivalent to the unconditional model).

    Implementation: for layer l, the transfer for layer l+1 can overlap with
    layer l's compute only if (l+1) - issue_layer >= 0, where issue_layer is
    when the read was issued. With prefetch_layers=P, layer l+1's read is
    issued at the start of layer max(0, l+1-P).

    Simplified: the first P layers get no overlap (reads haven't been issued
    far enough in advance). From layer P onward, overlap is full.
    """
    s = alloc_reqs[idx].s
    t = t0

    # Layer 0: always just transfer (no overlap regardless)
    t += _transfer_time(s, idx, alloc_reqs, policy_fn, trace, epoch, t0, t)

    for layer in range(1, L):
        dt = _transfer_time(s, idx, alloc_reqs, policy_fn, trace, epoch, t0, t)
        if layer < prefetch_layers:
            # No overlap possible: read not issued far enough in advance
            # Transfer then compute, sequentially
            t += dt + c_actual
        else:
            # Overlap: max(transfer, compute)
            t += max(dt, c_actual)

    # Final compute (no next layer to overlap with)
    t += c_actual
    return t - t0


def run_cell_conditional(alloc_reqs, c_actual_list, policy_name, trace,
                         prefetch_layers, epoch=QC_EPOCH, n_starts=N_STARTS):
    """Like run_cell but with conditional overlap."""
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
            ttft_s = sim_one_conditional(alloc_reqs, idx, c_actual_list[idx],
                                         policy_fn, trace, epoch, t0,
                                         prefetch_layers)
            agg += (ttft_s - L * req.c_iso) * 1000.0
        total += agg
    return total / len(starts)


def task3_conditional_overlap():
    print("=" * 80)
    print("TASK 3: Conditional-overlap sweep (fixed N=4)")
    print("=" * 80)
    print("  Sweep prefetch depth from 0 (no overlap) to 32 (full unconditional).")
    print("  Report interaction for Stall-opt, Workload A.")
    print()

    qf = queue_factor(N_CONCURRENT)
    reqs = build_workload()
    c_iso_list = [r.c_iso for r in reqs]
    c_q_list = [r.c_iso * qf for r in reqs]

    cap_Bps = gbps_to_Bps(80)
    const = make_constant_trace(cap_Bps)
    bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                 episode_sec=BIMODAL_EPISODE)

    print(f"  {'Prefetch':>8s}  {'c1':>10s}  {'c2':>10s}  {'c3':>10s}  {'c4':>10s}  "
          f"{'Interaction':>12s}  {'Sign':>12s}")

    for prefetch in [0, 1, 2, 4, 8, 16, 24, 32]:
        c1 = run_cell_conditional(reqs, c_iso_list, "Stall-opt", const, prefetch)
        c2 = run_cell_conditional(reqs, c_iso_list, "Stall-opt", bimodal, prefetch)
        c3 = run_cell_conditional(reqs, c_q_list, "Stall-opt", const, prefetch)
        c4 = run_cell_conditional(reqs, c_q_list, "Stall-opt", bimodal, prefetch)
        inter = c4 - (c2 + c3 - c1)
        sign = "sub-additive" if inter < -1 else ("SUPER-additive" if inter > 1 else "~additive")
        label = "no overlap" if prefetch == 0 else ("full" if prefetch >= L else f"P={prefetch}")
        print(f"  {label:>8s}  {c1:>10.1f}  {c2:>10.1f}  {c3:>10.1f}  {c4:>10.1f}  "
              f"{inter:>+12.1f}  {sign:>12s}")

    print()
    print("  Interpretation:")
    print("  - P=0 (no overlap): if sub-additive, the mechanism is NOT overlap-dependent")
    print("  - P=32 (full): matches the unconditional model")
    print("  - If sub-additivity flips between P=0 and P=32: overlap is the mechanism")
    print("  - Realistic prefetch: P=1 or P=2 (one or two layers ahead)")


def main():
    print()
    task1_seed_feedback()
    print()
    task2_convexity_baseline()
    print()
    task3_conditional_overlap()

    print()
    print("=" * 80)
    print("VERDICT")
    print("=" * 80)
    print("  Read the three sections above and state:")
    print("  1. Is super-additivity in the feedback regime confirmed across seeds?")
    print("  2. Does the convexity baseline reproduce it?")
    print("  3. Does sub-additivity in fixed-N survive conditional overlap?")


if __name__ == "__main__":
    main()
