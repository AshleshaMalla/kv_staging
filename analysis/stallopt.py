#!/usr/bin/env python3
"""ObjectCache bandwidth allocation policies and TTFT model.

Reproduces the scheduler from Zhu et al., "ObjectCache: An Elastic KV Cache
Management System for LLM Serving" (arXiv:2605.22850), sections 3.5-3.6.

All bandwidths are in bytes/sec internally; use gbps_to_Bps / Bps_to_gbps
for conversion to the paper's Gbps convention.
"""

import bisect
import csv
from dataclasses import dataclass
from itertools import permutations
import math
import random

# --------------------------------------------------------------------------
# Model constants (Llama 3.1 8B on A100 80GB, paper Table A8)
# --------------------------------------------------------------------------
L = 32                          # transformer layers
BYTES_PER_TOKEN_PER_LAYER = 4096  # KV cache bytes per token per layer
DELTA_GBPS = 5.0                # calibration offset (paper section 3.6)


def gbps_to_Bps(gbps):
    """Gigabits/sec to bytes/sec."""
    return gbps * 1e9 / 8


def Bps_to_gbps(Bps):
    """Bytes/sec to gigabits/sec."""
    return Bps * 8 / 1e9


def Bps_to_GBps(Bps):
    """Bytes/sec to gigabytes/sec."""
    return Bps / 1e9


# --------------------------------------------------------------------------
# Request representation
# --------------------------------------------------------------------------
@dataclass
class Request:
    """A KV-cache retrieval request, characterized by its Table A8 entry."""
    context_tokens: int
    hit_ratio: float
    t_total_ms: float  # measured single-request TTFT (Table A8)

    @property
    def cached_tokens(self):
        return round(self.context_tokens * self.hit_ratio)

    @property
    def s(self):
        """Per-layer transfer size (bytes)."""
        return self.cached_tokens * BYTES_PER_TOKEN_PER_LAYER

    @property
    def c(self):
        """Per-layer compute window (seconds).  c_i = T_total / L."""
        return self.t_total_ms / L / 1000.0

    @property
    def r_star(self):
        """Zero-stall bandwidth (bytes/sec).  r*_i = s_i / c_i."""
        return self.s / self.c

    @property
    def label(self):
        return f"{self.context_tokens // 1024}K/{self.hit_ratio:.1%}"


# --------------------------------------------------------------------------
# Table A8: single-request characterization data
# --------------------------------------------------------------------------
TABLE_A8 = [
    Request(4096,   0.500,   185.31),
    Request(4096,   0.875,    63.47),
    Request(16384,  0.500,   955.89),
    Request(16384,  0.875,   281.76),
    Request(32768,  0.500,  2589.25),
    Request(32768,  0.875,   763.19),
    Request(65536,  0.500,  8672.79),
    Request(65536,  0.875,  2423.90),
]

TABLE_A8_PUBLISHED = {
    (4096,   0.500): (2048,    185.31,   5.79, 1.45),
    (4096,   0.875): (3584,     63.47,   1.98, 7.41),
    (16384,  0.500): (8192,    955.89,  29.87, 1.12),
    (16384,  0.875): (14336,   281.76,   8.80, 6.67),
    (32768,  0.500): (16384,  2589.25,  80.91, 0.83),
    (32768,  0.875): (28672,   763.19,  23.85, 4.92),
    (65536,  0.500): (32768,  8672.79, 271.02, 0.50),
    (65536,  0.875): (57344,  2423.90,  75.75, 3.10),
}


# --------------------------------------------------------------------------
# Allocation policies
# --------------------------------------------------------------------------

def allocate_equal(requests, cap_Bps):
    """Equal: same bandwidth to every request."""
    n = len(requests)
    return [cap_Bps / n] * n


def allocate_kv_prop(requests, cap_Bps):
    """KV-prop: proportional to retrieved KV cache size (cached tokens)."""
    tokens = [r.cached_tokens for r in requests]
    total = sum(tokens)
    return [cap_Bps * t / total for t in tokens]


def allocate_bw_prop(requests, cap_Bps):
    """BW-prop: proportional to zero-stall bandwidth r*_i."""
    stars = [r.r_star for r in requests]
    total = sum(stars)
    return [cap_Bps * rs / total for rs in stars]


def allocate_stall_opt(requests, cap_Bps, delta_Bps=0.0):
    """Stall-opt (or Calibrated Stall-opt when delta_Bps > 0).

    Solves:  min sum_i(s_i / r_i)
             s.t. sum(r_i) <= B,  0 < r_i <= r*_i + delta

    Closed-form via KKT.  For uncapped request i:
        dL/dr_i = -s_i/r_i^2 + lambda = 0   =>   r_i = sqrt(s_i / lambda)

    Uncapped allocations are therefore proportional to sqrt(s_i).
    Iterative water-filling: cap requests whose uncapped optimum exceeds
    their bound, subtract their bandwidth, and re-solve for the rest.
    Converges in at most N iterations.
    """
    n = len(requests)
    caps = [r.r_star + delta_Bps for r in requests]

    if sum(caps) <= cap_Bps:
        return list(caps)

    active = list(range(n))
    alloc = [0.0] * n
    remaining = cap_Bps

    while active:
        sqrt_s = [math.sqrt(requests[i].s) for i in active]
        total_sqrt = sum(sqrt_s)

        newly_capped = []
        for j, i in enumerate(active):
            r_i = remaining * sqrt_s[j] / total_sqrt
            if r_i >= caps[i]:
                newly_capped.append(i)

        if not newly_capped:
            for j, i in enumerate(active):
                alloc[i] = remaining * sqrt_s[j] / total_sqrt
            break

        capped_set = set(newly_capped)
        for i in newly_capped:
            alloc[i] = caps[i]
            remaining -= caps[i]
        active = [i for i in active if i not in capped_set]

    return alloc


# --------------------------------------------------------------------------
# TTFT model (paper section 3.5, Equation 3)
# --------------------------------------------------------------------------
#
# Original formula:
#     T_TTFT = X_0 + sum_{l=0}^{L-2} max(X_{l+1}, C_l) + C_{L-1}
#
# With X_l = X = s/r and C_l = C constant across all L layers (footnote 1):
#     X_0 = X
#     sum_{l=0}^{L-2} max(X_{l+1}, C_l)  =  (L-1) identical terms max(X, C)
#                                          =  (L-1) * max(X, C)
#     C_{L-1} = C
#
# Therefore:
#     T = X + (L-1)*max(X, C) + C
#
# Case X >= C  (transfer-bound, each layer stalls):
#     T = X + (L-1)*X + C = L*X + C
#
# Case X < C  (compute-bound, transfers hide in pipeline):
#     T = X + (L-1)*C + C = X + L*C
#
# Baseline (infinite BW, X -> 0):
#     T_base = 0 + (L-1)*C + C = L*C
#
# Added TTFT  (delta = T - T_base):
#     X >= C:  L*X + C - L*C = L*(X - C) + C = L*tau + C
#     X <  C:  X + L*C - L*C = X
#     where tau = max(0, X - C)  is the per-layer stall.


def ttft_ms(request, r_Bps):
    """TTFT in milliseconds for a single request at bandwidth r_Bps."""
    x = request.s / r_Bps   # per-layer transfer time (sec)
    c = request.c            # per-layer compute time (sec)
    t = x + (L - 1) * max(x, c) + c
    return t * 1000.0


def delta_ttft_ms(request, r_Bps):
    """Added TTFT (ms) over the infinite-bandwidth baseline T_base = L*C."""
    return ttft_ms(request, r_Bps) - L * request.c * 1000.0


# --------------------------------------------------------------------------
# Workload definitions (paper Tables A9 / A12)
# --------------------------------------------------------------------------

def get_workload_requests():
    """The 4 requests used in Workloads A and B (Table A9)."""
    return [
        TABLE_A8[2],   # 16K,  50.0%
        TABLE_A8[3],   # 16K,  87.5%
        TABLE_A8[6],   # 64K,  50.0%
        TABLE_A8[7],   # 64K,  87.5%
    ]


POLICY_ORDER = ["Equal", "KV-prop", "BW-prop", "Stall-opt", "Cal. Stall-opt"]

POLICIES = {
    "Equal":          lambda reqs, cap: allocate_equal(reqs, cap),
    "KV-prop":        lambda reqs, cap: allocate_kv_prop(reqs, cap),
    "BW-prop":        lambda reqs, cap: allocate_bw_prop(reqs, cap),
    "Stall-opt":      lambda reqs, cap: allocate_stall_opt(reqs, cap),
    "Cal. Stall-opt": lambda reqs, cap: allocate_stall_opt(
        reqs, cap, delta_Bps=gbps_to_Bps(DELTA_GBPS)),
}


def run_workload(requests, cap_gbps):
    """Run all policies.  Returns {name: (allocs_gbps, agg_delta_ttft_ms)}."""
    cap_Bps = gbps_to_Bps(cap_gbps)
    results = {}
    for name in POLICY_ORDER:
        allocs = POLICIES[name](requests, cap_Bps)
        allocs_gbps = [Bps_to_gbps(a) for a in allocs]
        agg_delta = sum(
            delta_ttft_ms(r, a) for r, a in zip(requests, allocs)
        )
        results[name] = (allocs_gbps, agg_delta)
    return results


# --------------------------------------------------------------------------
# Time-varying bandwidth simulation (Step 2a)
# --------------------------------------------------------------------------

TIMESERIES_CSV = "data/raw/bw_timeseries_20260913T211736Z/timeseries.csv"


class BandwidthTrace:
    """Piecewise-constant bandwidth trace.

    bws[i] is active from times[i] to times[i+1]; bws[-1] extends to +inf.
    """

    def __init__(self, times, bws_Bps):
        self.times = list(times)
        self.bws = list(bws_Bps)

    def bw_at(self, t):
        idx = bisect.bisect_right(self.times, t) - 1
        return self.bws[max(0, min(idx, len(self.bws) - 1))]

    def next_change(self, t):
        idx = bisect.bisect_right(self.times, t)
        return self.times[idx] if idx < len(self.times) else float('inf')

    @property
    def duration(self):
        return self.times[-1] - self.times[0]

    @property
    def mean(self):
        total_t, total_bw_t = 0.0, 0.0
        for i in range(len(self.times) - 1):
            dt = self.times[i + 1] - self.times[i]
            total_t += dt
            total_bw_t += self.bws[i] * dt
        return total_bw_t / total_t if total_t > 0 else self.bws[0]

    @property
    def cv(self):
        m = self.mean
        if m <= 0:
            return 0.0
        total_t = 0.0
        total_sq_t = 0.0
        for i in range(len(self.times) - 1):
            dt = self.times[i + 1] - self.times[i]
            total_t += dt
            total_sq_t += self.bws[i] ** 2 * dt
        var = total_sq_t / total_t - m * m
        return math.sqrt(max(0.0, var)) / m


def make_constant_trace(cap_Bps, duration=1200.0):
    return BandwidthTrace([0.0, duration], [cap_Bps, cap_Bps])


def make_quiescent_trace(target_mean_Bps, csv_path=TIMESERIES_CSV):
    times, bws = [], []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['wall_time']))
            bws.append(float(row['bw_mbps']) * 1048576)
    total_t = times[-1] - times[0]
    raw_mean = sum(
        bws[i] * (times[i + 1] - times[i]) for i in range(len(bws) - 1)
    ) / total_t
    scale = target_mean_Bps / raw_mean
    bws = [b * scale for b in bws]
    return BandwidthTrace(times, bws)


def make_bimodal_trace(mean_Bps, degradation_ratio=2.3,
                       episode_sec=30.0, n_cycles=20):
    R = degradation_ratio
    B_high = mean_Bps * 2 * R / (R + 1)
    B_low = mean_Bps * 2 / (R + 1)
    times, bws = [], []
    t = 0.0
    for _ in range(n_cycles):
        times.append(t)
        bws.append(B_high)
        t += episode_sec
        times.append(t)
        bws.append(B_low)
        t += episode_sec
    times.append(t)
    bws.append(B_high)
    return BandwidthTrace(times, bws)


def make_uniform_trace(mean_Bps, interval_sec=0.1,
                       duration_sec=600.0, seed=42):
    rng = random.Random(seed)
    times, bws = [], []
    t = 0.0
    while t < duration_sec:
        times.append(t)
        bws.append(max(1.0, rng.uniform(0, 2 * mean_Bps)))
        t += interval_sec
    return BandwidthTrace(times, bws)


def _transfer_time(remaining, request_idx, all_reqs, policy_fn,
                   trace, epoch_sec, t0, t_now):
    """Wall-clock seconds to transfer `remaining` bytes starting at t_now."""
    t = t_now
    while remaining > 1e-6:
        epoch_num = math.floor((t - t0) / epoch_sec)
        epoch_start = t0 + epoch_num * epoch_sec
        epoch_end = epoch_start + epoch_sec

        B_obs = trace.bw_at(epoch_start)
        if B_obs <= 0:
            t = epoch_end
            continue

        allocs = policy_fn(all_reqs, B_obs)
        r_i = allocs[request_idx]

        t_boundary = min(epoch_end, trace.next_change(t))

        B_actual = trace.bw_at(t)
        r_actual = r_i * min(1.0, B_actual / B_obs)

        if r_actual <= 0:
            t = t_boundary
            continue

        dt = t_boundary - t
        if dt <= 0:
            t = t_boundary + 1e-9
            continue

        can_send = r_actual * dt
        if can_send >= remaining:
            t += remaining / r_actual
            remaining = 0.0
        else:
            remaining -= can_send
            t = t_boundary

    return t - t_now


def _sim_one(request, idx, all_reqs, policy_fn, trace, epoch_sec, t0):
    """TTFT in seconds for one request under time-varying BW."""
    t = t0
    t += _transfer_time(request.s, idx, all_reqs, policy_fn,
                        trace, epoch_sec, t0, t)
    for _ in range(1, L):
        dt_xfer = _transfer_time(request.s, idx, all_reqs, policy_fn,
                                 trace, epoch_sec, t0, t)
        t += max(dt_xfer, request.c)
    t += request.c
    return t - t0


def simulate_workload_varying(requests, policy_name, trace,
                              epoch_sec, t_start=0.0):
    """Per-request delta TTFT (ms) under time-varying BW."""
    policy_fn = POLICIES[policy_name]
    results = []
    for idx, req in enumerate(requests):
        ttft_s = _sim_one(req, idx, requests, policy_fn,
                          trace, epoch_sec, t_start)
        results.append((ttft_s - L * req.c) * 1000.0)
    return results


def run_trace_experiment(requests, policy_name, trace,
                         epoch_sec, n_starts=200):
    """Average aggregate delta TTFT (ms) over n_starts start times."""
    margin = 60.0
    max_start = trace.duration - margin
    if max_start <= 0:
        starts = [0.0]
    else:
        starts = [trace.times[0] + max_start * i / (n_starts - 1)
                  for i in range(n_starts)]
    total = 0.0
    for t0 in starts:
        deltas = simulate_workload_varying(requests, policy_name,
                                           trace, epoch_sec, t0)
        total += sum(deltas)
    return total / len(starts)
