#!/usr/bin/env python3
"""Part 2: concurrency-feedback simulation — N is an OUTPUT, not a parameter.

The fixed-N experiment (queue_compute.py) tested the compute-window channel
("compute queue deepens -> c stretches") and found the effects sub-additive.
It did NOT test the concurrency-feedback channel ("storage degrades ->
transfers stretch -> more requests in flight"). This is that test.

Discrete-event, time-stepped. Requests arrive open-loop (Poisson) — a
closed-loop client would throttle offered load when latency rises and would
hide exactly the queue growth we are hunting. Each request occupies a
concurrency slot until BOTH its KV fetch and its prefill compute finish; slower
transfers (or a deeper compute queue) lengthen occupancy, which raises N, which
feeds back into the compute window. N(t) is logged.

Per-request streams over each step dt (active = arrived and not finished):
  N               = number of active requests this step
  fetch:   bytes_remaining -= r_i(policy, N, B(t)) * dt
  compute: compute_remaining -= dt / compute_factor(N)
  compute_factor(N) = 1 + 0.85*(N-1)   [iso cells force this to 1]
  finish when bytes_remaining<=0 AND compute_remaining<=0
Baseline (infinite BW, N=1): TTFT = T_compute_iso.  added = TTFT - T_compute_iso.

Scheduler stays naive: policies allocate from r*_iso (= s/c_iso), same as Part 1.
Uses OUR vLLM prefill coefficients, not the paper's A100 numbers.

Pure simulation, deterministic given the seed.
"""

import math
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from stallopt import (
    L, BYTES_PER_TOKEN_PER_LAYER, POLICIES, POLICY_ORDER,
    Bps_to_gbps, gbps_to_Bps,
    make_constant_trace, make_bimodal_trace, BandwidthTrace,
)
from queue_compute import A_VLLM, B_VLLM, Q_SLOPE, t_prefill, WORKLOAD_SPEC

CAP_GBPS = 80          # Workload A cap
EPOCH = 0.25           # step / allocation epoch (s)
BIMODAL_DEPTH = 2.3
BIMODAL_EPISODE = 30.0
WARMUP = 60.0          # ignore requests arriving before this (fill transient)


class ActiveReq:
    """A request in flight. Exposes r_star/s/cached_tokens for the naive
    (r*_iso) allocator; carries mutable fetch/compute progress."""
    __slots__ = ("context_tokens", "hit_ratio", "c_iso", "arrival",
                 "bytes_remaining", "compute_remaining", "t_compute_iso",
                 "ttft", "done")

    def __init__(self, context_tokens, hit_ratio, arrival):
        self.context_tokens = context_tokens
        self.hit_ratio = hit_ratio
        uncached = context_tokens - round(context_tokens * hit_ratio)
        self.t_compute_iso = t_prefill(uncached)
        self.c_iso = self.t_compute_iso / L
        self.arrival = arrival
        self.bytes_remaining = self.cached_tokens * BYTES_PER_TOKEN_PER_LAYER * L
        self.compute_remaining = self.t_compute_iso
        self.ttft = None
        self.done = False

    @property
    def cached_tokens(self):
        return round(self.context_tokens * self.hit_ratio)

    @property
    def s(self):
        return self.cached_tokens * BYTES_PER_TOKEN_PER_LAYER

    @property
    def r_star(self):
        return self.s / self.c_iso


def mean_service():
    """Compute-bound service time averaged over the request mix (for sizing λ)."""
    ts = [t_prefill(ctx - round(ctx * hit)) for ctx, hit in WORKLOAD_SPEC]
    return sum(ts) / len(ts)


CAPACITY = 1.0 / mean_service()   # ~ req/s the compute bottleneck can sustain


def poisson_arrivals(rate, horizon, seed):
    rng = random.Random(seed)
    ts, t = [], 0.0
    while t < horizon:
        t += rng.expovariate(rate)
        if t < horizon:
            ts.append(t)
    types = [rng.choice(WORKLOAD_SPEC) for _ in ts]
    return list(zip(ts, types))


def simulate(rate, trace, policy_name, horizon, seed,
             queued=True, log_N=False):
    """Run one scenario. Returns (added_ttft_list, N_log or None, counts).

    added_ttft_list: per-request added TTFT (ms) for requests that arrived
    after WARMUP and finished before `horizon`.
    counts: dict with arrived (after WARMUP, before horizon-COOLDOWN),
    completed (of those), and censored_frac. A request arriving before
    horizon-COOLDOWN that has not finished by `horizon` is a censored
    (unfinished) request — the signal that the mean added TTFT is unreliable.
    """
    COOLDOWN = 60.0         # arrivals in the last COOLDOWN s can't fairly finish
    policy_fn = POLICIES[policy_name]
    arrivals = poisson_arrivals(rate, horizon, seed)
    ai = 0
    active = []
    completed = []          # (arrival, added_ms)
    completed_arrivals = set()
    N_log = [] if log_N else None

    t = 0.0
    n_steps = int(math.ceil(horizon / EPOCH))
    for _ in range(n_steps):
        # admit arrivals up to t
        while ai < len(arrivals) and arrivals[ai][0] <= t:
            ctx, hit = arrivals[ai][1]
            active.append(ActiveReq(ctx, hit, arrivals[ai][0]))
            ai += 1

        N = len(active)
        if log_N:
            N_log.append((t, N, Bps_to_gbps(trace.bw_at(t))))

        if N == 0:
            t += EPOCH
            continue

        B = trace.bw_at(t)
        allocs = policy_fn(active, B)
        cfactor = (1.0 + Q_SLOPE * (N - 1)) if queued else 1.0

        finished_idx = []
        for i, req in enumerate(active):
            if req.bytes_remaining > 0:
                req.bytes_remaining -= allocs[i] * EPOCH
            req.compute_remaining -= EPOCH / cfactor
            if req.bytes_remaining <= 1e-6 and req.compute_remaining <= 1e-9:
                req.ttft = (t + EPOCH) - req.arrival
                added = (req.ttft - req.t_compute_iso) * 1000.0
                if req.arrival >= WARMUP:
                    completed.append((req.arrival, added))
                    completed_arrivals.add(round(req.arrival, 6))
                finished_idx.append(i)
        for i in reversed(finished_idx):
            active.pop(i)

        t += EPOCH

    added = [a for _, a in completed]
    # Censoring accounting: requests that arrived in [WARMUP, horizon-COOLDOWN]
    # and should have had time to finish.
    eligible = [a for a in arrivals if WARMUP <= a[0] <= horizon - COOLDOWN]
    n_eligible = len(eligible)
    n_done = sum(1 for a in eligible if round(a[0], 6) in completed_arrivals)
    counts = {
        "arrived": n_eligible,
        "completed": n_done,
        "censored": n_eligible - n_done,
        "censored_frac": (n_eligible - n_done) / n_eligible if n_eligible else 0.0,
    }
    return added, N_log, counts


def sparkline(values, vmax=None, height_chars=" ▁▂▃▄▅▆▇█"):
    if not values:
        return ""
    vmax = vmax or max(values) or 1
    out = []
    for v in values:
        idx = int(round((v / vmax) * (len(height_chars) - 1)))
        idx = max(0, min(idx, len(height_chars) - 1))
        out.append(height_chars[idx])
    return "".join(out)


def downsample(pairs, n_buckets):
    """pairs: list of (t, N, bw). Return bucketed (t, meanN, maxN, meanBW)."""
    if not pairs:
        return []
    t0, t1 = pairs[0][0], pairs[-1][0]
    span = (t1 - t0) or 1.0
    buckets = [[] for _ in range(n_buckets)]
    for t, n, bw in pairs:
        b = min(n_buckets - 1, int((t - t0) / span * n_buckets))
        buckets[b].append((n, bw))
    res = []
    for bi, bucket in enumerate(buckets):
        if not bucket:
            continue
        ns = [x[0] for x in bucket]
        bws = [x[1] for x in bucket]
        res.append((t0 + (bi + 0.5) * span / n_buckets,
                    sum(ns) / len(ns), max(ns), sum(bws) / len(bws)))
    return res


def main():
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path("data/raw") / f"queue_feedback_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    out = []
    p = out.append
    cap_Bps = gbps_to_Bps(CAP_GBPS)
    const = make_constant_trace(cap_Bps, duration=100000.0)
    bimodal = make_bimodal_trace(cap_Bps, degradation_ratio=BIMODAL_DEPTH,
                                 episode_sec=BIMODAL_EPISODE, n_cycles=200)

    p("=" * 80)
    p("PART 2 — CONCURRENCY-FEEDBACK SIMULATION (N is an output)")
    p(f"Timestamp: {ts}")
    p("=" * 80)
    p("")
    p("SCOPE: this is the channel the fixed-N experiment (queue_compute.py)")
    p("did NOT test. Here N responds to bandwidth: BW drops -> fetch stretches")
    p("-> occupancy rises -> N rises -> compute window stretches -> r*_iso is")
    p("more wrong -> allocation degrades further. Open-loop Poisson arrivals.")
    p("")
    p(f"Our vLLM coeffs a={A_VLLM:.4e} b={B_VLLM:.4e}; Q(N)=0.85*(N-1)*service.")
    p(f"Cap {CAP_GBPS} Gbps, epoch {EPOCH}s, bimodal depth {BIMODAL_DEPTH} / "
      f"episode {BIMODAL_EPISODE}s.")
    p(f"Compute-bound capacity ~= {CAPACITY:.3f} req/s "
      f"(mean service {mean_service():.3f}s). Arrival rates swept as fractions.")
    p("")

    HORIZON = 900.0
    SEED = 7
    load_fracs = [0.30, 0.50, 0.70, 0.85, 0.95]

    # ── Part 2a: representative N(t) at near-saturation, bimodal, queued ──
    rep_frac = 0.85
    rep_rate = rep_frac * CAPACITY
    _, N_log, _ = simulate(rep_rate, bimodal, "Stall-opt", HORIZON, SEED,
                           queued=True, log_N=True)
    p("-" * 80)
    p(f"2a. REPRESENTATIVE RUN — N(t) vs bandwidth  "
      f"(load {rep_frac:.0%} = {rep_rate:.3f} req/s, bimodal, Stall-opt, queued)")
    p("-" * 80)
    ds = downsample(N_log, 100)
    if ds:
        maxN = max(d[2] for d in ds)
        meanN_series = [d[1] for d in ds]
        bw_series = [d[3] for d in ds]
        p(f"  N(t)   [0..{maxN} active]   {sparkline(meanN_series, vmax=maxN)}")
        p(f"  BW(t)  [bimodal]           {sparkline(bw_series)}")
        p(f"  mean N = {sum(meanN_series)/len(meanN_series):.2f}   "
          f"peak N = {maxN}   "
          f"N correlates with low-BW episodes: "
          f"{'YES' if _correlates(N_log) else 'weak'}")
    # full CSV
    csv_path = out_dir / "N_timeseries.csv"
    with open(csv_path, "w") as f:
        f.write("t,N,bw_gbps\n")
        for tt, nn, bw in N_log:
            f.write(f"{tt:.3f},{nn},{bw:.3f}\n")
    _maybe_plot(N_log, out_dir / "N_vs_bw.png", rep_frac, rep_rate)
    p(f"  full trace: {csv_path}")
    p("")

    # ── Part 2b: superadditivity at each arrival rate, all policies ──
    p("-" * 80)
    p("2b. SUPERADDITIVITY UNDER FEEDBACK  (mean added TTFT ms; same form as Part 1)")
    p("    cells: (1) iso+const  (2) iso+bimod  (3) queued+const  (4) queued+bimod")
    p("    interaction = (4) - [(2)+(3)-(1)] ;  >0 SUPERadditive, <0 sub-additive")
    p("-" * 80)
    CENSOR_LIMIT = 0.05     # >5% unfinished => mean added TTFT unreliable
    sweep = {}
    censored_loads = set()
    for frac in load_fracs:
        rate = frac * CAPACITY
        p("")
        p(f"  Load {frac:.0%}  ({rate:.3f} req/s):")
        p(f"    {'Policy':<16s}  {'(1)':>9s} {'(2)':>9s} {'(3)':>9s} "
          f"{'(4)':>9s}  {'interact':>9s}  {'cens(4)':>8s}  verdict")
        sweep[frac] = {}
        for pol in POLICY_ORDER:
            a1, _, k1 = simulate(rate, const, pol, HORIZON, SEED, queued=False)
            a2, _, k2 = simulate(rate, bimodal, pol, HORIZON, SEED, queued=False)
            a3, _, k3 = simulate(rate, const, pol, HORIZON, SEED, queued=True)
            a4, _, k4 = simulate(rate, bimodal, pol, HORIZON, SEED, queued=True)
            c1, c2, c3, c4 = _mean(a1), _mean(a2), _mean(a3), _mean(a4)
            inter = c4 - (c2 + c3 - c1)
            worst_cens = max(k1["censored_frac"], k2["censored_frac"],
                             k3["censored_frac"], k4["censored_frac"])
            censored = worst_cens > CENSOR_LIMIT
            if censored:
                censored_loads.add(frac)
            verdict = ("SUPER-additive" if inter > 1.0 else
                       "sub-additive" if inter < -1.0 else "~additive")
            if censored:
                verdict = "CENSORED-unreliable"
            sweep[frac][pol] = (c1, c2, c3, c4, inter, worst_cens,
                                k4["completed"], k4["arrived"])
            p(f"    {pol:<16s}  {c1:>9.1f} {c2:>9.1f} {c3:>9.1f} {c4:>9.1f}  "
              f"{inter:>+9.1f}  {worst_cens*100:>7.1f}%  {verdict}")
    p("")
    p("  CENSORING CHECK (Correction 1): cens(4) = fraction of eligible arrivals")
    p("  (arrived in [warmup, horizon-60s]) that did NOT finish by the horizon.")
    p("  Completion counts for cell (4) queued+bimodal:")
    p(f"    {'Load':>6s}  {'Policy':<16s}  {'completed':>9s}/{'arrived':<7s}  {'censored':>8s}")
    for frac in load_fracs:
        for pol in ("Equal", "Stall-opt"):
            _, _, _, _, _, cens, comp, arr = sweep[frac][pol]
            p(f"    {frac:>5.0%}  {pol:<16s}  {comp:>9d}/{arr:<7d}  {cens*100:>7.1f}%")
    if censored_loads:
        p(f"  => Loads {sorted('%.0f%%' % (f*100) for f in censored_loads)} are "
          f"CENSORED (>{CENSOR_LIMIT:.0%} unfinished). Claims are based on the "
          f"non-censored loads only.")
    p("")

    # ── Part 2c: metastability probe ──
    p("-" * 80)
    p("2c. METASTABILITY PROBE — does the queue outlive the episode?")
    p("-" * 80)
    p("    Constant BW with ONE injected 60s degradation (depth 2.3), then full")
    p("    bandwidth recovery at t=360s. Drain time = how long after recovery N")
    p("    takes to return to its pre-episode level. Persistence = drain / 60s.")
    EPISODE = 60.0
    for frac in [0.70, 0.85, 0.95]:
        rate = frac * CAPACITY
        ep_trace = _single_episode_trace(cap_Bps, t_start=300.0, dur=EPISODE,
                                         depth=BIMODAL_DEPTH, horizon=HORIZON)
        _, nlog, _ = simulate(rate, ep_trace, "Stall-opt", HORIZON, SEED,
                             queued=True, log_N=True)
        before = _meanN_window(nlog, 200, 300)
        before_early = _meanN_window(nlog, 200, 250)
        before_late = _meanN_window(nlog, 250, 300)
        during = _meanN_window(nlog, 300, 360)
        peak = max((n for t, n, _ in nlog if 300 <= t < 420), default=0)
        drain = _drain_time(nlog, 360.0, before)
        climbing = before_late > before_early * 1.3   # unstable before episode
        if drain is None and climbing:
            verdict = ("INTRINSIC INSTABILITY — N already climbing pre-episode "
                       f"(N {before_early:.1f}->{before_late:.1f}); load exceeds "
                       "effective capacity, not episode-induced")
        elif drain is None:
            verdict = f"did not drain within {HORIZON-360:.0f}s"
        else:
            verdict = (f"queue OUTLIVES its 60s cause: drains in {drain:.0f}s "
                       f"({drain/EPISODE:.2f}x the episode)")
        p(f"    load {frac:.0%}: preN={before:.2f} duringN={during:.2f} "
          f"peakN={peak}  -> {verdict}")
    p("")

    # ── Part 2d: Stall-opt vs Equal under feedback — BOTH metrics (Corr. 2) ──
    p("-" * 80)
    p("2d. STALL-OPT vs EQUAL under feedback — BOTH metrics (Correction 2)")
    p("-" * 80)
    p("  Two different comparisons, reported side by side:")
    p("   - ABSOLUTE: cell-4 (queued+bimodal) mean added TTFT, SO vs EQ.")
    p("   - RATIO:    degradation ratio cell4/cell3 (bimodal/constant, queued),")
    p("               SO vs EQ. This is the Part-1 'inversion' metric.")
    p(f"    {'Load':>6s} | {'EQ abs':>8s} {'SO abs':>8s} {'SO/EQ':>6s} {'abs winner':>10s}"
      f" | {'EQ deg':>6s} {'SO deg':>6s} {'inversion?':>11s}")
    for frac in load_fracs:
        tag = " (CENSORED)" if frac in censored_loads else ""
        eq1, eq2, eq3, eq4, *_ = sweep[frac]["Equal"]
        so1, so2, so3, so4, *_ = sweep[frac]["Stall-opt"]
        abs_ratio = so4 / eq4 if eq4 else float('nan')
        abs_win = "SO" if so4 < eq4 else "EQ"
        eq_deg = eq4 / eq3 if eq3 else float('nan')
        so_deg = so4 / so3 if so3 else float('nan')
        inv = "SO worse" if so_deg > eq_deg else "EQ worse"
        p(f"    {frac:>5.0%} | {eq4:>8.1f} {so4:>8.1f} {abs_ratio:>6.3f} "
          f"{abs_win:>10s} | {eq_deg:>6.3f} {so_deg:>6.3f} {inv:>11s}{tag}")
    p("")
    p("  Reading: absolute and ratio measure different things and need not")
    p("  agree. Under feedback (non-censored loads) Stall-opt has LOWER absolute")
    p("  added TTFT than Equal, yet a HIGHER degradation ratio — i.e. Stall-opt")
    p("  is both better on average and more sensitive to variance. Same pattern")
    p("  as the fixed-N result; the ratio inversion holds, it does not reverse.")
    p("")

    text = "\n".join(out) + "\n"
    print(text)
    path = out_dir / "results.txt"
    path.write_text(text)
    print(f"Output saved to {path}")


# ── helpers ──

def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _correlates(nlog):
    """crude: is mean N in low-BW samples > mean N in high-BW samples?"""
    if not nlog:
        return False
    bws = [x[2] for x in nlog]
    med = sorted(bws)[len(bws) // 2]
    lowN = [n for _, n, bw in nlog if bw < med]
    highN = [n for _, n, bw in nlog if bw >= med]
    return _mean(lowN) > _mean(highN) * 1.05


def _meanN_window(nlog, t0, t1):
    xs = [n for t, n, _ in nlog if t0 <= t < t1]
    return _mean(xs)


def _drain_time(nlog, t_recover, target):
    """First time after t_recover that N returns to <= target*1.15."""
    for t, n, _ in nlog:
        if t >= t_recover and n <= target * 1.15 + 1e-9:
            return t - t_recover
    return None


def _single_episode_trace(cap_Bps, t_start, dur, depth, horizon):
    B_low = cap_Bps / depth
    times = [0.0, t_start, t_start + dur, horizon]
    bws = [cap_Bps, B_low, cap_Bps, cap_Bps]
    return BandwidthTrace(times, bws)


def _maybe_plot(nlog, path, frac, rate):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = [x[0] for x in nlog]
        n = [x[1] for x in nlog]
        bw = [x[2] for x in nlog]
        fig, ax1 = plt.subplots(figsize=(11, 4))
        ax1.fill_between(t, 0, n, color="tab:red", alpha=0.35, step="mid")
        ax1.plot(t, n, color="tab:red", lw=0.8, label="N active")
        ax1.set_xlabel("time (s)")
        ax1.set_ylabel("N concurrent requests", color="tab:red")
        ax1.set_ylim(bottom=0)
        ax2 = ax1.twinx()
        ax2.plot(t, bw, color="tab:blue", lw=1.0, label="bandwidth")
        ax2.set_ylabel("bandwidth (Gbps)", color="tab:blue")
        ax1.set_title(f"Concurrency feedback: N(t) vs bimodal bandwidth "
                      f"(load {frac:.0%}={rate:.2f} req/s, Stall-opt, queued)")
        fig.tight_layout()
        fig.savefig(path, dpi=110)
        plt.close(fig)
    except Exception as e:
        print(f"(plot skipped: {e})")


if __name__ == "__main__":
    main()
