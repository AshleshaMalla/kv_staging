#!/usr/bin/env python3
"""Analyze controlled contention experiment results.

Reads the victim timeseries CSV and aggressor event log from the experiment
directory. Produces transition timing analysis, dose-response curve, and
autocorrelation comparison against quiescent baseline.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


BASELINE_BW = 2588.0   # MB/s, from quiescent timeseries
BASELINE_STD = 17.0     # MB/s
THRESH_SIGMA = 3


def parse_args():
    p = argparse.ArgumentParser(description="Contention experiment analysis")
    p.add_argument("exp_dir", help="Experiment directory")
    p.add_argument("--out-fig", default=None, help="Output figure path")
    p.add_argument("--quiescent-csv", default=None,
                   help="Quiescent timeseries CSV for ACF comparison")
    return p.parse_args()


def load_victim(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "sample": int(row["sample"]),
                "timestamp": row["timestamp"],
                "wall_time": int(row["wall_time"]),
                "phase": row["phase"],
                "agg_nodes": int(row["agg_nodes"]),
                "bw_mbps": float(row["bw_mbps"]),
                "iops": float(row["iops"]),
                "lat_mean_us": float(row["lat_mean_us"]),
                "lat_p50_us": float(row["lat_p50_us"]),
                "lat_p95_us": float(row["lat_p95_us"]),
                "lat_p99_us": float(row["lat_p99_us"]),
                "lat_max_us": float(row["lat_max_us"]),
            })
    return rows


def load_schedule(path):
    cycles = []
    t0 = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("T0="):
                t0 = int(line.split("=")[1])
            elif line and not line.startswith("#"):
                parts = line.split()
                if len(parts) >= 6:
                    cycles.append({
                        "cycle": int(parts[0]),
                        "agg_nodes": int(parts[1]),
                        "quiet_start": int(parts[2]),
                        "load_start": int(parts[3]),
                        "load_stop": int(parts[4]),
                        "recover_stop": int(parts[5]),
                    })
    return t0, cycles


def load_aggressor_events(path):
    events = []
    if not path.exists():
        return events
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                events.append({"label": parts[0], "epoch": float(parts[1])})
    return events


def autocorrelation(x, max_lag):
    n = len(x)
    x = x - np.mean(x)
    var = np.sum(x ** 2) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    return np.array([np.sum(x[:n-k] * x[k:]) / (n * var) for k in range(max_lag + 1)])


def main():
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    victim_csv = exp_dir / "victim_timeseries.csv"
    schedule_file = exp_dir / "schedule.txt"
    events_file = exp_dir / "aggressor_events.log"

    if not victim_csv.exists():
        print(f"ERROR: {victim_csv} not found", file=sys.stderr)
        sys.exit(1)

    rows = load_victim(victim_csv)
    if not rows:
        print("ERROR: no victim data", file=sys.stderr)
        sys.exit(1)

    t0, cycles = load_schedule(schedule_file)
    agg_events = load_aggressor_events(events_file)

    bw = np.array([r["bw_mbps"] for r in rows])
    p99 = np.array([r["lat_p99_us"] for r in rows])
    wall = np.array([r["wall_time"] for r in rows])
    wall_min = wall / 60.0
    n = len(bw)

    sample_interval = float(np.median(np.diff(wall))) if n > 1 else 20.0
    degrade_thresh = BASELINE_BW - THRESH_SIGMA * BASELINE_STD

    print("=" * 70)
    print("CONTENTION EXPERIMENT RESULTS")
    print("=" * 70)
    print(f"  Samples:           {n}")
    print(f"  Duration:          {wall[-1]/60:.1f} min")
    print(f"  Baseline:          {BASELINE_BW:.0f} ± {BASELINE_STD:.0f} MB/s")
    print(f"  Degradation thresh: {degrade_thresh:.0f} MB/s (baseline - {THRESH_SIGMA}σ)")
    print(f"  Sample interval:   ~{sample_interval:.0f} s")
    print()

    # ── Per-cycle transition analysis ───────────────────────────
    print("=" * 70)
    print("TRANSITION ANALYSIS")
    print("=" * 70)

    cycle_results = []
    for cyc in cycles:
        cn = cyc["cycle"]
        agg_n = cyc["agg_nodes"]
        l_start = cyc["load_start"]
        l_stop = cyc["load_stop"]
        r_stop = cyc["recover_stop"]
        q_start = cyc["quiet_start"]

        # Find samples in each phase
        quiet_mask = (wall >= q_start) & (wall < l_start)
        load_mask = (wall >= l_start) & (wall < l_stop)
        recover_mask = (wall >= l_stop) & (wall < r_stop)

        quiet_bw = bw[quiet_mask]
        load_bw = bw[load_mask]
        recover_bw = bw[recover_mask]
        load_p99 = p99[load_mask]
        quiet_p99 = p99[quiet_mask]

        quiet_mean = float(np.mean(quiet_bw)) if len(quiet_bw) else BASELINE_BW
        load_mean = float(np.mean(load_bw)) if len(load_bw) else 0
        recover_mean = float(np.mean(recover_bw)) if len(recover_bw) else 0

        degradation = 1.0 - load_mean / quiet_mean if quiet_mean > 0 else 0

        # Onset lag: first sample after load_start that drops below threshold
        onset_lag = None
        load_wall = wall[load_mask]
        load_bw_arr = bw[load_mask]
        for j in range(len(load_bw_arr)):
            if load_bw_arr[j] < degrade_thresh:
                onset_lag = float(load_wall[j] - l_start)
                break

        # Recovery lag: first sample after load_stop that returns above threshold
        recovery_lag = None
        rec_wall = wall[recover_mask]
        rec_bw_arr = bw[recover_mask]
        for j in range(len(rec_bw_arr)):
            if rec_bw_arr[j] >= degrade_thresh:
                recovery_lag = float(rec_wall[j] - l_stop)
                break

        # Check if p99 degrades before mean BW
        p99_onset = None
        quiet_p99_mean = float(np.mean(quiet_p99)) if len(quiet_p99) else 4112.0
        quiet_p99_std = float(np.std(quiet_p99)) if len(quiet_p99) > 1 else 100.0
        p99_thresh = quiet_p99_mean + THRESH_SIGMA * max(quiet_p99_std, 100.0)
        load_p99_arr = p99[load_mask]
        for j in range(len(load_p99_arr)):
            if load_p99_arr[j] > p99_thresh:
                p99_onset = float(load_wall[j] - l_start)
                break

        result = {
            "cycle": cn, "agg_nodes": agg_n,
            "quiet_bw": quiet_mean, "load_bw": load_mean,
            "recover_bw": recover_mean, "degradation": degradation,
            "onset_lag": onset_lag, "recovery_lag": recovery_lag,
            "p99_onset": p99_onset,
            "load_p99_mean": float(np.mean(load_p99)) if len(load_p99) else 0,
            "quiet_p99_mean": quiet_p99_mean,
        }
        cycle_results.append(result)

        print(f"\n  Cycle {cn}: {agg_n} aggressor nodes")
        print(f"    Quiet BW:      {quiet_mean:>8.0f} MB/s")
        print(f"    Load BW:       {load_mean:>8.0f} MB/s  ({degradation:.0%} degradation)")
        print(f"    Recovery BW:   {recover_mean:>8.0f} MB/s")
        print(f"    Onset lag:     {onset_lag:.0f} s" if onset_lag is not None
              else "    Onset lag:     NO DEGRADATION DETECTED")
        print(f"    Recovery lag:  {recovery_lag:.0f} s" if recovery_lag is not None
              else "    Recovery lag:  DID NOT RECOVER (or no degradation)")
        print(f"    p99 onset:     {p99_onset:.0f} s" if p99_onset is not None
              else "    p99 onset:     no p99 spike detected")
        if p99_onset is not None and onset_lag is not None:
            if p99_onset < onset_lag:
                print(f"    *** p99 latency degraded {onset_lag - p99_onset:.0f}s BEFORE mean BW ***")
            elif p99_onset > onset_lag:
                print(f"    BW degraded {p99_onset - onset_lag:.0f}s before p99")
            else:
                print(f"    p99 and BW degraded simultaneously")

    # ── Dose-response ───────────────────────────────────────────
    print("\n")
    print("=" * 70)
    print("DOSE-RESPONSE CURVE")
    print("=" * 70)
    print(f"  {'Agg nodes':>10}  {'Quiet BW':>10}  {'Load BW':>10}  {'Degradation':>12}  {'Load p99':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*12}  {'-'*10}")
    for r in cycle_results:
        print(f"  {r['agg_nodes']:>10}  {r['quiet_bw']:>8.0f} MB  {r['load_bw']:>8.0f} MB  "
              f"{r['degradation']:>11.0%}  {r['load_p99_mean']:>8.0f} us")
    print()

    # ── Autocorrelation comparison ──────────────────────────────
    max_lag = min(30, n // 3)
    acf_full = autocorrelation(bw, max_lag)

    print("=" * 70)
    print("AUTOCORRELATION COMPARISON")
    print("=" * 70)
    print(f"  {'Lag':>4}  {'~Time':>8}  {'Full expt':>10}", end="")

    quiescent_acf = None
    if args.quiescent_csv and Path(args.quiescent_csv).exists():
        q_rows = list(csv.DictReader(open(args.quiescent_csv)))
        q_bw = np.array([float(r["bw_mbps"]) for r in q_rows])
        quiescent_acf = autocorrelation(q_bw, max_lag)
        print(f"  {'Quiescent':>10}", end="")
    print()
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}", end="")
    if quiescent_acf is not None:
        print(f"  {'-'*10}", end="")
    print()

    for lag in range(1, max_lag + 1):
        time_str = f"{lag * sample_interval:.0f}s"
        line = f"  {lag:>4}  {time_str:>8}  {acf_full[lag]:>10.3f}"
        if quiescent_acf is not None:
            line += f"  {quiescent_acf[lag]:>10.3f}"
        print(line)
    print()

    half_life_full = None
    for lag in range(1, max_lag + 1):
        if acf_full[lag] < 0.5:
            half_life_full = lag
            break

    if half_life_full:
        print(f"  Full experiment ACF < 0.5 at lag {half_life_full} "
              f"(~{half_life_full * sample_interval:.0f} s)")
    else:
        print(f"  Full experiment ACF >= 0.5 through lag {max_lag}")

    if quiescent_acf is not None:
        print(f"  Quiescent ACF at lag 1: {quiescent_acf[1]:.3f}")
    print(f"  Full expt ACF at lag 1: {acf_full[1]:.3f}")
    print()

    if quiescent_acf is not None and acf_full[1] > quiescent_acf[1] + 0.1:
        print(f"  The contention signal INCREASES autocorrelation — the regime shifts")
        print(f"  (quiet → degraded → quiet) create slow-moving structure that a")
        print(f"  persistence estimator can partially track.")
    elif quiescent_acf is not None:
        print(f"  Autocorrelation difference is small — the contention transitions")
        print(f"  are too fast relative to the sample interval for a persistence")
        print(f"  estimator to exploit.")
    print()

    # ── Verdict ─────────────────────────────────────────────────
    print("=" * 70)
    print("ESTIMATOR VERDICT")
    print("=" * 70)
    onset_lags = [r["onset_lag"] for r in cycle_results if r["onset_lag"] is not None]
    recovery_lags = [r["recovery_lag"] for r in cycle_results if r["recovery_lag"] is not None]

    if onset_lags:
        print(f"  Onset lags:    {', '.join(f'{l:.0f}s' for l in onset_lags)}")
        print(f"  Recovery lags: {', '.join(f'{l:.0f}s' for l in recovery_lags)}")
    else:
        print(f"  No degradation detected at any aggressor level.")

    p99_leads = sum(1 for r in cycle_results
                    if r["p99_onset"] is not None and r["onset_lag"] is not None
                    and r["p99_onset"] < r["onset_lag"])
    if p99_leads > 0:
        print(f"\n  p99 latency was an EARLIER signal than mean BW in {p99_leads}/{len(cycle_results)} cycles.")
        print(f"  A controller should watch p99 latency, not just throughput.")
    print()

    # ── Plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(16, 14), sharex=True)

    # Phase shading
    for ax in axes[:2]:
        for cyc in cycles:
            ax.axvspan(cyc["load_start"]/60, cyc["load_stop"]/60,
                       alpha=0.15, color="red", label=None)
            ax.axvline(cyc["load_start"]/60, c="red", ls="--", lw=0.8, alpha=0.5)
            ax.axvline(cyc["load_stop"]/60, c="green", ls="--", lw=0.8, alpha=0.5)

    # Top: bandwidth
    ax = axes[0]
    ax.plot(wall_min, bw, ".-", c="steelblue", markersize=2, linewidth=0.8)
    ax.axhline(BASELINE_BW, c="gray", ls="--", lw=1, alpha=0.6,
               label=f"Baseline: {BASELINE_BW:.0f} MB/s")
    ax.axhline(degrade_thresh, c="orange", ls=":", lw=1, alpha=0.6,
               label=f"Degrade threshold: {degrade_thresh:.0f} MB/s")
    for cyc in cycles:
        mid = (cyc["load_start"] + cyc["load_stop"]) / 2 / 60
        ax.text(mid, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 100, f"{cyc['agg_nodes']}N",
                ha="center", va="bottom", fontsize=9, color="red", fontweight="bold")
    ax.set_ylabel("Read bandwidth (MB/s)")
    ax.set_title("Victim bandwidth during contention experiment")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Middle: p99 latency
    ax = axes[1]
    ax.plot(wall_min, p99, ".-", c="coral", markersize=2, linewidth=0.8)
    ax.set_ylabel("p99 latency (us)")
    ax.set_title("p99 read latency")
    ax.grid(True, alpha=0.3)

    # Bottom: dose-response
    ax = axes[2]
    agg_ns = [r["agg_nodes"] for r in cycle_results]
    degs = [r["degradation"] * 100 for r in cycle_results]
    ax.bar(range(len(agg_ns)), degs, color="steelblue", alpha=0.7)
    ax.set_xticks(range(len(agg_ns)))
    ax.set_xticklabels([str(n) for n in agg_ns])
    ax.set_xlabel("Aggressor node count")
    ax.set_ylabel("Bandwidth degradation (%)")
    ax.set_title("Dose-response: aggressor count vs victim degradation")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    out_path = Path(args.out_fig) if args.out_fig else exp_dir / "contention_response.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    main()
