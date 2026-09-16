#!/usr/bin/env python3
"""Analyze NFS bandwidth time series for variance structure.

Reads the CSV from bw_timeseries.sh. Key output: autocorrelation structure,
degradation episode characterization, and a verdict on whether recent samples
predict near-future bandwidth.
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="NFS bandwidth variance analysis")
    p.add_argument("csv_path", help="Time series CSV from bw_timeseries.sh")
    p.add_argument("--out-fig", default="data/bw_timeseries.png",
                   help="Output figure path")
    p.add_argument("--degrade-threshold", type=float, default=0.50,
                   help="Fraction of median below which a sample is 'degraded' (default 0.50)")
    return p.parse_args()


def load_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "sample": int(row["sample"]),
                "timestamp": row["timestamp"],
                "wall_time": int(row["wall_time"]),
                "bw_mbps": float(row["bw_mbps"]),
                "iops": float(row["iops"]),
                "lat_mean_us": float(row["lat_mean_us"]),
                "lat_p50_us": float(row["lat_p50_us"]),
                "lat_p95_us": float(row["lat_p95_us"]),
                "lat_p99_us": float(row["lat_p99_us"]),
                "lat_max_us": float(row["lat_max_us"]),
            })
    return rows


def autocorrelation(x, max_lag):
    """Compute autocorrelation at lags 0..max_lag."""
    n = len(x)
    x = x - np.mean(x)
    var = np.sum(x ** 2) / n
    if var == 0:
        return np.zeros(max_lag + 1)
    acf = np.array([np.sum(x[:n - k] * x[k:]) / (n * var) for k in range(max_lag + 1)])
    return acf


def find_degradation_episodes(bw, threshold, sample_interval_s):
    """Find contiguous runs below threshold. Returns list of dicts."""
    episodes = []
    in_episode = False
    start = 0
    for i, v in enumerate(bw):
        if v < threshold:
            if not in_episode:
                in_episode = True
                start = i
        else:
            if in_episode:
                episodes.append({
                    "start_idx": start,
                    "end_idx": i - 1,
                    "length": i - start,
                    "duration_s": (i - start) * sample_interval_s,
                    "min_bw": float(np.min(bw[start:i])),
                    "mean_bw": float(np.mean(bw[start:i])),
                })
                in_episode = False
    if in_episode:
        episodes.append({
            "start_idx": start,
            "end_idx": len(bw) - 1,
            "length": len(bw) - start,
            "duration_s": (len(bw) - start) * sample_interval_s,
            "min_bw": float(np.min(bw[start:])),
            "mean_bw": float(np.mean(bw[start:])),
        })
    return episodes


def main():
    args = parse_args()
    rows = load_csv(args.csv_path)

    if len(rows) < 10:
        print("ERROR: need at least 10 samples for meaningful analysis", file=sys.stderr)
        sys.exit(1)

    bw = np.array([r["bw_mbps"] for r in rows])
    p99 = np.array([r["lat_p99_us"] for r in rows])
    wall = np.array([r["wall_time"] for r in rows])
    wall_min = wall / 60.0

    n = len(bw)
    if n > 1:
        sample_interval = float(np.median(np.diff(wall)))
    else:
        sample_interval = 20.0

    # ── Distribution ────────────────────────────────────────────
    print("=" * 70)
    print("BANDWIDTH DISTRIBUTION")
    print("=" * 70)
    print(f"  Samples:   {n}")
    print(f"  Duration:  {wall[-1] / 60:.1f} minutes ({wall[-1]:.0f} s)")
    print(f"  Interval:  ~{sample_interval:.0f} s between samples")
    print()
    print(f"  Min:       {np.min(bw):>8.1f} MB/s")
    print(f"  P5:        {np.percentile(bw, 5):>8.1f} MB/s")
    print(f"  P25:       {np.percentile(bw, 25):>8.1f} MB/s")
    print(f"  Median:    {np.median(bw):>8.1f} MB/s")
    print(f"  Mean:      {np.mean(bw):>8.1f} MB/s")
    print(f"  P75:       {np.percentile(bw, 75):>8.1f} MB/s")
    print(f"  P95:       {np.percentile(bw, 95):>8.1f} MB/s")
    print(f"  Max:       {np.max(bw):>8.1f} MB/s")
    print(f"  Std:       {np.std(bw):>8.1f} MB/s")
    print(f"  CV:        {np.std(bw) / np.mean(bw):>8.1%}")
    print(f"  Range:     {np.min(bw):.0f} – {np.max(bw):.0f} MB/s "
          f"({np.max(bw) / np.min(bw):.1f}x spread)")
    print()

    # ── Latency distribution ────────────────────────────────────
    print("=" * 70)
    print("P99 LATENCY DISTRIBUTION")
    print("=" * 70)
    print(f"  Min:       {np.min(p99):>8.0f} us")
    print(f"  Median:    {np.median(p99):>8.0f} us")
    print(f"  P95:       {np.percentile(p99, 95):>8.0f} us")
    print(f"  Max:       {np.max(p99):>8.0f} us")
    print()

    # ── Autocorrelation ─────────────────────────────────────────
    max_lag = min(30, n // 3)
    acf = autocorrelation(bw, max_lag)
    target_lags = [1, 2, 5, 10, 20]
    target_lags = [l for l in target_lags if l <= max_lag]

    print("=" * 70)
    print("AUTOCORRELATION (key analysis)")
    print("=" * 70)
    print(f"  {'Lag':>4}  {'Samples':>8}  {'~Time':>8}  {'ACF':>8}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*8}  {'-'*8}")
    for lag in range(1, max_lag + 1):
        time_str = f"{lag * sample_interval:.0f}s"
        marker = "  <---" if lag in target_lags else ""
        print(f"  {lag:>4}  {lag:>8}  {time_str:>8}  {acf[lag]:>8.3f}{marker}")
    print()

    half_life_lag = None
    for lag in range(1, max_lag + 1):
        if acf[lag] < 0.5:
            half_life_lag = lag
            break

    if half_life_lag is not None:
        half_life_s = half_life_lag * sample_interval
        print(f"  Autocorrelation drops below 0.5 at lag {half_life_lag} "
              f"(~{half_life_s:.0f} s)")
    else:
        print(f"  Autocorrelation remains >= 0.5 through lag {max_lag} "
              f"(~{max_lag * sample_interval:.0f} s)")
    print()

    # ── Degradation episodes ────────────────────────────────────
    median_bw = float(np.median(bw))
    threshold = median_bw * args.degrade_threshold
    episodes = find_degradation_episodes(bw, threshold, sample_interval)

    print("=" * 70)
    print(f"DEGRADATION EPISODES (below {args.degrade_threshold:.0%} of median = "
          f"{threshold:.0f} MB/s)")
    print("=" * 70)
    if episodes:
        print(f"  Found {len(episodes)} episode(s):\n")
        for i, ep in enumerate(episodes):
            t_start = wall[ep["start_idx"]] / 60
            t_end = wall[ep["end_idx"]] / 60
            depth = 1.0 - ep["mean_bw"] / median_bw
            print(f"  Episode {i+1}:")
            print(f"    Samples:   {ep['start_idx']+1} – {ep['end_idx']+1} "
                  f"({ep['length']} samples)")
            print(f"    Time:      {t_start:.1f} – {t_end:.1f} min "
                  f"(~{ep['duration_s']:.0f} s)")
            print(f"    Mean BW:   {ep['mean_bw']:.0f} MB/s "
                  f"({depth:.0%} below median)")
            print(f"    Min BW:    {ep['min_bw']:.0f} MB/s")
            print()

        durations = [ep["duration_s"] for ep in episodes]
        if len(episodes) >= 2:
            gaps = []
            for i in range(1, len(episodes)):
                gap_samples = episodes[i]["start_idx"] - episodes[i-1]["end_idx"] - 1
                gaps.append(gap_samples * sample_interval)
            print(f"  Typical episode duration: {np.median(durations):.0f} s "
                  f"(range {min(durations):.0f} – {max(durations):.0f} s)")
            print(f"  Gap between episodes:     {np.median(gaps):.0f} s "
                  f"(range {min(gaps):.0f} – {max(gaps):.0f} s)")
        else:
            print(f"  Episode duration: {durations[0]:.0f} s")
    else:
        print(f"  No episodes below {threshold:.0f} MB/s observed.")
    print()

    # ── Predictor verdict ───────────────────────────────────────
    print("=" * 70)
    print("VERDICT: IS A SINGLE RECENT SAMPLE A USEFUL PREDICTOR?")
    print("=" * 70)

    acf1 = acf[1] if len(acf) > 1 else 0
    acf5 = acf[5] if len(acf) > 5 else 0

    if half_life_lag is not None and half_life_s < 120:
        print(f"  NO. Autocorrelation drops below 0.5 within {half_life_s:.0f} s")
        print(f"  (lag {half_life_lag}). A bandwidth measurement taken now tells you")
        print(f"  very little about bandwidth {half_life_s * 2:.0f}+ seconds from now.")
        print()
        print(f"  Lag-1 autocorrelation: {acf1:.3f}")
        print(f"  Lag-5 autocorrelation: {acf5:.3f}")
        print()
        print(f"  A persistence-based predictor ('bandwidth will stay roughly what it")
        print(f"  was last time I checked') has a useful horizon of at most")
        print(f"  ~{half_life_s:.0f} s on this system. Beyond that, it is guessing.")
    elif half_life_lag is not None:
        print(f"  PARTIALLY. Autocorrelation decays to 0.5 at lag {half_life_lag}")
        print(f"  (~{half_life_s:.0f} s). A recent sample is informative for the next")
        print(f"  ~{half_life_s:.0f} s, but not beyond.")
        print()
        print(f"  Lag-1 autocorrelation: {acf1:.3f}")
        print(f"  Lag-5 autocorrelation: {acf5:.3f}")
    else:
        print(f"  YES (within our observation window). Autocorrelation remains >= 0.5")
        print(f"  through lag {max_lag} (~{max_lag * sample_interval:.0f} s).")
        print(f"  Bandwidth state is persistent at least on the timescale of minutes.")
        print()
        print(f"  Lag-1 autocorrelation: {acf1:.3f}")
        print(f"  Lag-5 autocorrelation: {acf5:.3f}")
        print()
        print(f"  However, this does NOT mean the system is stable — it may mean")
        print(f"  degradation episodes last longer than our measurement interval,")
        print(f"  making them look persistent. Check episode durations above.")
    print()

    # ── Observed range summary ──────────────────────────────────
    print("=" * 70)
    print("SUMMARY FOR RECORDING")
    print("=" * 70)
    print(f"  (a) Observed bandwidth range: {np.min(bw):.0f} – {np.max(bw):.0f} MB/s "
          f"({np.max(bw)/np.min(bw):.1f}x)")
    if episodes:
        typical_dur = np.median([ep["duration_s"] for ep in episodes])
        print(f"  (b) Typical degradation episode: ~{typical_dur:.0f} s")
    else:
        print(f"  (b) No degradation episodes below {args.degrade_threshold:.0%} of median")
    if half_life_lag is not None:
        print(f"  (c) Autocorrelation < 0.5 at lag {half_life_lag} (~{half_life_s:.0f} s)")
    else:
        print(f"  (c) Autocorrelation >= 0.5 through lag {max_lag} (~{max_lag * sample_interval:.0f} s)")
    if half_life_lag and half_life_s < 120:
        print(f"  (d) A single recent sample is NOT a useful predictor beyond ~{half_life_s:.0f} s")
    elif half_life_lag:
        print(f"  (d) A single recent sample is useful for ~{half_life_s:.0f} s, then decays")
    else:
        print(f"  (d) A single recent sample has predictive value for at least "
              f"~{max_lag * sample_interval:.0f} s")
    print()

    # ── Plot ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: bandwidth time series
    ax = axes[0, 0]
    ax.plot(wall_min, bw, ".-", c="steelblue", markersize=3, linewidth=0.8)
    ax.axhline(median_bw, c="gray", ls="--", lw=1, alpha=0.6, label=f"Median: {median_bw:.0f} MB/s")
    ax.axhline(threshold, c="red", ls=":", lw=1, alpha=0.5,
               label=f"Degradation threshold ({args.degrade_threshold:.0%} of median)")
    for ep in episodes:
        t0 = wall[ep["start_idx"]] / 60
        t1 = wall[ep["end_idx"]] / 60
        ax.axvspan(t0, t1, alpha=0.15, color="red")
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("Read bandwidth (MB/s)")
    ax.set_title("NFS read bandwidth over time")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Top-right: p99 latency time series
    ax = axes[0, 1]
    ax.plot(wall_min, p99, ".-", c="coral", markersize=3, linewidth=0.8)
    ax.set_xlabel("Time (minutes)")
    ax.set_ylabel("p99 latency (us)")
    ax.set_title("p99 read latency over time")
    ax.grid(True, alpha=0.3)

    # Bottom-left: bandwidth histogram
    ax = axes[1, 0]
    ax.hist(bw, bins=min(50, n // 3), color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(median_bw, c="gray", ls="--", lw=1.5, label=f"Median: {median_bw:.0f}")
    ax.axvline(np.mean(bw), c="orange", ls="--", lw=1.5, label=f"Mean: {np.mean(bw):.0f}")
    ax.set_xlabel("Read bandwidth (MB/s)")
    ax.set_ylabel("Count")
    ax.set_title("Bandwidth distribution")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Bottom-right: autocorrelation
    ax = axes[1, 1]
    lags = np.arange(0, max_lag + 1)
    ax.bar(lags, acf, color="steelblue", alpha=0.7, width=0.8)
    ax.axhline(0.5, c="red", ls="--", lw=1, alpha=0.6, label="ACF = 0.5")
    ax.axhline(0, c="black", lw=0.5)
    # 95% confidence interval for white noise
    ci = 1.96 / np.sqrt(n)
    ax.axhline(ci, c="gray", ls=":", lw=1, alpha=0.5)
    ax.axhline(-ci, c="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel(f"Lag (1 lag ≈ {sample_interval:.0f} s)")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("Bandwidth autocorrelation")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f"NFS bandwidth variance analysis — {n} samples over {wall[-1]/60:.0f} min",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    out_path = Path(args.out_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
