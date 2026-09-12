#!/usr/bin/env python3
"""Compute KV-cache fetch vs recompute crossover points.

Reads a prefill sweep JSON and one or more bandwidth summary CSVs.
Fits T_prefill(L) = a*L + b*L^2, then for each storage tier computes
the context length L* where fetching becomes faster than recomputing.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="KV-cache crossover analysis")
    p.add_argument("prefill_json", help="Path to prefill sweep JSON")
    p.add_argument("bw_csvs", nargs="+", help="Bandwidth summary CSVs (one per tier)")
    p.add_argument("--out-fig", default="data/crossover.png", help="Output figure path")
    return p.parse_args()


def load_prefill(path):
    with open(path) as f:
        data = json.load(f)
    lengths = []
    mean_times = []
    for m in data["measurements"]:
        if m["status"] != "ok" or not m["times_s"]:
            continue
        lengths.append(m["length"])
        mean_times.append(sum(m["times_s"]) / len(m["times_s"]))
    return (
        np.array(lengths, dtype=np.float64),
        np.array(mean_times, dtype=np.float64),
        data["kv_bytes_per_token"],
        data.get("model_config", {}).get("max_position_embeddings"),
    )


def load_bandwidth(csv_path):
    """Return {nstreams: bw_MB_s} and the tier name."""
    tier = None
    bw_by_nstreams = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tier = row["tier"]
            n = int(row["nstreams"])
            bw = float(row["agg_read_mbps"])
            bw_by_nstreams[n] = bw
    return tier, bw_by_nstreams


def fit_prefill(lengths, times):
    """Fit T = a*L + b*L^2 (no intercept) via least squares."""
    A = np.column_stack([lengths, lengths**2])
    coeffs, residuals, _, _ = np.linalg.lstsq(A, times, rcond=None)
    a, b = coeffs

    ss_res = np.sum((times - A @ coeffs) ** 2)
    ss_tot = np.sum((times - np.mean(times)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return a, b, r2


def compute_crossover(kv_bytes_per_token, bw_bytes_per_sec, a, b):
    """Compute L* = (kv_bytes_per_token / BW - a) / b.

    If kv_bytes_per_token / BW < a, fetching is always faster than
    recomputing (even at L=0 the fetch time is below the linear prefill
    term), so there is no crossover — fetching always wins.

    If b <= 0, the quadratic model is degenerate and we cannot solve.
    """
    t_fetch_per_token = kv_bytes_per_token / bw_bytes_per_sec

    if t_fetch_per_token <= a:
        return None, "fetching_always_wins"

    if b <= 0:
        return None, "degenerate_fit"

    l_star = (t_fetch_per_token - a) / b
    return l_star, "crossover"


def main():
    args = parse_args()

    lengths, times, kv_bpt, max_ctx = load_prefill(args.prefill_json)
    if len(lengths) < 2:
        print("ERROR: need at least 2 successful prefill measurements", file=sys.stderr)
        sys.exit(1)

    a, b, r2 = fit_prefill(lengths, times)
    print(f"Prefill fit: T = {a:.4e}*L + {b:.4e}*L^2  (R^2 = {r2:.6f})")
    print(f"KV bytes/token: {kv_bpt}")
    if max_ctx:
        print(f"Model max context: {max_ctx}")
    print()

    tiers = {}
    for csv_path in args.bw_csvs:
        tier_name, bw_map = load_bandwidth(csv_path)
        if tier_name is None:
            print(f"WARNING: empty CSV {csv_path}, skipping", file=sys.stderr)
            continue
        best_bw = max(bw_map.values())
        best_n = max(bw_map, key=bw_map.get)
        tiers[tier_name] = {
            "bw_map": bw_map,
            "best_bw_mbps": best_bw,
            "best_nstreams": best_n,
        }
        print(f"Tier '{tier_name}': peak {best_bw:.1f} MB/s at {best_n} streams")

    print()

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    L_plot = np.logspace(np.log10(lengths.min()), np.log10(max(lengths.max(), max_ctx or lengths.max())), 500)
    T_prefill_plot = a * L_plot + b * L_plot**2

    ax.plot(L_plot, T_prefill_plot, "k-", linewidth=2, label="T_prefill (measured fit)")
    ax.scatter(lengths, times, c="black", zorder=5, label="measured points")

    colors = plt.cm.tab10.colors
    verdicts = []

    for i, (tier_name, info) in enumerate(tiers.items()):
        bw_bytes = info["best_bw_mbps"] * 1e6
        T_fetch_plot = (kv_bpt / bw_bytes) * L_plot
        color = colors[i % len(colors)]

        ax.plot(L_plot, T_fetch_plot, "--", color=color, linewidth=1.5,
                label=f"T_fetch {tier_name} ({info['best_bw_mbps']:.0f} MB/s)")

        l_star, status = compute_crossover(kv_bpt, bw_bytes, a, b)

        if status == "fetching_always_wins":
            verdict = (f"Tier '{tier_name}': fetching ALWAYS wins — the per-token fetch "
                       f"time ({kv_bpt/bw_bytes:.4e} s) is below the linear prefill "
                       f"coefficient ({a:.4e} s), so recompute is never cheaper.")
            verdicts.append(verdict)
        elif status == "degenerate_fit":
            verdict = f"Tier '{tier_name}': degenerate quadratic fit (b <= 0), cannot compute crossover."
            verdicts.append(verdict)
        else:
            l_star_int = int(round(l_star))
            in_range = "inside" if (max_ctx and l_star_int <= max_ctx) else "outside"
            verdict = (f"Tier '{tier_name}': fetching beats recompute above "
                       f"{l_star_int:,} tokens ({in_range} the model's context limit).")
            verdicts.append(verdict)

            t_at_star = a * l_star + b * l_star**2
            ax.axvline(l_star, color=color, linestyle=":", alpha=0.7)
            ax.annotate(
                f"L*={l_star_int:,}",
                xy=(l_star, t_at_star),
                xytext=(l_star * 1.3, t_at_star * 1.2),
                arrowprops=dict(arrowstyle="->", color=color),
                fontsize=9,
                color=color,
            )

    if max_ctx:
        ax.axvline(max_ctx, color="gray", linestyle="--", alpha=0.5, linewidth=1)
        ax.text(max_ctx, ax.get_ylim()[1] * 0.9, f"max_ctx={max_ctx:,}",
                ha="right", va="top", fontsize=8, color="gray", rotation=90)

    ax.set_xscale("log")
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("KV-cache fetch vs recompute crossover")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    out_path = Path(args.out_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    print()

    print("=" * 60)
    print("VERDICTS")
    print("=" * 60)
    for v in verdicts:
        print(f"  {v}")
    print()


if __name__ == "__main__":
    main()
