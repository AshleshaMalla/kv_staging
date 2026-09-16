#!/usr/bin/env python3
"""Analyze multi-node NFS bandwidth scaling.

Reads the per-node and aggregate CSVs from bw_multinode.sh, produces a scaling
table, a verdict on whether the backend is a per-node or shared ceiling, and a
plot of aggregate bandwidth vs node count.
"""

import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Multi-node NFS bandwidth scaling analysis")
    p.add_argument("per_node_csv", help="Per-node summary CSV from bw_multinode.sh")
    p.add_argument("aggregate_csv", help="Aggregate summary CSV from bw_multinode.sh")
    p.add_argument("--out-fig", default="data/multinode_scaling.png",
                   help="Output figure path")
    p.add_argument("--scaling-threshold", type=float, default=0.75,
                   help="Efficiency above this = near-linear scaling (default 0.75)")
    p.add_argument("--flat-threshold", type=float, default=0.35,
                   help="Efficiency below this = shared ceiling (default 0.35)")
    return p.parse_args()


def load_per_node(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "nodes": int(row["nodes"]),
                "rep": int(row["rep"]),
                "hostname": row["hostname"],
                "bw_mbps": float(row["bw_mbps"]),
                "iops": float(row["iops"]),
                "lat_mean_us": float(row["lat_mean_us"]),
                "lat_p50_us": float(row["lat_p50_us"]),
                "lat_p95_us": float(row["lat_p95_us"]),
                "lat_p99_us": float(row["lat_p99_us"]),
                "lat_max_us": float(row["lat_max_us"]),
                "start_ts": float(row["start_ts"]),
                "end_ts": float(row["end_ts"]),
            })
    return rows


def load_aggregate(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({
                "nodes": int(row["nodes"]),
                "rep": int(row["rep"]),
                "aggregate_bw_mbps": float(row["aggregate_bw_mbps"]),
                "mean_per_node_bw_mbps": float(row["mean_per_node_bw_mbps"]),
                "overlap_fraction": float(row["overlap_fraction"]),
            })
    return rows


def main():
    args = parse_args()
    per_node = load_per_node(args.per_node_csv)
    aggregate = load_aggregate(args.aggregate_csv)

    if not aggregate:
        print("ERROR: no data in aggregate CSV", file=sys.stderr)
        sys.exit(1)

    # ── Overlap check ───────────────────────────────────────────
    print("=" * 70)
    print("OVERLAP CHECK")
    print("=" * 70)
    bad_overlap = False
    for row in aggregate:
        if row["nodes"] > 1:
            flag = ""
            if row["overlap_fraction"] < 0.80:
                flag = "  *** POOR ***"
                bad_overlap = True
            print(f"  {row['nodes']} nodes  rep {row['rep']}:  "
                  f"overlap = {row['overlap_fraction']:.3f}{flag}")
    if bad_overlap:
        print("\n  WARNING: Some runs had poor overlap. Those aggregate numbers")
        print("  may not reflect true simultaneous load. Interpret with caution.")
    print()

    # ── Per node-count aggregation ──────────────────────────────
    by_nc = defaultdict(list)
    for row in aggregate:
        by_nc[row["nodes"]].append(row)

    per_node_by_nc = defaultdict(list)
    for row in per_node:
        per_node_by_nc[row["nodes"]].append(row)

    node_counts = sorted(by_nc.keys())
    baseline_bw_reps = [r["aggregate_bw_mbps"] for r in by_nc[1]] if 1 in by_nc else None

    if baseline_bw_reps:
        baseline_bw = np.mean(baseline_bw_reps)
    else:
        baseline_bw = None

    # ── Main table ──────────────────────────────────────────────
    print("=" * 70)
    print("SCALING TABLE (all reps shown)")
    print("=" * 70)
    header = (f"{'Nodes':>5}  {'Rep':>3}  {'Agg BW':>10}  {'Per-Node':>10}  "
              f"{'Efficiency':>10}  {'p99 lat':>10}  {'Overlap':>8}")
    print(header)
    print("-" * len(header))

    summary = {}  # nc -> {mean_agg, mean_per_node, efficiency, p99}
    for nc in node_counts:
        reps = by_nc[nc]
        agg_vals = [r["aggregate_bw_mbps"] for r in reps]
        per_node_vals = [r["mean_per_node_bw_mbps"] for r in reps]

        p99_vals = [r["lat_p99_us"] for r in per_node_by_nc[nc]]
        mean_p99 = np.mean(p99_vals) if p99_vals else 0.0

        if baseline_bw and baseline_bw > 0:
            eff = np.mean(agg_vals) / (nc * baseline_bw)
        else:
            eff = float("nan")

        summary[nc] = {
            "mean_agg": np.mean(agg_vals),
            "std_agg": np.std(agg_vals),
            "mean_per_node": np.mean(per_node_vals),
            "efficiency": eff,
            "mean_p99": mean_p99,
            "all_agg": agg_vals,
        }

        for r in reps:
            overlap_str = f"{r['overlap_fraction']:.3f}" if nc > 1 else "N/A"
            p99_rep = [row["lat_p99_us"] for row in per_node_by_nc[nc]
                       if row["rep"] == r["rep"]]
            p99_this = np.mean(p99_rep) if p99_rep else 0.0

            if baseline_bw and baseline_bw > 0:
                eff_this = r["aggregate_bw_mbps"] / (nc * baseline_bw)
            else:
                eff_this = float("nan")

            print(f"{nc:>5}  {r['rep']:>3}  {r['aggregate_bw_mbps']:>8.1f} MB/s  "
                  f"{r['mean_per_node_bw_mbps']:>8.1f} MB/s  {eff_this:>10.1%}  "
                  f"{p99_this:>8.0f} us  {overlap_str:>8}")
        print()

    # ── Summary table ───────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY (mean ± std across reps)")
    print("=" * 70)
    header2 = (f"{'Nodes':>5}  {'Agg BW (MB/s)':>18}  {'Per-Node (MB/s)':>18}  "
               f"{'Efficiency':>10}  {'p99 (us)':>10}")
    print(header2)
    print("-" * len(header2))
    for nc in node_counts:
        s = summary[nc]
        print(f"{nc:>5}  {s['mean_agg']:>8.1f} ± {s['std_agg']:<7.1f}  "
              f"{s['mean_per_node']:>8.1f}           {s['efficiency']:>10.1%}  "
              f"{s['mean_p99']:>8.0f}")
    print()

    # ── Per-node detail ─────────────────────────────────────────
    print("=" * 70)
    print("PER-NODE DETAIL")
    print("=" * 70)
    hdr3 = (f"{'Nodes':>5}  {'Rep':>3}  {'Host':>20}  {'BW':>10}  {'IOPS':>8}  "
            f"{'lat_mean':>10}  {'lat_p50':>10}  {'lat_p95':>10}  {'lat_p99':>10}  {'lat_max':>10}")
    print(hdr3)
    print("-" * len(hdr3))
    for row in sorted(per_node, key=lambda r: (r["nodes"], r["rep"], r["hostname"])):
        print(f"{row['nodes']:>5}  {row['rep']:>3}  {row['hostname']:>20}  "
              f"{row['bw_mbps']:>8.1f} MB  {row['iops']:>8.0f}  "
              f"{row['lat_mean_us']:>8.0f} us  {row['lat_p50_us']:>8.0f} us  "
              f"{row['lat_p95_us']:>8.0f} us  {row['lat_p99_us']:>8.0f} us  "
              f"{row['lat_max_us']:>8.0f} us")
    print()

    # ── Variance check ──────────────────────────────────────────
    print("=" * 70)
    print("VARIANCE CHECK")
    print("=" * 70)
    for nc in node_counts:
        s = summary[nc]
        cv = s["std_agg"] / s["mean_agg"] if s["mean_agg"] > 0 else 0
        flag = "  *** HIGH VARIANCE — possible background contention ***" if cv > 0.10 else ""
        print(f"  {nc} node(s): CV = {cv:.1%}  (std/mean){flag}")
    print()

    # ── Latency degradation check ───────────────────────────────
    print("=" * 70)
    print("LATENCY DEGRADATION")
    print("=" * 70)
    if len(node_counts) >= 2 and 1 in summary:
        baseline_p99 = summary[1]["mean_p99"]
        for nc in node_counts:
            s = summary[nc]
            ratio = s["mean_p99"] / baseline_p99 if baseline_p99 > 0 else 0
            print(f"  {nc} node(s): mean p99 = {s['mean_p99']:.0f} us  "
                  f"({ratio:.2f}x vs single-node)")
        max_nc = max(node_counts)
        p99_ratio = summary[max_nc]["mean_p99"] / baseline_p99 if baseline_p99 > 0 else 0
        if p99_ratio > 1.5 and summary[max_nc]["efficiency"] > args.scaling_threshold:
            print(f"\n  NOTE: p99 latency degraded {p99_ratio:.1f}x at {max_nc} nodes even")
            print(f"        though aggregate throughput scaled well. Tail latency is a")
            print(f"        separate concern from throughput scaling.")
    else:
        print("  Insufficient data for latency degradation analysis.")
    print()

    # ── Verdict ─────────────────────────────────────────────────
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)

    if baseline_bw is None:
        print("  Cannot determine verdict without a 1-node baseline.")
        print()
    else:
        max_nc = max(node_counts)
        max_eff = summary[max_nc]["efficiency"]
        max_agg = summary[max_nc]["mean_agg"]

        knee_nc = None
        for i in range(1, len(node_counts)):
            nc = node_counts[i]
            prev = node_counts[i - 1]
            eff_curr = summary[nc]["efficiency"]
            eff_prev = summary[prev]["efficiency"]
            if eff_prev > args.scaling_threshold and eff_curr <= args.scaling_threshold:
                knee_nc = nc
                break

        if max_eff >= args.scaling_threshold:
            print(f"  PER-NODE CEILING: aggregate scales near-linearly with node count;")
            print(f"  backend has headroom at this scale.")
            print(f"")
            print(f"  At {max_nc} nodes: {max_agg:.0f} MB/s aggregate, "
                  f"efficiency = {max_eff:.0%}")
            print(f"  Ideal linear:    {max_nc * baseline_bw:.0f} MB/s")
            print(f"")
            print(f"  Implication: ~{baseline_bw:.0f} MB/s is a per-node ceiling. To create")
            print(f"  cross-tenant contention, you'd need enough concurrent nodes to")
            print(f"  exhaust the backend's total capacity, which exceeds {max_nc}x{baseline_bw:.0f}.")
        elif max_eff <= args.flat_threshold:
            print(f"  SHARED BACKEND CEILING: aggregate is flat; nodes are splitting a")
            print(f"  fixed budget. Per-node bandwidth degrades as ~1/N.")
            print(f"")
            print(f"  At {max_nc} nodes: {max_agg:.0f} MB/s aggregate, "
                  f"efficiency = {max_eff:.0%}")
            print(f"  Single-node:     {baseline_bw:.0f} MB/s")
            print(f"")
            print(f"  Implication: ~{max_agg:.0f} MB/s is the backend ceiling shared across")
            print(f"  all tenants. Cross-tenant contention is the default state.")
        else:
            knee_str = f"knee at N={knee_nc}" if knee_nc else f"efficiency = {max_eff:.0%} at N={max_nc}"
            print(f"  PARTIAL SCALING: {knee_str}; aggregate ceiling ~{max_agg:.0f} MB/s.")
            print(f"")
            print(f"  At {max_nc} nodes: {max_agg:.0f} MB/s aggregate, "
                  f"efficiency = {max_eff:.0%}")
            print(f"  Single-node:     {baseline_bw:.0f} MB/s")
            print(f"  Ideal linear:    {max_nc * baseline_bw:.0f} MB/s")
            print(f"")
            print(f"  Implication: the backend has some headroom beyond a single node but")
            print(f"  saturates before linear scaling. Contention begins around the knee.")
        print()

    # ── Plot ────────────────────────────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: aggregate BW vs node count
    for nc in node_counts:
        for r in by_nc[nc]:
            ax1.scatter(nc, r["aggregate_bw_mbps"], c="steelblue", alpha=0.5,
                        s=60, zorder=3)
    means = [summary[nc]["mean_agg"] for nc in node_counts]
    ax1.plot(node_counts, means, "o-", c="steelblue", linewidth=2, markersize=8,
             label="Measured (mean)", zorder=4)

    if baseline_bw:
        ideal = [nc * baseline_bw for nc in node_counts]
        ax1.plot(node_counts, ideal, "k--", linewidth=1.5, alpha=0.6,
                 label="Ideal linear scaling")

    ax1.set_xlabel("Node count")
    ax1.set_ylabel("Aggregate read bandwidth (MB/s)")
    ax1.set_title("Multi-node NFS read bandwidth scaling")
    ax1.set_xticks(node_counts)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: p99 latency vs node count
    for nc in node_counts:
        p99_vals = [r["lat_p99_us"] for r in per_node_by_nc[nc]]
        for v in p99_vals:
            ax2.scatter(nc, v, c="coral", alpha=0.4, s=30, zorder=3)
    p99_means = [summary[nc]["mean_p99"] for nc in node_counts]
    ax2.plot(node_counts, p99_means, "o-", c="coral", linewidth=2, markersize=8,
             label="p99 latency (mean)")
    ax2.set_xlabel("Node count")
    ax2.set_ylabel("p99 latency (us)")
    ax2.set_title("Tail latency vs node count")
    ax2.set_xticks(node_counts)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(args.out_fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved to {out_path}")
    print()


if __name__ == "__main__":
    main()
