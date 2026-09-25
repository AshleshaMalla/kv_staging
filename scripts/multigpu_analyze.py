#!/usr/bin/env python3
"""Analyze multi-GPU prefill scaling results.

Reads per-GPU JSON outputs from the multigpu_scaling run, computes aggregate
throughput, scaling efficiency, per-GPU clock distributions, and the node-level
fetch-vs-recompute crossover.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Analyze multi-GPU scaling results")
    p.add_argument("--data-dir", required=True, help="Directory with gpu*_n*.json files")
    return p.parse_args()


def parse_clock_csv(csv_path):
    """Parse nvidia-smi CSV into per-GPU clock/power distributions."""
    per_gpu = {}
    try:
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("timestamp"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 5:
                    continue
                try:
                    gpu_idx = int(parts[1])
                    sm_mhz = int(parts[2].replace(" MHz", ""))
                    power_w = float(parts[4].replace(" W", ""))
                    per_gpu.setdefault(gpu_idx, {"clocks": [], "power": []})
                    per_gpu[gpu_idx]["clocks"].append(sm_mhz)
                    per_gpu[gpu_idx]["power"].append(power_w)
                except (ValueError, IndexError):
                    continue
    except Exception:
        pass
    return per_gpu


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    import yaml
    constants_path = Path(__file__).resolve().parent.parent / "config" / "measured_constants.yaml"
    with open(constants_path) as f:
        consts = yaml.safe_load(f)

    S = consts["model"]["kv_bytes_per_token"]["value"]
    H_s = consts["fetch_overhead"]["v2_h_to_gpu_us_per_token"]["value"] * 1e-6
    M_s = consts["fetch_overhead"]["metadata_amortized_s_per_token"]["value"]
    a = consts["prefill"]["coeff_a_vllm_s_per_token"]["value"]
    b = consts["prefill"]["coeff_b_vllm_s_per_token2"]["value"]

    BW_STATES = {
        "degraded": consts["bandwidth"]["hammerspace_degraded_mbps"]["value"],
        "quiescent": consts["bandwidth"]["hammerspace_quiescent_mbps"]["value"],
        "quiet_evening": consts["bandwidth"]["hammerspace_quiet_evening_mbps"]["value"],
        "peak_1stream": consts["bandwidth"]["hammerspace_peak_1stream_mbps"]["value"],
    }

    # Load all GPU result files
    runs = {}  # (n_gpus, gpu_id) -> data
    for f in sorted(data_dir.glob("gpu*_n*.json")):
        name = f.stem  # e.g. gpu0_n2
        parts = name.split("_")
        gpu_id = int(parts[0].replace("gpu", ""))
        n_gpus = int(parts[1].replace("n", ""))
        with open(f) as fh:
            data = json.load(fh)
        runs[(n_gpus, gpu_id)] = data

    if not runs:
        print("ERROR: no result files found")
        sys.exit(1)

    # Gather all gpu counts and lengths
    all_n_gpus = sorted(set(k[0] for k in runs))
    all_lengths = set()
    for data in runs.values():
        for m in data.get("measurements", []):
            all_lengths.add(m["length"])
    all_lengths = sorted(all_lengths)

    # Collect single-GPU baselines
    single_gpu = {}  # length -> mean_wall_s
    for (ng, gid), data in runs.items():
        if ng == 1:
            for m in data["measurements"]:
                single_gpu[m["length"]] = m["mean_wall_s"]

    print("=" * 80)
    print("MULTI-GPU PREFILL SCALING ANALYSIS")
    print("=" * 80)

    # Per-GPU results
    print(f"\n{'nGPU':>4}  {'GPU':>3}  {'Length':>6}  {'Mean(s)':>8}  "
          f"{'Stdev(s)':>8}  {'Tput(tok/s)':>12}")
    print("-" * 55)
    for (ng, gid) in sorted(runs):
        for m in runs[(ng, gid)].get("measurements", []):
            print(f"{ng:>4}  {gid:>3}  {m['length']:>6}  "
                  f"{m['mean_wall_s']:>8.4f}  {m.get('stdev_wall_s', 0):>8.4f}  "
                  f"{m['throughput_tok_per_s']:>12.0f}")

    # Aggregate throughput and scaling efficiency
    print()
    print("=" * 80)
    print("AGGREGATE THROUGHPUT AND SCALING EFFICIENCY")
    print("=" * 80)

    print(f"\n{'nGPU':>4}  {'Length':>6}  {'Agg tput':>12}  {'1-GPU tput':>12}  "
          f"{'Ideal':>12}  {'Efficiency':>10}  {'Per-GPU svc(s)':>14}")
    print("-" * 80)

    scaling = {}  # (n_gpus, length) -> dict
    for ng in all_n_gpus:
        for L in all_lengths:
            tputs = []
            walls = []
            for gid in range(ng):
                data = runs.get((ng, gid))
                if data is None:
                    continue
                for m in data["measurements"]:
                    if m["length"] == L:
                        tputs.append(m["throughput_tok_per_s"])
                        walls.append(m["mean_wall_s"])
            if not tputs:
                continue

            agg_tput = sum(tputs)
            single = single_gpu.get(L)
            if single is None:
                continue
            single_tput = L / single
            ideal = ng * single_tput
            eff = agg_tput / ideal if ideal > 0 else 0
            per_gpu_mean_wall = sum(walls) / len(walls)

            print(f"{ng:>4}  {L:>6}  {agg_tput:>12.0f}  {single_tput:>12.0f}  "
                  f"{ideal:>12.0f}  {eff:>9.1%}  {per_gpu_mean_wall:>14.4f}")

            scaling[(ng, L)] = {
                "n_gpus": ng,
                "length": L,
                "aggregate_tput": agg_tput,
                "single_gpu_tput": single_tput,
                "ideal_tput": ideal,
                "efficiency": round(eff, 4),
                "per_gpu_mean_wall_s": per_gpu_mean_wall,
                "per_gpu_walls": walls,
                "per_gpu_tputs": tputs,
            }

    # Clock distributions
    print()
    print("=" * 80)
    print("PER-GPU CLOCK AND POWER DISTRIBUTIONS")
    print("=" * 80)

    clock_csv = data_dir / "gpu_clocks.csv"
    clock_data = parse_clock_csv(str(clock_csv))
    clock_summary = {}
    for gpu_id in sorted(clock_data):
        clocks = clock_data[gpu_id]["clocks"]
        power = clock_data[gpu_id]["power"]
        c = Counter(clocks)
        total = len(clocks)
        dist_str = ", ".join(f"{mhz}MHz:{count/total:.0%}"
                             for mhz, count in c.most_common(5))
        mean_power = sum(power) / len(power) if power else 0
        min_c = min(clocks) if clocks else 0
        max_c = max(clocks) if clocks else 0
        print(f"  GPU {gpu_id}: {dist_str}  power={mean_power:.1f}W  "
              f"range=[{min_c}, {max_c}] MHz")
        clock_summary[gpu_id] = {
            "n_samples": total,
            "clock_dist": dict(c),
            "mean_power_w": round(mean_power, 1),
            "min_mhz": min_c,
            "max_mhz": max_c,
        }

    # Node-level fetch vs recompute
    print()
    print("=" * 80)
    print("NODE-LEVEL: HOW MANY GPUs BEFORE AGGREGATE RECOMPUTE > FETCH?")
    print("=" * 80)
    print()
    print("  Fetch model (per GPU when G GPUs fetch simultaneously):")
    print(f"    T_fetch_per_tok(G) = G * S/BW_node + H + M")
    print(f"    S={S} bytes/tok, H={H_s*1e6:.2f} us/tok, M={M_s*1e6:.3f} us/tok")
    print(f"    Storage BW is a per-node ceiling shared by G GPUs")
    print(f"    PCIe H2D is per-GPU (each GPU has its own link)")
    print()
    print("  Recompute: measured per-GPU service time from this experiment")
    print()

    node_crossover = {}
    for L in all_lengths:
        print(f"  L = {L}:")
        for bw_label, bw_mbps in BW_STATES.items():
            bw_bytes = bw_mbps * 1e6
            results_row = []
            for ng in all_n_gpus:
                sr = scaling.get((ng, L))
                if sr is None:
                    continue

                agg_recompute = sr["aggregate_tput"]

                # Per-GPU fetch: G GPUs share node BW, each has own PCIe
                t_fetch_per_tok = (ng * S / bw_bytes) + H_s + M_s
                per_gpu_fetch_tput = L / (t_fetch_per_tok * L)
                agg_fetch = ng * per_gpu_fetch_tput

                ratio = agg_recompute / agg_fetch if agg_fetch > 0 else float('inf')
                winner = "RECOMPUTE" if ratio > 1 else "FETCH"

                results_row.append({
                    "n_gpus": ng,
                    "agg_recompute": agg_recompute,
                    "agg_fetch": agg_fetch,
                    "ratio": round(ratio, 4),
                    "winner": winner,
                    "t_fetch_per_tok_us": t_fetch_per_tok * 1e6,
                })

                print(f"    {bw_label:>15} G={ng}: "
                      f"recomp={agg_recompute:>10.0f}  fetch={agg_fetch:>10.0f}  "
                      f"ratio={ratio:.3f}  {winner}")

            node_crossover.setdefault(L, {})[bw_label] = results_row
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for L in all_lengths:
        print(f"\n  L = {L}:")
        for bw_label in BW_STATES:
            rows = node_crossover.get(L, {}).get(bw_label, [])
            first_win = None
            for r in rows:
                if r["ratio"] > 1:
                    first_win = r["n_gpus"]
                    break
            if first_win:
                print(f"    {bw_label:>15}: recompute wins at G >= {first_win}")
            else:
                print(f"    {bw_label:>15}: fetch wins at all measured G (up to {all_n_gpus[-1]})")

    # Save combined results
    combined = {
        "data_dir": str(data_dir),
        "constants": {
            "S": S, "H_us": H_s * 1e6, "M_us": M_s * 1e6,
            "a": a, "b": b,
        },
        "bw_states": BW_STATES,
        "scaling": {f"{k[0]}gpu_L{k[1]}": v for k, v in scaling.items()},
        "clock_summary": clock_summary,
        "node_crossover": {str(k): v for k, v in node_crossover.items()},
    }

    out_path = data_dir / "analysis.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)
    print(f"\nAnalysis saved to {out_path}")


if __name__ == "__main__":
    main()
