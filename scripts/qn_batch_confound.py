#!/usr/bin/env python3
"""Controlled comparison: Q(N) slope vs max_num_batched_tokens.

The extended sweep found slope ~1.0 at max_num_batched_tokens=131072,
while job 155248 found 0.85 at 16384. Two confounds changed: the batching
parameter and the node. This script sweeps the batching parameter on a
SINGLE node to isolate the effect.

Outer loop: max_num_batched_tokens = [8192, 16384, 32768, 65536, 131072]
Inner loop: lengths x concurrencies
Everything else fixed: same node, model, vLLM version, warmup/rep, enforce_eager.

Reports the fitted slope for each (batch_budget, length) pair.
"""

import argparse
import gc
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Q(N) slope vs max_num_batched_tokens")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="16384,32768")
    p.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    p.add_argument("--batch-budgets", default="8192,16384,32768,65536,131072")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=131072)
    return p.parse_args()


def start_clock_logger(out_path, interval_ms=100):
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
         "--format=csv", f"-lms", str(interval_ms)],
        stdout=open(out_path, "w"), stderr=subprocess.STDOUT)
    time.sleep(0.5)
    return proc if proc.poll() is None else None


def stop_clock_logger(proc):
    if proc is None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def clock_distribution(csv_path):
    try:
        clocks = []
        with open(csv_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("timestamp"):
                    continue
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        sm = int(parts[2].replace(" MHz", ""))
                        clocks.append(sm)
                    except (ValueError, IndexError):
                        continue
        if not clocks:
            return "no data"
        from collections import Counter
        c = Counter(clocks)
        total = len(clocks)
        dist = ", ".join(f"{mhz}MHz:{count/total:.0%}" for mhz, count in c.most_common(5))
        return f"n={total}, {dist}"
    except Exception as e:
        return f"error: {e}"


def fit_slope(measurements, n1_wall):
    """Fit ratio = 1 + slope*(N-1) from measurements at a single (budget, length)."""
    ns, ratios = [], []
    for m in measurements:
        if m["status"] != "ok" or m["concurrency"] < 2:
            continue
        ratio = m["mean_wall_s"] / n1_wall
        ns.append(m["concurrency"])
        ratios.append(ratio)
    if len(ns) < 2:
        return None, None, None
    ns = np.array(ns, dtype=np.float64)
    ratios = np.array(ratios, dtype=np.float64)
    nm1 = ns - 1
    deltas = ratios - 1
    slope = float(np.sum(nm1 * deltas) / np.sum(nm1 ** 2))
    pred = 1 + slope * nm1
    rmse = float(np.sqrt(np.mean((ratios - pred) ** 2)))
    return slope, rmse, int(max(ns))


def run_one_budget(model, budget, lengths, concurrencies, reps, warmup,
                   gpu_mem, max_model_len):
    """Run the full length x N matrix for one max_num_batched_tokens value."""
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    import vllm

    config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    vocab_size = config.vocab_size

    print(f"\n  Loading vLLM with max_num_batched_tokens={budget}...")
    llm = LLM(
        model=model,
        enable_prefix_caching=False,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=budget,
    )
    print(f"  vLLM {vllm.__version__} loaded.")

    sampling_params = SamplingParams(max_tokens=1)
    measurements = []
    n1_service = {}

    print(f"  {'Length':>6}  {'N':>4}  {'Wall(s)':>8}  {'Svc1(s)':>8}  "
          f"{'Ratio':>6}  {'Slope':>6}  {'Tput':>10}  {'Status':>6}")

    for length in lengths:
        for conc in concurrencies:
            prompts = []
            for i in range(conc):
                ids = torch.randint(0, vocab_size, (length,)).tolist()
                prompts.append(ids)

            entry = {
                "length": length,
                "concurrency": conc,
                "total_tokens": length * conc,
                "wall_times_s": [],
                "status": "ok",
                "max_num_batched_tokens": budget,
            }

            try:
                for _ in range(warmup):
                    try:
                        llm.generate(prompts=prompts,
                                     sampling_params=sampling_params,
                                     use_tqdm=False)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        raise

                for rep in range(reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    llm.generate(prompts=prompts,
                                 sampling_params=sampling_params,
                                 use_tqdm=False)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    entry["wall_times_s"].append(t1 - t0)

                mean_wall = sum(entry["wall_times_s"]) / len(entry["wall_times_s"])
                entry["mean_wall_s"] = mean_wall

                if conc == 1:
                    n1_service[length] = mean_wall

                svc1 = n1_service.get(length, mean_wall)
                ratio = mean_wall / svc1
                slope = (ratio - 1) / (conc - 1) if conc > 1 else 0.0

                print(f"  {length:>6}  {conc:>4}  {mean_wall:>8.3f}  {svc1:>8.3f}  "
                      f"{ratio:>6.2f}  {slope:>6.3f}  "
                      f"{entry['total_tokens']/mean_wall:>10.0f}  {'ok':>6}")
                sys.stdout.flush()

            except torch.cuda.OutOfMemoryError:
                entry["status"] = "OOM"
                print(f"  {length:>6}  {conc:>4}  {'---':>8}  {'---':>8}  "
                      f"{'---':>6}  {'---':>6}  {'---':>10}  {'OOM':>6}")
                sys.stdout.flush()
                torch.cuda.empty_cache()
                measurements.append(entry)
                break

            except Exception as e:
                entry["status"] = f"ERROR: {e}"
                print(f"  {length:>6}  {conc:>4}  {'---':>8}  {'---':>8}  "
                      f"{'---':>6}  {'---':>6}  {'---':>10}  {'ERR':>6}")
                sys.stdout.flush()
                measurements.append(entry)
                continue

            measurements.append(entry)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    time.sleep(2)

    return measurements, n1_service


def main():
    args = parse_args()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"qn_batch_confound_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    lengths = [int(x) for x in args.lengths.split(",")]
    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    budgets = [int(x) for x in args.batch_budgets.split(",")]

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    gpu_info = r.stdout.strip()
    hostname = subprocess.run(["hostname"], capture_output=True, text=True).stdout.strip()

    print("=" * 80)
    print(f"Q(N) SLOPE vs max_num_batched_tokens — CONTROLLED COMPARISON")
    print(f"Timestamp: {ts}")
    print(f"Node: {hostname}")
    print(f"GPU: {gpu_info}")
    print("=" * 80)
    print(f"Lengths: {lengths}")
    print(f"Concurrencies: {concurrencies}")
    print(f"Batch budgets: {budgets}")
    print()

    clock_csv = out_dir / f"gpu_clocks_{ts}.csv"
    clock_proc = start_clock_logger(str(clock_csv))

    all_results = {
        "timestamp": ts,
        "hostname": hostname,
        "gpu_info": gpu_info,
        "model": args.model,
        "lengths": lengths,
        "concurrencies": concurrencies,
        "batch_budgets": budgets,
        "sweeps": [],
    }

    slope_table = {}

    for budget in budgets:
        print("=" * 80)
        print(f"max_num_batched_tokens = {budget}")
        print("=" * 80)

        measurements, n1_service = run_one_budget(
            args.model, budget, lengths, concurrencies,
            args.reps, args.warmup, args.gpu_mem, args.max_model_len)

        sweep_entry = {
            "max_num_batched_tokens": budget,
            "measurements": measurements,
            "n1_service": {str(k): v for k, v in n1_service.items()},
            "slopes": {},
        }

        for length in lengths:
            length_ms = [m for m in measurements if m["length"] == length]
            n1 = n1_service.get(length)
            if n1 is None:
                continue
            slope, rmse, max_n = fit_slope(length_ms, n1)
            if slope is not None:
                sweep_entry["slopes"][str(length)] = {
                    "slope": slope, "rmse": rmse, "max_N": max_n
                }
                slope_table[(budget, length)] = slope
                print(f"  → L={length}: slope={slope:.4f}, RMSE={rmse:.4f}, max_N={max_n}")

        all_results["sweeps"].append(sweep_entry)
        print()

    stop_clock_logger(clock_proc)
    clk_dist = clock_distribution(str(clock_csv))

    # Summary table
    print()
    print("=" * 80)
    print("SLOPE TABLE: slope(max_num_batched_tokens, length)")
    print("=" * 80)
    print(f"  Node: {hostname}")
    print(f"  SM clocks: {clk_dist}")
    print()
    header = f"  {'Budget':>8s}" + "".join(f"  {'L='+str(l):>10s}" for l in lengths)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for budget in budgets:
        row = f"  {budget:>8d}"
        for length in lengths:
            s = slope_table.get((budget, length))
            if s is not None:
                row += f"  {s:>10.4f}"
            else:
                row += f"  {'---':>10s}"
        print(row)

    print()
    print("  Job 155248 reference: slope = 0.85 at budget=16384")
    print("  Extended sweep:       slope ≈ 1.0 at budget=131072")
    print()

    # Check monotonicity
    for length in lengths:
        slopes_for_len = [(b, slope_table.get((b, length))) for b in budgets
                          if slope_table.get((b, length)) is not None]
        if len(slopes_for_len) >= 2:
            vals = [s for _, s in slopes_for_len]
            monotonic_up = all(vals[i] <= vals[i+1] + 0.001 for i in range(len(vals)-1))
            monotonic_down = all(vals[i] >= vals[i+1] - 0.001 for i in range(len(vals)-1))
            direction = "MONOTONIC UP" if monotonic_up else ("MONOTONIC DOWN" if monotonic_down else "NON-MONOTONIC")
            print(f"  L={length}: {direction} ({', '.join(f'{s:.3f}' for s in vals)})")

    all_results["clock_distribution"] = clk_dist
    all_results["slope_table"] = {f"{b},{l}": s for (b, l), s in slope_table.items()}

    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
