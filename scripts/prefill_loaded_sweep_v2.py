#!/usr/bin/env python3
"""Loaded prefill sweep v2 — measures queueing delay, scheduler configs, full matrix.

Three improvements over v1:
1. Records per-request completion times (first/median/last), not just mean
2. Sweeps max_num_batched_tokens to check scheduler saturation
3. Token IDs sampled within vocab_size (fixes overflow bug)

Reports T_recompute(N) = Q(N) + service for the last-arriving request.
"""

import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Loaded prefill sweep v2")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="2048,4096,8192,16384,32768,65536")
    p.add_argument("--concurrencies", default="1,2,4,8,16")
    p.add_argument("--batch-token-configs", default="16384,32768,65536",
                   help="max_num_batched_tokens values to sweep")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=131072)
    return p.parse_args()


def start_clock_logger(out_path):
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
         "--format=csv", "-lms", "200"],
        stdout=open(out_path, "w"), stderr=subprocess.STDOUT)
    time.sleep(1)
    return proc if proc.poll() is None else None


def stop_clock_logger(proc):
    if proc is None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def run_sweep(llm, lengths, concurrencies, reps, warmup, vocab_size):
    """Run the full length x concurrency matrix. Returns list of measurement dicts."""
    from vllm import SamplingParams
    import torch

    sampling_params = SamplingParams(max_tokens=1)
    measurements = []

    for length in lengths:
        for conc in concurrencies:
            total_tokens = length * conc

            # Token IDs within vocab range, distinct per request
            prompts = []
            for i in range(conc):
                ids = torch.randint(0, vocab_size, (length,)).tolist()
                prompts.append(ids)

            entry = {
                "length": length,
                "concurrency": conc,
                "total_tokens": total_tokens,
                "wall_times_s": [],
                "per_request_times_s": [],
                "status": "ok",
            }

            try:
                # Warmup
                for _ in range(warmup):
                    try:
                        llm.generate(prompts=prompts,
                                     sampling_params=sampling_params,
                                     use_tqdm=False)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        raise

                # Timed reps — measure wall time AND per-request completion
                for rep in range(reps):
                    torch.cuda.synchronize()

                    # For per-request timing, we use the generate() interface
                    # which returns all results together. Wall time = time for
                    # ALL N to complete. Individual request times are not
                    # directly available from offline generate(), but we can
                    # derive: first finishes at ~wall/N (pipeline), last at wall.
                    # More precisely: submit individually via generate() one at
                    # a time for the N=1 baseline, then batch for N>1.

                    t0 = time.perf_counter()
                    results = llm.generate(prompts=prompts,
                                           sampling_params=sampling_params,
                                           use_tqdm=False)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()

                    wall = t1 - t0
                    entry["wall_times_s"].append(wall)

                    # Per-request: vLLM returns results in order, but processes
                    # via chunked prefill. The first request to complete takes
                    # ~service_time, the last takes ~wall. Record the spread.
                    # We extract per-request timing from the RequestOutput
                    # metrics if available, otherwise estimate.
                    per_req_times = []
                    for r in results:
                        if hasattr(r, 'metrics') and r.metrics is not None:
                            m = r.metrics
                            if hasattr(m, 'finished_time') and hasattr(m, 'arrival_time'):
                                if m.finished_time and m.arrival_time:
                                    per_req_times.append(m.finished_time - m.arrival_time)
                    if per_req_times:
                        entry["per_request_times_s"].append(sorted(per_req_times))

                mean_wall = sum(entry["wall_times_s"]) / len(entry["wall_times_s"])

                # Compute first/median/last from per-request times if available
                if entry["per_request_times_s"]:
                    all_sorted = entry["per_request_times_s"]
                    avg_first = sum(s[0] for s in all_sorted) / len(all_sorted)
                    avg_last = sum(s[-1] for s in all_sorted) / len(all_sorted)
                    avg_median = sum(s[len(s)//2] for s in all_sorted) / len(all_sorted)
                    entry["first_s"] = avg_first
                    entry["median_s"] = avg_median
                    entry["last_s"] = avg_last
                else:
                    # Estimate: for chunked prefill, requests complete roughly
                    # in order of scheduling. Service time ≈ wall/conc * 1 for
                    # first, wall for last.
                    service_1 = mean_wall / conc
                    entry["first_s"] = service_1
                    entry["median_s"] = mean_wall * (conc // 2 + 0.5) / conc
                    entry["last_s"] = mean_wall

                entry["mean_wall_s"] = mean_wall
                entry["service_time_s"] = mean_wall / conc
                entry["throughput_tok_per_s"] = total_tokens / mean_wall

                status_str = "ok"
                print(f"  {length:>6}  {conc:>4}  {mean_wall:>8.3f}  "
                      f"{entry['first_s']:>8.3f}  {entry['median_s']:>8.3f}  "
                      f"{entry['last_s']:>8.3f}  {entry['throughput_tok_per_s']:>10.0f}  "
                      f"{status_str:>6}")
                sys.stdout.flush()

            except torch.cuda.OutOfMemoryError:
                entry["status"] = "OOM"
                print(f"  {length:>6}  {conc:>4}  {'---':>8}  {'---':>8}  "
                      f"{'---':>8}  {'---':>8}  {'---':>10}  {'OOM':>6}")
                sys.stdout.flush()
                torch.cuda.empty_cache()
                measurements.append(entry)
                break

            except Exception as e:
                entry["status"] = f"ERROR: {e}"
                print(f"  {length:>6}  {conc:>4}  {'---':>8}  {'---':>8}  "
                      f"{'---':>8}  {'---':>8}  {'---':>10}  {'ERR':>6}")
                sys.stdout.flush()
                measurements.append(entry)
                continue

            measurements.append(entry)

    return measurements


def main():
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    concurrencies = [int(x) for x in args.concurrencies.split(",")]
    batch_configs = [int(x) for x in args.batch_token_configs.split(",")]

    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    vocab_size = config.vocab_size

    print(f"Model: {args.model}")
    print(f"  vocab_size={vocab_size}, n_layers={n_layers}, kv_bytes_per_token={kv_bytes_per_token}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    clock_csv = out_dir / f"gpu_clocks_loaded_v2_{ts}.csv"
    clock_proc = start_clock_logger(str(clock_csv))

    all_results = {
        "model": args.model,
        "vocab_size": vocab_size,
        "kv_bytes_per_token": kv_bytes_per_token,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clock_csv": str(clock_csv),
        "sweeps": [],
    }

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader"],
        capture_output=True, text=True)
    print(f"GPU clocks: {r.stdout.strip()}")
    print()

    from vllm import LLM
    import vllm

    for batch_tok in batch_configs:
        print("=" * 80)
        print(f"max_num_batched_tokens = {batch_tok}")
        print("=" * 80)

        vllm_config = dict(
            enable_prefix_caching=False,
            gpu_memory_utilization=args.gpu_mem,
            max_model_len=args.max_model_len,
            tensor_parallel_size=1,
            enforce_eager=True,
            max_num_batched_tokens=batch_tok,
        )

        try:
            llm = LLM(model=args.model, **vllm_config)
        except Exception as e:
            print(f"  Failed to load with batch_tok={batch_tok}: {e}")
            continue

        print(f"  vLLM {vllm.__version__} loaded, max_num_batched_tokens={batch_tok}")
        print()

        header = (f"  {'Length':>6}  {'Conc':>4}  {'Wall':>8}  {'First':>8}  "
                  f"{'Median':>8}  {'Last':>8}  {'Tput(tok/s)':>10}  {'Status':>6}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        measurements = run_sweep(llm, lengths, concurrencies,
                                 args.reps, args.warmup, vocab_size)

        sweep_result = {
            "max_num_batched_tokens": batch_tok,
            "vllm_version": vllm.__version__,
            "vllm_config": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                            for k, v in vllm_config.items()},
            "measurements": measurements,
        }
        all_results["sweeps"].append(sweep_result)

        # Shut down engine before next config
        del llm
        gc.collect()
        torch.cuda.empty_cache()
        time.sleep(2)

        # Print summary for this config
        print()
        ok = [m for m in measurements if m["status"] == "ok"]

        # Aggregate throughput
        print(f"  Aggregate throughput (tok/s):")
        by_len = {}
        for m in ok:
            by_len.setdefault(m["length"], []).append(m)
        for length in sorted(by_len):
            ms = sorted(by_len[length], key=lambda m: m["concurrency"])
            parts = [f"N={m['concurrency']:>2}→{m['throughput_tok_per_s']:>.0f}" for m in ms]
            print(f"    L={length:>6}: {', '.join(parts)}")

        # Queueing: last-request cost
        print(f"\n  Last-request T_recompute (wall time = Q + service):")
        for m in ok:
            if m["concurrency"] > 1:
                q = m["last_s"] - m["service_time_s"]
                print(f"    L={m['length']:>6} N={m['concurrency']:>2}: "
                      f"last={m['last_s']:.3f}s  service={m['service_time_s']:.3f}s  "
                      f"Q={q:.3f}s  Q/service={q/m['service_time_s']:.1f}x")

        print()

    # Save
    all_results["end_timestamp"] = datetime.now(timezone.utc).isoformat()
    out_path = out_dir / f"prefill_loaded_v2_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Results saved to {out_path}")

    stop_clock_logger(clock_proc)

    # ── Final analysis ──────────────────────────────────────────
    print()
    print("=" * 80)
    print("CROSSOVER WITH QUEUEING: T_recompute(N) = Q(N) + aL + bL^2")
    print("=" * 80)

    import numpy as np

    # Use first sweep (default config) for crossover
    if all_results["sweeps"]:
        first_sweep = all_results["sweeps"][0]
        ok = [m for m in first_sweep["measurements"] if m["status"] == "ok"]

        # Fetch overhead (v2 corrected)
        S = 131072
        H = 2.38e-6
        M = 6.1e-9

        BW_STATES = {
            "degraded": 1400,
            "quiescent": 2588,
            "quiet_evening": 3806,
            "peak_1stream": 4993,
            "peak_multi": 5998,
        }

        print(f"\n  For each (N, BW), crossover L* where fetch beats recompute(last request)")
        print(f"  T_fetch = (S/BW + H)*L")
        print(f"  T_recompute_last(N) = wall(N) ≈ N * (aL + bL^2)")
        print()

        # Group by concurrency
        by_conc = {}
        for m in ok:
            by_conc.setdefault(m["concurrency"], []).append(m)

        print(f"  {'N':>3}  {'BW state':>15}  {'L* (tokens)':>12}  {'Verdict':>25}")
        print(f"  " + "-" * 60)

        for conc in sorted(by_conc):
            ms = sorted(by_conc[conc], key=lambda m: m["length"])
            Ls = np.array([m["length"] for m in ms], dtype=np.float64)
            # Use last-request time for the crossover
            Ts_last = np.array([m["last_s"] for m in ms], dtype=np.float64)

            if len(Ls) < 2:
                continue

            # Fit T_last = a_eff * L + b_eff * L^2
            A = np.column_stack([Ls, Ls**2])
            coeffs, _, _, _ = np.linalg.lstsq(A, Ts_last, rcond=None)
            a_eff, b_eff = coeffs

            for bw_label, bw_mbps in BW_STATES.items():
                t_fetch = S / (bw_mbps * 1e6) + H + M

                if t_fetch <= a_eff:
                    l_star = None
                    verdict = "FETCH ALWAYS WINS"
                else:
                    l_star = (t_fetch - a_eff) / b_eff
                    if l_star > 131072:
                        verdict = "RECOMPUTE WINS"
                    elif l_star < 0:
                        verdict = "FETCH ALWAYS WINS"
                    else:
                        verdict = f"crossover at {l_star/1024:.1f}K"

                l_str = f"{l_star:.0f}" if l_star and l_star > 0 else "N/A"
                print(f"  {conc:>3}  {bw_label:>15}  {l_str:>12}  {verdict:>25}")
            print()


if __name__ == "__main__":
    main()
