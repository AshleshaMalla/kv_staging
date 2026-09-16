#!/usr/bin/env python3
"""Measure prefill latency under concurrent load via vLLM offline LLM.

Submits N identical-length prompts in a single generate() call so vLLM
batches them and they compete for GPU compute simultaneously. This measures
how prefill cost scales with concurrency — the loaded prefill scaling law.

For each (length, concurrency) pair:
  - Per-request latency = total time / concurrency (all complete together)
  - Aggregate throughput = total tokens prefilled / wall time
  - SM clock and power sampled throughout

Uses vLLM's offline LLM class with max_tokens=1 (prefill only, no decode).
This is the correct mechanism because:
  1. vLLM's scheduler batches the prompts into a single forward pass
  2. The GPU sees exactly the concurrent-prefill workload we want to measure
  3. generate() returns only after all N prefills complete, giving wall time

Usage:
  python3 scripts/prefill_loaded_sweep.py --model /path/to/model
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Loaded prefill sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="2048,4096,8192,16384,32768,65536",
                   help="Context lengths (131072 likely OOMs at concurrency>1)")
    p.add_argument("--concurrencies", default="1,2,4,8,16")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=131072)
    return p.parse_args()


def start_clock_logger(out_path):
    """Log SM clock, mem clock, power, utilization at 100ms intervals."""
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
         "--format=csv", "-lms", "100"],
        stdout=open(out_path, "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(1)
    if proc.poll() is not None:
        print(f"WARNING: nvidia-smi logger died", file=sys.stderr)
        return None
    return proc


def stop_clock_logger(proc):
    if proc is None:
        return
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()


def main():
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    concurrencies = [int(x) for x in args.concurrencies.split(",")]

    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    import torch

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    kv_bytes_per_token = 2 * n_layers * n_kv_heads * head_dim * 2
    max_ctx = getattr(config, "max_position_embeddings", None)

    print(f"Model: {args.model}")
    print(f"  n_layers={n_layers}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  kv_bytes_per_token={kv_bytes_per_token}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    clock_csv = out_dir / f"gpu_clocks_loaded_{ts}.csv"
    clock_proc = start_clock_logger(str(clock_csv))

    vllm_config = dict(
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
    )

    print(f"Loading vLLM: {vllm_config}")
    llm = LLM(model=args.model, **vllm_config)
    import vllm
    print(f"vLLM {vllm.__version__} loaded")
    print()

    sampling_params = SamplingParams(max_tokens=1)

    results = {
        "model": args.model,
        "engine": "vllm",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "vllm_config": {k: str(v) if not isinstance(v, (int, float, bool)) else v
                        for k, v in vllm_config.items()},
        "kv_bytes_per_token": kv_bytes_per_token,
        "reps": args.reps,
        "warmup": args.warmup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "clock_csv": str(clock_csv),
        "measurements": [],
    }

    # Confirm clock before starting
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader"],
        capture_output=True, text=True)
    print(f"GPU clocks (idle): {r.stdout.strip()}")

    print()
    header = (f"{'Length':>8}  {'Conc':>4}  {'Wall(s)':>10}  {'Per-req(s)':>10}  "
              f"{'Tput(tok/s)':>12}  {'Status':>8}")
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    for length in lengths:
        for conc in concurrencies:
            total_tokens = length * conc

            # Build N distinct prompts of the same length
            # Use different starting token IDs so vLLM doesn't deduplicate
            prompts = []
            for i in range(conc):
                start = 10 + i * length
                prompts.append(list(range(start, start + length)))

            entry = {
                "length": length,
                "concurrency": conc,
                "total_tokens": total_tokens,
                "wall_times_s": [],
                "status": "ok",
            }

            try:
                # Warmup
                for _ in range(args.warmup):
                    try:
                        llm.generate(prompts=prompts, sampling_params=sampling_params,
                                     use_tqdm=False)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        raise

                # Timed reps
                for rep in range(args.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    llm.generate(prompts=prompts, sampling_params=sampling_params,
                                 use_tqdm=False)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    entry["wall_times_s"].append(t1 - t0)

                mean_wall = sum(entry["wall_times_s"]) / len(entry["wall_times_s"])
                per_req = mean_wall / conc
                throughput = total_tokens / mean_wall

                entry["mean_wall_s"] = mean_wall
                entry["per_request_s"] = per_req
                entry["throughput_tok_per_s"] = throughput

                print(f"{length:>8}  {conc:>4}  {mean_wall:>10.4f}  {per_req:>10.4f}  "
                      f"{throughput:>12.0f}  {'ok':>8}")
                sys.stdout.flush()

            except torch.cuda.OutOfMemoryError:
                entry["status"] = "OOM"
                print(f"{length:>8}  {conc:>4}  {'---':>10}  {'---':>10}  "
                      f"{'---':>12}  {'OOM':>8}")
                sys.stdout.flush()
                torch.cuda.empty_cache()
                results["measurements"].append(entry)
                break  # skip higher concurrencies at this length

            except Exception as e:
                entry["status"] = f"ERROR: {e}"
                print(f"{length:>8}  {conc:>4}  {'---':>10}  {'---':>10}  "
                      f"{'---':>12}  {'ERR':>8}: {e}")
                sys.stdout.flush()
                results["measurements"].append(entry)
                continue

            results["measurements"].append(entry)

    # Confirm clock after
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=clocks.sm,clocks.max.sm", "--format=csv,noheader"],
        capture_output=True, text=True)
    print(f"\nGPU clocks (post): {r.stdout.strip()}")

    results["end_timestamp"] = datetime.now(timezone.utc).isoformat()

    out_path = out_dir / f"prefill_loaded_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")

    stop_clock_logger(clock_proc)

    # ── Inline analysis ─────────────────────────────────────────
    print()
    print("=" * 70)
    print("PER-CONCURRENCY PREFILL FIT: T = a*L + b*L^2")
    print("=" * 70)

    import numpy as np

    ok = [m for m in results["measurements"] if m["status"] == "ok"]
    by_conc = {}
    for m in ok:
        by_conc.setdefault(m["concurrency"], []).append(m)

    fits = {}
    for conc in sorted(by_conc):
        ms = sorted(by_conc[conc], key=lambda m: m["length"])
        Ls = np.array([m["length"] for m in ms], dtype=np.float64)
        Ts = np.array([m["per_request_s"] for m in ms], dtype=np.float64)
        if len(Ls) < 2:
            continue
        A = np.column_stack([Ls, Ls**2])
        coeffs, _, _, _ = np.linalg.lstsq(A, Ts, rcond=None)
        a_fit, b_fit = coeffs
        ss_res = np.sum((Ts - A @ coeffs)**2)
        ss_tot = np.sum((Ts - np.mean(Ts))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        fits[conc] = (a_fit, b_fit, r2)
        print(f"  Concurrency {conc:>2}: a = {a_fit:.4e}  b = {b_fit:.4e}  R² = {r2:.6f}")

    if 1 in fits:
        a1, b1, _ = fits[1]
        print()
        print("  Scaling vs isolated (concurrency=1):")
        for conc in sorted(fits):
            ac, bc, _ = fits[conc]
            print(f"    Conc {conc:>2}: a = {ac/a1:.2f}x  b = {bc/b1:.2f}x")

    print()
    print("=" * 70)
    print("AGGREGATE THROUGHPUT vs CONCURRENCY (at each length)")
    print("=" * 70)
    by_len = {}
    for m in ok:
        by_len.setdefault(m["length"], []).append(m)
    for length in sorted(by_len):
        ms = sorted(by_len[length], key=lambda m: m["concurrency"])
        line = f"  L={length:>6}: "
        for m in ms:
            line += f"  conc={m['concurrency']:>2} → {m['throughput_tok_per_s']:>8.0f} tok/s"
        print(line)


if __name__ == "__main__":
    main()
