#!/usr/bin/env python3
"""Multi-GPU prefill scaling — measure aggregate recompute throughput at 1-4 GPUs.

Each GPU runs its own independent vLLM process (CUDA_VISIBLE_DEVICES pinned,
tensor_parallel_size=1, no inter-GPU communication). All processes run
simultaneously and measure the same lengths with the same settings as the
authoritative vLLM sweep (enforce_eager, prefix caching off, max_tokens=1,
warmup + 5 reps).

Logs per-GPU SM clock and power at 100ms throughout. Reports:
  - Per-GPU service times at 1-4 GPUs active
  - Aggregate throughput (tokens/s summed across GPUs)
  - Scaling efficiency relative to 4 * single-GPU throughput
  - Per-GPU clock distributions (checking for power/thermal throttling)

Combined with the measured per-node NFS storage ceiling (flat 4.4-4.8 GB/s
across 1-16 streams), computes how many GPUs must be busy before aggregate
recompute capacity exceeds aggregate fetch capacity at each bandwidth state.
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
    p = argparse.ArgumentParser(description="Multi-GPU prefill scaling measurement")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="16384,32768",
                   help="Context lengths to measure")
    p.add_argument("--gpu-counts", default="1,2,3,4",
                   help="Number of GPUs to run simultaneously")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--max-num-batched-tokens", type=int, default=65536)
    return p.parse_args()


WORKER_SCRIPT = '''
import gc
import json
import os
import sys
import time

gpu_id = int(os.environ["CUDA_VISIBLE_DEVICES"])
model = sys.argv[1]
lengths = [int(x) for x in sys.argv[2].split(",")]
reps = int(sys.argv[3])
warmup = int(sys.argv[4])
gpu_mem = float(sys.argv[5])
max_model_len = int(sys.argv[6])
max_num_batched_tokens = int(sys.argv[7])
out_path = sys.argv[8]

import torch
from transformers import AutoConfig
from vllm import LLM, SamplingParams
import vllm

config = AutoConfig.from_pretrained(model, trust_remote_code=True)
vocab_size = config.vocab_size

llm = LLM(
    model=model,
    enable_prefix_caching=False,
    gpu_memory_utilization=gpu_mem,
    max_model_len=max_model_len,
    tensor_parallel_size=1,
    enforce_eager=True,
    max_num_batched_tokens=max_num_batched_tokens,
)

sampling_params = SamplingParams(max_tokens=1)
results = []

for length in lengths:
    ids = torch.randint(0, vocab_size, (length,)).tolist()
    prompts = [ids]

    for _ in range(warmup):
        try:
            llm.generate(prompts=prompts, sampling_params=sampling_params,
                         use_tqdm=False)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            raise

    wall_times = []
    for rep in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(prompts=prompts, sampling_params=sampling_params,
                     use_tqdm=False)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_times.append(t1 - t0)

    mean_wall = sum(wall_times) / len(wall_times)
    results.append({
        "gpu_id": gpu_id,
        "length": length,
        "wall_times_s": wall_times,
        "mean_wall_s": mean_wall,
        "throughput_tok_per_s": length / mean_wall,
        "status": "ok",
    })

del llm
gc.collect()
torch.cuda.empty_cache()

with open(out_path, "w") as f:
    json.dump({
        "gpu_id": gpu_id,
        "vllm_version": vllm.__version__,
        "measurements": results,
    }, f, indent=2)
'''


def start_clock_logger(out_path, interval_ms=100):
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,memory.used",
         "--format=csv", "-lms", str(interval_ms)],
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


def run_gpu_count(n_gpus, args, out_dir, ts, lengths_str):
    """Run n_gpus independent vLLM workers simultaneously. Returns per-GPU results."""
    gpu_ids = list(range(n_gpus))
    worker_out_paths = []
    procs = []

    worker_script_path = out_dir / "_worker.py"
    worker_script_path.write_text(WORKER_SCRIPT)

    for gpu_id in gpu_ids:
        worker_out = out_dir / f"gpu{gpu_id}_n{n_gpus}.json"
        worker_out_paths.append(worker_out)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        env["PYTHONNOUSERSITE"] = "1"
        conda_lib = str(Path(sys.executable).resolve().parent.parent / "lib")
        env["LD_LIBRARY_PATH"] = conda_lib + ":" + env.get("LD_LIBRARY_PATH", "")

        proc = subprocess.Popen(
            [sys.executable, str(worker_script_path),
             args.model, lengths_str,
             str(args.reps), str(args.warmup),
             str(args.gpu_mem), str(args.max_model_len),
             str(args.max_num_batched_tokens), str(worker_out)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        procs.append((gpu_id, proc))

    print(f"  Launched {n_gpus} worker(s), waiting for completion...")
    sys.stdout.flush()

    results_per_gpu = {}
    for gpu_id, proc in procs:
        stdout, stderr = proc.communicate(timeout=1800)
        if proc.returncode != 0:
            print(f"  GPU {gpu_id}: FAILED (rc={proc.returncode})")
            stderr_text = stderr.decode("utf-8", errors="replace")[-2000:]
            print(f"    stderr (last 2000 chars): {stderr_text}")
            results_per_gpu[gpu_id] = {"error": stderr_text, "measurements": []}
        else:
            try:
                worker_out = worker_out_paths[gpu_id]
                with open(worker_out) as f:
                    data = json.load(f)
                results_per_gpu[gpu_id] = data
            except Exception as e:
                print(f"  GPU {gpu_id}: output parse error: {e}")
                results_per_gpu[gpu_id] = {"error": str(e), "measurements": []}

    return results_per_gpu


def main():
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    gpu_counts = [int(x) for x in args.gpu_counts.split(",")]
    lengths_str = args.lengths

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"multigpu_scaling_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,clocks.sm,clocks.max.sm,power.draw,power.limit",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    gpu_info = r.stdout.strip()
    n_available = len(gpu_info.strip().split("\n"))

    print("=" * 80)
    print(f"MULTI-GPU PREFILL SCALING — {ts}")
    print("=" * 80)
    print(f"GPUs available: {n_available}")
    print(f"GPU info:\n{gpu_info}")
    print(f"Model: {args.model}")
    print(f"Lengths: {lengths}")
    print(f"GPU counts to test: {gpu_counts}")
    print(f"Reps: {args.reps}, Warmup: {args.warmup}")
    print()

    clock_csv = out_dir / f"gpu_clocks_{ts}.csv"
    clock_proc = start_clock_logger(str(clock_csv))

    all_results = {
        "timestamp": ts,
        "model": args.model,
        "gpu_info": gpu_info,
        "n_gpus_available": n_available,
        "clock_csv": str(clock_csv),
        "lengths": lengths,
        "reps": args.reps,
        "warmup": args.warmup,
        "runs": {},
    }

    for n_gpus in gpu_counts:
        if n_gpus > n_available:
            print(f"\nSkipping n_gpus={n_gpus} (only {n_available} available)")
            continue

        print()
        print("=" * 80)
        print(f"  n_gpus = {n_gpus}")
        print("=" * 80)

        results = run_gpu_count(n_gpus, args, out_dir, ts, lengths_str)
        all_results["runs"][str(n_gpus)] = results

        for gpu_id in sorted(results):
            data = results[gpu_id]
            if "error" in data and not data.get("measurements"):
                print(f"  GPU {gpu_id}: ERROR")
                continue
            for m in data.get("measurements", []):
                print(f"  GPU {gpu_id}: L={m['length']:>6}  "
                      f"wall={m['mean_wall_s']:.4f}s  "
                      f"tput={m['throughput_tok_per_s']:.0f} tok/s")

    stop_clock_logger(clock_proc)

    # Parse clock data
    clock_data = parse_clock_csv(str(clock_csv))
    all_results["clock_distributions"] = {}
    for gpu_id, data in sorted(clock_data.items()):
        clocks = data["clocks"]
        power = data["power"]
        from collections import Counter
        c = Counter(clocks)
        total = len(clocks)
        dist_str = ", ".join(f"{mhz}MHz:{count/total:.0%}"
                             for mhz, count in c.most_common(5))
        mean_power = sum(power) / len(power) if power else 0
        all_results["clock_distributions"][str(gpu_id)] = {
            "n_samples": total,
            "clock_dist": dict(c),
            "mean_power_w": round(mean_power, 1),
            "min_clock_mhz": min(clocks) if clocks else None,
            "max_clock_mhz": max(clocks) if clocks else None,
        }
        print(f"\n  GPU {gpu_id} clocks: {dist_str}, mean power={mean_power:.1f}W")

    # Aggregate analysis
    print()
    print("=" * 80)
    print("AGGREGATE THROUGHPUT AND SCALING EFFICIENCY")
    print("=" * 80)

    single_gpu_tput = {}
    aggregate_tput = {}

    for n_gpus_str, run_data in sorted(all_results["runs"].items()):
        n_gpus = int(n_gpus_str)
        for gpu_id_str, gpu_data in run_data.items():
            for m in gpu_data.get("measurements", []):
                L = m["length"]
                tput = m["throughput_tok_per_s"]
                aggregate_tput.setdefault((n_gpus, L), []).append(tput)
                if n_gpus == 1:
                    single_gpu_tput[L] = tput

    print(f"\n  {'nGPU':>4}  {'Length':>6}  {'Agg tput':>12}  {'1-GPU tput':>12}  "
          f"{'Ideal (Nx1)':>12}  {'Efficiency':>10}")
    print("  " + "-" * 65)

    scaling_results = {}
    for (n_gpus, L), tputs in sorted(aggregate_tput.items()):
        agg = sum(tputs)
        single = single_gpu_tput.get(L)
        if single is None:
            continue
        ideal = n_gpus * single
        eff = agg / ideal if ideal > 0 else 0
        per_gpu_mean = agg / n_gpus

        print(f"  {n_gpus:>4}  {L:>6}  {agg:>12.0f}  {single:>12.0f}  "
              f"{ideal:>12.0f}  {eff:>9.1%}")

        this_run = all_results["runs"].get(str(n_gpus), {})
        per_gpu_walls = [mm["mean_wall_s"]
                         for gd in this_run.values()
                         for mm in gd.get("measurements", [])
                         if mm["length"] == L]

        scaling_results[(n_gpus, L)] = {
            "n_gpus": n_gpus,
            "length": L,
            "aggregate_tput_tok_s": agg,
            "single_gpu_tput_tok_s": single,
            "ideal_tput_tok_s": ideal,
            "efficiency": round(eff, 4),
            "per_gpu_tput_tok_s": per_gpu_mean,
            "per_gpu_service_s": L / per_gpu_mean if per_gpu_mean > 0 else None,
            "per_gpu_wall_times_s": per_gpu_walls,
        }

    all_results["scaling"] = {f"{k[0]}gpu_L{k[1]}": v
                              for k, v in scaling_results.items()}

    # Node-level fetch vs recompute
    print()
    print("=" * 80)
    print("NODE-LEVEL: HOW MANY GPUs BEFORE RECOMPUTE > FETCH?")
    print("=" * 80)

    import yaml
    constants_path = Path(__file__).resolve().parent.parent / "config" / "measured_constants.yaml"
    with open(constants_path) as f:
        consts = yaml.safe_load(f)

    S = consts["model"]["kv_bytes_per_token"]["value"]
    H_s = consts["fetch_overhead"]["v2_h_to_gpu_us_per_token"]["value"] * 1e-6
    M_s = consts["fetch_overhead"]["metadata_amortized_s_per_token"]["value"]

    BW_STATES = {
        "degraded": consts["bandwidth"]["hammerspace_degraded_mbps"]["value"],
        "quiescent": consts["bandwidth"]["hammerspace_quiescent_mbps"]["value"],
        "quiet_evening": consts["bandwidth"]["hammerspace_quiet_evening_mbps"]["value"],
        "peak_1stream": consts["bandwidth"]["hammerspace_peak_1stream_mbps"]["value"],
    }

    print(f"\n  Fetch model: T_fetch(G, L) = (G * S/BW_node + H + M) * L")
    print(f"    S = {S} bytes/tok, H = {H_s:.4e} s/tok, M = {M_s:.4e} s/tok")
    print(f"    BW_node = per-node storage ceiling (shared by all G GPUs)")
    print(f"    H = per-GPU PCIe (each GPU has its own link)")
    print()
    print(f"  Recompute model: T_recompute(L) per GPU = measured service time")
    print(f"    Aggregate capacity = G * (L / T_service_per_gpu(G))")
    print()

    node_crossover = {}
    for L in lengths:
        print(f"\n  L = {L}:")
        for bw_label, bw_mbps in BW_STATES.items():
            bw_bytes = bw_mbps * 1e6
            print(f"    {bw_label} ({bw_mbps} MB/s):")

            for n_gpus in gpu_counts:
                key = (n_gpus, L)
                if key not in scaling_results:
                    continue
                sr = scaling_results[key]
                per_gpu_service = sr["per_gpu_service_s"]
                if per_gpu_service is None:
                    continue

                # Aggregate recompute: G GPUs each doing L tokens in per_gpu_service seconds
                agg_recompute_tput = n_gpus * (L / per_gpu_service)

                # Aggregate fetch: node BW shared by G GPUs, each with own PCIe
                # Per-GPU fetch time = (G * S / BW_node + H + M) * L  (G GPUs share storage)
                # But each GPU has its own PCIe, so H is per-GPU
                # Total node fetch throughput = G * L / T_fetch_per_gpu
                t_fetch_per_tok = (n_gpus * S / bw_bytes) + H_s + M_s
                t_fetch_per_gpu = t_fetch_per_tok * L
                agg_fetch_tput = n_gpus * L / t_fetch_per_gpu

                ratio = agg_recompute_tput / agg_fetch_tput if agg_fetch_tput > 0 else float('inf')
                winner = "RECOMPUTE" if ratio > 1 else "FETCH"

                print(f"      G={n_gpus}: recompute={agg_recompute_tput:>10.0f} tok/s  "
                      f"fetch={agg_fetch_tput:>10.0f} tok/s  "
                      f"ratio={ratio:.3f}  → {winner}")

                node_crossover.setdefault(L, {}).setdefault(bw_label, {})[n_gpus] = {
                    "agg_recompute_tput": agg_recompute_tput,
                    "agg_fetch_tput": agg_fetch_tput,
                    "ratio_recompute_over_fetch": round(ratio, 4),
                    "winner": winner,
                    "per_gpu_service_s": per_gpu_service,
                    "t_fetch_per_tok_us": t_fetch_per_tok * 1e6,
                }

    all_results["node_crossover"] = {str(k): v for k, v in node_crossover.items()}

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for L in lengths:
        print(f"\n  L = {L}:")
        for bw_label, bw_mbps in BW_STATES.items():
            data = node_crossover.get(L, {}).get(bw_label, {})
            if not data:
                continue
            # Find the G where recompute first wins
            first_win = None
            for g in sorted(data):
                if data[g]["ratio_recompute_over_fetch"] > 1:
                    first_win = g
                    break
            if first_win:
                print(f"    {bw_label:>15}: recompute wins at G >= {first_win}")
            else:
                print(f"    {bw_label:>15}: fetch wins at all measured G")

    # Save
    out_path = out_dir / "results.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
