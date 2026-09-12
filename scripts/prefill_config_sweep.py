#!/usr/bin/env python3
"""Sweep vLLM prefill configs at a fixed context length to find the compute ceiling.

Tests multiple vLLM configurations to determine whether single-stream prefill
on H100 NVL is structurally memory-bound or can reach compute saturation.
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


CONDA_LIB = "/mnt/REPACSS/home/asmalla/miniforge3/envs/m1/lib"
CUDA_HOME = "/opt/apps/nfs/spack-1.1.0/opt/spack/linux-sapphirerapids/cuda-13.0.2-q2wona4ndh2i5ntywq4jwqkwkji5rm22"

MODEL = "/mnt/SHARED-AREA/Llama-series/Llama-3.1-8B"
REF_LENGTH = 32768
REPS = 3
WARMUP = 2


def start_power_logger(out_path):
    proc = subprocess.Popen(
        ["nvidia-smi",
         "--query-gpu=timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used",
         "--format=csv", "-l", "1"],
        stdout=open(out_path, "w"), stderr=subprocess.STDOUT,
    )
    time.sleep(2)
    if proc.poll() is not None:
        print(f"ERROR: nvidia-smi logger died", file=sys.stderr)
        sys.exit(1)
    lines = sum(1 for _ in open(out_path))
    if lines < 2:
        print(f"ERROR: power logger not producing data", file=sys.stderr)
        proc.kill()
        sys.exit(1)
    return proc


def stop_power_logger(proc, out_path):
    time.sleep(2)
    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
    return sum(1 for _ in open(out_path))


def run_config(config_name, llm_kwargs, batch_size=1):
    """Run a single config. Returns dict with results or error."""
    import torch
    from vllm import LLM, SamplingParams

    print(f"\n{'='*60}")
    print(f"Config {config_name}: {llm_kwargs}, batch={batch_size}")
    print(f"{'='*60}")

    result = {
        "config": config_name,
        "llm_kwargs": {k: str(v) for k, v in llm_kwargs.items()},
        "batch_size": batch_size,
        "length": REF_LENGTH,
        "status": "ok",
    }

    try:
        llm = LLM(model=MODEL, **llm_kwargs)
    except Exception as e:
        print(f"  FAILED to load: {e}")
        result["status"] = f"LOAD_FAILED: {e}"
        return result

    sampling_params = SamplingParams(max_tokens=1)
    token_ids = list(range(10, 10 + REF_LENGTH))
    prompts = [token_ids] * batch_size

    try:
        for _ in range(WARMUP):
            llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)

        times = []
        for _ in range(REPS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        result["times_s"] = times
        mean_t = sum(times) / len(times)
        result["mean_time_s"] = mean_t
        result["per_prompt_time_s"] = mean_t / batch_size
        result["total_tokens_per_sec"] = REF_LENGTH * batch_size / mean_t

        print(f"  mean={mean_t:.4f}s, per_prompt={mean_t/batch_size:.4f}s, "
              f"tok/s={REF_LENGTH*batch_size/mean_t:.0f}")

    except torch.cuda.OutOfMemoryError:
        result["status"] = "OOM"
        print(f"  OOM")
        import torch
        torch.cuda.empty_cache()
    except Exception as e:
        result["status"] = f"ERROR: {e}"
        print(f"  ERROR: {e}")

    del llm
    import gc
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    except Exception:
        pass
    time.sleep(2)

    return result


def main():
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    power_csv = out_dir / f"gpu_power_config_sweep_{ts}.csv"
    power_proc = start_power_logger(str(power_csv))
    print(f"Power logger: {power_csv}")

    base = dict(
        gpu_memory_utilization=0.85,
        max_model_len=131072,
        tensor_parallel_size=1,
        enable_prefix_caching=False,
    )

    configs = [
        ("A_baseline_eager_chunk16k", {**base, "enforce_eager": True, "max_num_batched_tokens": 16384}),
        ("B_compiled_chunk16k", {**base, "enforce_eager": False, "max_num_batched_tokens": 16384}),
        ("C_compiled_unchunked", {**base, "enforce_eager": False, "enable_chunked_prefill": False, "max_num_batched_tokens": 65536}),
        ("D_compiled_unchunked_batch4", {**base, "enforce_eager": False, "enable_chunked_prefill": False, "max_num_batched_tokens": 131072}),
    ]

    results = []

    for name, kwargs in configs:
        batch = 4 if "batch4" in name else 1
        r = run_config(name, kwargs, batch_size=batch)
        results.append(r)

    lines = stop_power_logger(power_proc, str(power_csv))
    print(f"\nPower CSV: {lines} lines in {power_csv}")

    out_path = out_dir / f"prefill_config_sweep_{ts}.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL,
            "ref_length": REF_LENGTH,
            "reps": REPS,
            "warmup": WARMUP,
            "power_csv": str(power_csv),
            "configs": results,
        }, f, indent=2)
    print(f"Results: {out_path}")


if __name__ == "__main__":
    main()
