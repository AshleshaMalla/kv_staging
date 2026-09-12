#!/usr/bin/env python3
"""Measure prefill latency via vLLM's offline LLM class.

Uses generate() with max_tokens=1 so we measure prefill only, not decode.
Prefix caching is explicitly disabled to ensure clean from-scratch prefill.
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
    p = argparse.ArgumentParser(description="vLLM prefill latency sweep")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--lengths",
        default="2048,4096,8192,16384,32768,65536,131072",
        help="Comma-separated context lengths to test",
    )
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--gpu-mem", type=float, default=0.85, help="gpu_memory_utilization")
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--tp", type=int, default=1, help="tensor_parallel_size")
    return p.parse_args()


def start_power_logger(out_path):
    proc = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,power.draw,utilization.gpu,utilization.memory,memory.used",
            "--format=csv",
            "-l", "1",
        ],
        stdout=open(out_path, "w"),
        stderr=subprocess.STDOUT,
    )
    time.sleep(2)
    if proc.poll() is not None:
        print(f"ERROR: nvidia-smi logger died (exit={proc.returncode})", file=sys.stderr)
        sys.exit(1)
    lines = sum(1 for _ in open(out_path))
    if lines < 2:
        print(f"ERROR: nvidia-smi logger not producing data ({lines} lines after 2s)", file=sys.stderr)
        proc.kill()
        sys.exit(1)
    print(f"Power logger running (PID={proc.pid}), {lines} lines after 2s")
    return proc


def stop_power_logger(proc, out_path):
    time.sleep(2)
    proc.send_signal(signal.SIGINT)
    proc.wait(timeout=10)
    lines = sum(1 for _ in open(out_path))
    print(f"Power logger stopped, {lines} total lines in {out_path}")


def compute_kv_bytes_per_token(config):
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return 2 * n_layers * n_kv_heads * head_dim * 2


def main():
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    from vllm import LLM, SamplingParams
    from transformers import AutoConfig
    import torch

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    n_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // n_heads)
    hidden_size = config.hidden_size
    kv_bytes_per_token = compute_kv_bytes_per_token(config)
    max_context = getattr(config, "max_position_embeddings", None)
    intermediate_size = getattr(config, "intermediate_size", None)

    print(f"Model: {args.model}")
    print(f"  n_layers={n_layers}, n_heads={n_heads}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  hidden_size={hidden_size}, intermediate_size={intermediate_size}")
    print(f"  kv_bytes_per_token={kv_bytes_per_token}, max_context={max_context}")
    print()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    power_csv = out_dir / f"gpu_power_vllm_{ts}.csv"
    power_proc = start_power_logger(str(power_csv))

    vllm_config = dict(
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
    )

    print(f"Loading vLLM with config: {vllm_config}")
    llm = LLM(model=args.model, **vllm_config)
    import vllm
    print(f"vLLM {vllm.__version__} loaded successfully")
    print()

    sampling_params = SamplingParams(max_tokens=1)

    results = {
        "model": args.model,
        "engine": "vllm",
        "vllm_version": vllm.__version__,
        "torch_version": torch.__version__,
        "vllm_config": {k: str(v) if not isinstance(v, (int, float, bool)) else v for k, v in vllm_config.items()},
        "model_config": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "n_heads": n_heads,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_attention_heads": n_heads,
            "max_position_embeddings": max_context,
        },
        "kv_bytes_per_token": kv_bytes_per_token,
        "reps": args.reps,
        "warmup": args.warmup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "power_csv": str(power_csv),
        "measurements": [],
    }

    # Build dummy prompts as token ID lists
    vocab_size = config.vocab_size

    header = f"{'Length':>8s}  {'mean_s':>10s}  {'std_s':>10s}  {'status':>8s}"
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    for length in lengths:
        token_ids = list(range(10, 10 + length))

        entry = {"length": length, "times_s": [], "status": "ok"}

        try:
            # Warmup
            for _ in range(args.warmup):
                llm.generate(
                    prompts=[token_ids],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )

            # Timed runs
            for _ in range(args.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                llm.generate(
                    prompts=[token_ids],
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                entry["times_s"].append(t1 - t0)

            mean_t = sum(entry["times_s"]) / len(entry["times_s"])
            std_t = (sum((t - mean_t) ** 2 for t in entry["times_s"]) / len(entry["times_s"])) ** 0.5
            print(f"{length:>8d}  {mean_t:>10.4f}  {std_t:>10.4f}  {'ok':>8s}")
            sys.stdout.flush()

        except torch.cuda.OutOfMemoryError:
            entry["status"] = "OOM"
            print(f"{length:>8d}  {'---':>10s}  {'---':>10s}  {'OOM':>8s}")
            sys.stdout.flush()
            torch.cuda.empty_cache()
            results["measurements"].append(entry)
            break
        except Exception as e:
            entry["status"] = f"ERROR: {e}"
            print(f"{length:>8d}  {'---':>10s}  {'---':>10s}  {'ERROR':>8s}: {e}")
            sys.stdout.flush()
            results["measurements"].append(entry)
            continue

        results["measurements"].append(entry)

    results["end_timestamp"] = datetime.now(timezone.utc).isoformat()

    out_path = out_dir / f"prefill_vllm_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {out_path}")

    stop_power_logger(power_proc, str(power_csv))


if __name__ == "__main__":
    main()
