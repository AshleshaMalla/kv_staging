#!/usr/bin/env python3
"""Measure prefill latency vs context length via raw forward passes.

Deliberately bypasses vLLM to avoid a day of setup. Uses bare
transformers + torch to time single-sequence forward passes at
increasing context lengths on a single GPU.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="Prefill latency sweep")
    p.add_argument("--model", required=True, help="HF model id or local path")
    p.add_argument(
        "--lengths",
        default="2048,4096,8192,16384,32768,65536,131072",
        help="Comma-separated context lengths to test",
    )
    p.add_argument("--reps", type=int, default=5, help="Timed repetitions per length")
    p.add_argument("--warmup", type=int, default=2, help="Warmup iterations per length")
    p.add_argument("--out", default="data/raw", help="Output directory")
    return p.parse_args()


def compute_kv_bytes_per_token(config):
    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    # 2 for K+V, 2 bytes for bfloat16
    return 2 * n_layers * n_kv_heads * head_dim * 2


def main():
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available", file=sys.stderr)
        sys.exit(1)

    print(f"Loading model: {args.model}")
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)

    n_layers = config.num_hidden_layers
    n_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    kv_bytes_per_token = compute_kv_bytes_per_token(config)
    max_context = getattr(config, "max_position_embeddings", None)

    print(f"  n_layers={n_layers}, n_kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"  kv_bytes_per_token = 2 * {n_layers} * {n_kv_heads} * {head_dim} * 2 = {kv_bytes_per_token}")
    print(f"  max_position_embeddings={max_context}")
    print()

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size

    results = {
        "model": args.model,
        "model_config": {
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            "hidden_size": config.hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "max_position_embeddings": max_context,
        },
        "kv_bytes_per_token": kv_bytes_per_token,
        "reps": args.reps,
        "warmup": args.warmup,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "measurements": [],
    }

    header = f"{'length':>8s}  {'mean_s':>10s}  {'std_s':>10s}  {'peak_hbm_gib':>13s}  {'status':>8s}"
    print(header)
    print("-" * len(header))
    sys.stdout.flush()

    for length in lengths:
        torch.cuda.reset_peak_memory_stats()

        input_ids = torch.randint(0, vocab_size, (1, length), device="cuda:0")

        entry = {"length": length, "times_s": [], "peak_hbm_bytes": None, "status": "ok"}

        try:
            # Warmup
            for _ in range(args.warmup):
                with torch.no_grad():
                    model(input_ids)
                torch.cuda.synchronize()

            # Timed runs
            for _ in range(args.reps):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    model(input_ids)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                entry["times_s"].append(t1 - t0)

            entry["peak_hbm_bytes"] = torch.cuda.max_memory_allocated()

            mean_t = sum(entry["times_s"]) / len(entry["times_s"])
            std_t = (sum((t - mean_t) ** 2 for t in entry["times_s"]) / len(entry["times_s"])) ** 0.5
            peak_gib = entry["peak_hbm_bytes"] / (1024**3)

            print(f"{length:>8d}  {mean_t:>10.4f}  {std_t:>10.4f}  {peak_gib:>13.2f}  {'ok':>8s}")
            sys.stdout.flush()

        except torch.cuda.OutOfMemoryError:
            entry["status"] = "OOM"
            entry["peak_hbm_bytes"] = torch.cuda.max_memory_allocated()
            peak_gib = entry["peak_hbm_bytes"] / (1024**3)
            print(f"{length:>8d}  {'---':>10s}  {'---':>10s}  {peak_gib:>13.2f}  {'OOM':>8s}")
            sys.stdout.flush()
            torch.cuda.empty_cache()
            results["measurements"].append(entry)
            break

        results["measurements"].append(entry)
        del input_ids

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"prefill_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
