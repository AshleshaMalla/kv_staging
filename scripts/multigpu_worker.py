#!/usr/bin/env python3
"""Single-GPU prefill worker — run on one GPU via CUDA_VISIBLE_DEVICES.

Measures N=1 prefill service time at specified lengths with the same settings
as the authoritative vLLM sweep (enforce_eager, prefix caching off, max_tokens=1).
Writes results to a JSON file.
"""

import argparse
import gc
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def rss_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)


def parse_args():
    p = argparse.ArgumentParser(description="Single-GPU prefill worker")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="16384,32768")
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=32768)
    p.add_argument("--max-num-batched-tokens", type=int, default=32768)
    p.add_argument("--out", required=True, help="Output JSON path")
    return p.parse_args()


def main():
    args = parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]

    gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
    print(f"[GPU {gpu_id}] Starting worker, lengths={lengths}, reps={args.reps}")
    sys.stdout.flush()

    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    import vllm

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size

    llm = LLM(
        model=args.model,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    host_rss = rss_gib()
    print(f"[GPU {gpu_id}] vLLM {vllm.__version__} loaded, host RSS={host_rss:.1f} GiB")
    sys.stdout.flush()

    sampling_params = SamplingParams(max_tokens=1)
    measurements = []

    for length in lengths:
        ids = torch.randint(0, vocab_size, (length,)).tolist()
        prompts = [ids]

        for _ in range(args.warmup):
            llm.generate(prompts=prompts, sampling_params=sampling_params,
                         use_tqdm=False)

        wall_times = []
        for rep in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            llm.generate(prompts=prompts, sampling_params=sampling_params,
                         use_tqdm=False)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            wall_times.append(t1 - t0)

        mean_wall = sum(wall_times) / len(wall_times)
        stdev_wall = (sum((w - mean_wall)**2 for w in wall_times) / max(len(wall_times) - 1, 1)) ** 0.5

        measurements.append({
            "gpu_id": gpu_id,
            "length": length,
            "wall_times_s": wall_times,
            "mean_wall_s": mean_wall,
            "stdev_wall_s": stdev_wall,
            "throughput_tok_per_s": length / mean_wall,
        })

        print(f"[GPU {gpu_id}] L={length}: mean={mean_wall:.4f}s "
              f"stdev={stdev_wall:.4f}s tput={length/mean_wall:.0f} tok/s")
        sys.stdout.flush()

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    peak_rss = rss_gib()
    result = {
        "gpu_id": gpu_id,
        "vllm_version": vllm.__version__,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "host_peak_rss_gib": round(peak_rss, 2),
        "measurements": measurements,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[GPU {gpu_id}] Done, results at {args.out}")


if __name__ == "__main__":
    main()
