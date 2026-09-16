#!/usr/bin/env python3
"""Diagnose whether vLLM actually batches concurrent prefills.

Three checks:
1. Does offline generate() batch multiple prompts in shared forward passes?
   → Enable per-step logging, report num_running + batched_tokens per step.
2. Does the per-request finished_time exist, and does it show overlap?
3. AsyncLLMEngine with truly concurrent arrivals: does Q(N) change?
"""

import argparse
import asyncio
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
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--length", type=int, default=4096)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--out", default="data/raw")
    return p.parse_args()


def check_offline_batching(args):
    """Check 1: Does offline LLM.generate() batch multiple prompts?"""
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    import vllm

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size

    print("=" * 80)
    print("CHECK 1: Offline LLM.generate() — does vLLM batch multiple prompts?")
    print("=" * 80)

    # Enable detailed logging
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

    llm = LLM(
        model=args.model,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=131072,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=16384,
        disable_log_stats=False,  # Enable stats logging
    )

    sampling_params = SamplingParams(max_tokens=1)

    N = args.concurrency
    L = args.length
    prompts = []
    for i in range(N):
        ids = torch.randint(0, vocab_size, (L,)).tolist()
        prompts.append(ids)

    # ── Single request baseline ──
    print(f"\n--- Single request baseline (N=1, L={L}) ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results_1 = llm.generate(prompts=[prompts[0]], sampling_params=sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wall_1 = t1 - t0
    print(f"  Wall time N=1: {wall_1:.4f}s")

    # Check per-request metrics
    r = results_1[0]
    print(f"  RequestOutput type: {type(r).__name__}")
    if hasattr(r, 'metrics') and r.metrics is not None:
        m = r.metrics
        attrs = [a for a in dir(m) if not a.startswith('_')]
        print(f"  Metrics attributes: {attrs}")
        if hasattr(m, 'finished_time') and hasattr(m, 'arrival_time'):
            print(f"    arrival_time: {m.arrival_time}")
            print(f"    finished_time: {m.finished_time}")
            if m.finished_time and m.arrival_time:
                print(f"    latency: {m.finished_time - m.arrival_time:.4f}s")
        if hasattr(m, 'first_token_time'):
            print(f"    first_token_time: {m.first_token_time}")
        if hasattr(m, 'time_in_queue'):
            print(f"    time_in_queue: {m.time_in_queue}")
    else:
        print(f"  No metrics available on RequestOutput")

    # ── Concurrent requests ──
    print(f"\n--- Concurrent requests (N={N}, L={L}) ---")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    results_n = llm.generate(prompts=prompts, sampling_params=sampling_params, use_tqdm=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    wall_n = t1 - t0
    print(f"  Wall time N={N}: {wall_n:.4f}s")
    print(f"  Ratio wall_N / wall_1: {wall_n / wall_1:.2f}x  (if serialized: {N:.1f}x)")

    # Check per-request metrics for all N
    print(f"\n  Per-request metrics (N={N}):")
    per_req_latencies = []
    for i, r in enumerate(results_n):
        if hasattr(r, 'metrics') and r.metrics is not None:
            m = r.metrics
            if hasattr(m, 'finished_time') and hasattr(m, 'arrival_time'):
                if m.finished_time is not None and m.arrival_time is not None:
                    lat = m.finished_time - m.arrival_time
                    per_req_latencies.append((i, m.arrival_time, m.finished_time, lat))
                    print(f"    req[{i}]: arrival={m.arrival_time:.4f} finished={m.finished_time:.4f} lat={lat:.4f}s")
                else:
                    print(f"    req[{i}]: arrival={m.arrival_time} finished={m.finished_time} (None)")
            else:
                print(f"    req[{i}]: no arrival/finished_time attributes")
        else:
            print(f"    req[{i}]: no metrics")

    if per_req_latencies:
        arrivals = [x[1] for x in per_req_latencies]
        finishes = [x[2] for x in per_req_latencies]
        lats = [x[3] for x in per_req_latencies]
        print(f"\n  Arrival spread: {max(arrivals) - min(arrivals):.6f}s")
        print(f"  Finish spread:  {max(finishes) - min(finishes):.6f}s")
        print(f"  Latency range:  {min(lats):.4f} - {max(lats):.4f}s")

        # Check overlap: if batched, requests should overlap in time
        sorted_by_finish = sorted(per_req_latencies, key=lambda x: x[2])
        first_finish = sorted_by_finish[0][2]
        last_finish = sorted_by_finish[-1][2]
        first_arrival = min(arrivals)
        print(f"  First request finishes at: {first_finish - first_arrival:.4f}s after submission")
        print(f"  Last request finishes at:  {last_finish - first_arrival:.4f}s after submission")
        if (first_finish - first_arrival) < wall_1 * 1.1:
            print(f"  → First finishes at ~service time: SERIAL execution (no batching benefit)")
        else:
            print(f"  → First finishes LATER than service time: possible batching")

    # ── Total tokens check ──
    total_tokens = N * L
    print(f"\n  Total tokens: {total_tokens}")
    print(f"  max_num_batched_tokens: 16384")
    if total_tokens <= 16384:
        print(f"  → All {N} requests COULD fit in a single batch step ({total_tokens} ≤ 16384)")
    else:
        print(f"  → Requires multiple batch steps ({total_tokens} > 16384)")
        chunks_needed = (total_tokens + 16383) // 16384
        print(f"  → Minimum {chunks_needed} steps if perfectly packed")

    # Shut down for next test
    del llm
    gc.collect()
    import torch as th
    th.cuda.empty_cache()
    time.sleep(2)

    return wall_1, wall_n


def check_async_engine(args):
    """Check 3: AsyncLLMEngine with truly concurrent arrivals."""
    import torch
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size

    print("\n" + "=" * 80)
    print("CHECK 2: AsyncLLMEngine — truly concurrent arrivals")
    print("=" * 80)

    # Use vllm's async engine
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from vllm.sampling_params import RequestOutputKind

    N = args.concurrency
    L = args.length

    prompts = []
    for i in range(N):
        ids = torch.randint(0, vocab_size, (L,)).tolist()
        prompts.append(ids)

    engine_args = AsyncEngineArgs(
        model=args.model,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=131072,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=16384,
        disable_log_stats=False,
    )

    async def run_async_test():
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)

        # ── N=1 baseline ──
        print(f"\n--- Async single request (N=1, L={L}) ---")
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        req_id = "baseline-0"
        final = None
        async for output in engine.generate(
            prompt={"prompt_token_ids": prompts[0]},
            sampling_params=sampling_params,
            request_id=req_id,
        ):
            final = output
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_1 = t1 - t0
        print(f"  Wall time N=1: {wall_1:.4f}s")

        if final and hasattr(final, 'metrics') and final.metrics:
            m = final.metrics
            if hasattr(m, 'finished_time') and hasattr(m, 'arrival_time'):
                if m.finished_time and m.arrival_time:
                    print(f"  Metrics latency: {m.finished_time - m.arrival_time:.4f}s")

        # ── N concurrent requests submitted simultaneously ──
        print(f"\n--- Async concurrent requests (N={N}, L={L}) ---")
        results = {}
        finish_times = {}
        submit_times = {}

        async def submit_one(idx):
            req_id = f"conc-{idx}"
            submit_times[idx] = time.perf_counter()
            final = None
            async for output in engine.generate(
                prompt={"prompt_token_ids": prompts[idx]},
                sampling_params=sampling_params,
                request_id=req_id,
            ):
                final = output
            finish_times[idx] = time.perf_counter()
            results[idx] = final

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        # Submit ALL requests concurrently
        await asyncio.gather(*[submit_one(i) for i in range(N)])
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        wall_n = t1 - t0

        print(f"  Wall time N={N}: {wall_n:.4f}s")
        print(f"  Ratio wall_N / wall_1: {wall_n / wall_1:.2f}x  (serialized={N:.1f}x)")

        # Per-request timing
        print(f"\n  Per-request completion times (from perf_counter):")
        sorted_by_finish = sorted(finish_times.items(), key=lambda x: x[1])
        t_submit = min(submit_times.values())
        for idx, ft in sorted_by_finish:
            st = submit_times[idx]
            print(f"    req[{idx}]: submit=+{st - t_submit:.4f}s  finish=+{ft - t_submit:.4f}s  total={ft - st:.4f}s")

        first_finish = sorted_by_finish[0][1] - t_submit
        last_finish = sorted_by_finish[-1][1] - t_submit
        print(f"\n  First completes at: +{first_finish:.4f}s")
        print(f"  Last completes at:  +{last_finish:.4f}s")
        print(f"  Spread (last-first): {last_finish - first_finish:.4f}s")

        # Check per-request metrics from vLLM
        print(f"\n  vLLM metrics per request:")
        for idx in range(N):
            r = results[idx]
            if r and hasattr(r, 'metrics') and r.metrics:
                m = r.metrics
                parts = []
                if hasattr(m, 'arrival_time') and m.arrival_time:
                    parts.append(f"arrival={m.arrival_time:.4f}")
                if hasattr(m, 'first_scheduled_time') and m.first_scheduled_time:
                    parts.append(f"first_sched={m.first_scheduled_time:.4f}")
                if hasattr(m, 'first_token_time') and m.first_token_time:
                    parts.append(f"first_tok={m.first_token_time:.4f}")
                if hasattr(m, 'finished_time') and m.finished_time:
                    parts.append(f"finished={m.finished_time:.4f}")
                if hasattr(m, 'time_in_queue') and m.time_in_queue is not None:
                    parts.append(f"queue_time={m.time_in_queue:.4f}")
                print(f"    req[{idx}]: {', '.join(parts)}")
            else:
                print(f"    req[{idx}]: no metrics")

        # If we have arrival/finished from metrics, compute real Q
        metric_lats = []
        for idx in range(N):
            r = results[idx]
            if r and hasattr(r, 'metrics') and r.metrics:
                m = r.metrics
                if (hasattr(m, 'finished_time') and m.finished_time and
                    hasattr(m, 'arrival_time') and m.arrival_time):
                    metric_lats.append((idx, m.arrival_time, m.finished_time,
                                       m.finished_time - m.arrival_time))

        if metric_lats:
            sorted_lats = sorted(metric_lats, key=lambda x: x[3])
            print(f"\n  Latency from vLLM metrics:")
            for idx, arr, fin, lat in sorted_lats:
                print(f"    req[{idx}]: {lat:.4f}s")
            first_lat = sorted_lats[0][3]
            last_lat = sorted_lats[-1][3]
            print(f"  First: {first_lat:.4f}s  Last: {last_lat:.4f}s  Ratio: {last_lat/first_lat:.2f}x")

        # Cleanup
        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()

        return wall_1, wall_n

    return asyncio.run(run_async_test())


def main():
    args = parse_args()

    print(f"Model: {args.model}")
    print(f"Test case: L={args.length}, N={args.concurrency}")
    print()

    # Check GPU clocks
    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    print(f"GPU: {r.stdout.strip()}")
    print()

    # Check 1: Offline batching
    wall_1_offline, wall_n_offline = check_offline_batching(args)

    # Check 2: Async engine
    try:
        wall_1_async, wall_n_async = check_async_engine(args)
    except Exception as e:
        print(f"\n  AsyncLLMEngine test failed: {e}")
        import traceback
        traceback.print_exc()
        wall_1_async, wall_n_async = None, None

    # ── Summary ──
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\n  Offline LLM.generate():")
    print(f"    N=1: {wall_1_offline:.4f}s  N={args.concurrency}: {wall_n_offline:.4f}s  ratio: {wall_n_offline/wall_1_offline:.2f}x")
    if wall_1_async:
        print(f"\n  Async engine (concurrent arrivals):")
        print(f"    N=1: {wall_1_async:.4f}s  N={args.concurrency}: {wall_n_async:.4f}s  ratio: {wall_n_async/wall_1_async:.2f}x")
    print(f"\n  If perfectly serialized: ratio = {args.concurrency:.1f}x")
    print(f"  If perfectly batched:    ratio ≈ 1.0x (for L*N ≤ batch budget)")


if __name__ == "__main__":
    main()
