#!/usr/bin/env python3
"""Extended Q(N) sweep — measure wall-time scaling at N >> 16.

Job 155248 measured Q(N) = 0.85*(N-1)*service at N <= 16. The feedback
simulation (queue_feedback.py) reaches N ~ 41-68 during metastability.
This script extends the measurement to N = 24, 32, 48, 64 to determine
whether the linear relationship holds or whether the slope changes.

Method:
  Phase 1 — Offline LLM.generate() (same as prefill_loaded_sweep_v2.py)
    Submit N prompts of length L simultaneously. Measure wall time.
    Record N=1 baseline for each length.
  Phase 2 — AsyncLLMEngine with staggered arrivals (serving-realistic)
    Submit N prompts with near-simultaneous asyncio.gather().
    Record per-request metrics (arrival, first_scheduled, finished).

Reports T_wall(N) / T_service(N=1) = 1 + slope*(N-1) and fits the slope
across the full measured range.

Clock logging: nvidia-smi at 100ms throughout every measurement.
OOM recording: records (length, N) at which vLLM refuses or CUDA OOMs.
"""

import argparse
import asyncio
import gc
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Extended Q(N) concurrency sweep")
    p.add_argument("--model", required=True)
    p.add_argument("--lengths", default="4096,16384,32768",
                   help="Context lengths to sweep (short first)")
    p.add_argument("--concurrencies", default="1,2,4,8,16,24,32,48,64",
                   help="Concurrency levels to sweep")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--out", default="data/raw")
    p.add_argument("--gpu-mem", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=131072)
    p.add_argument("--max-num-batched-tokens", type=int, default=65536)
    p.add_argument("--skip-async", action="store_true",
                   help="Skip AsyncLLMEngine phase")
    p.add_argument("--async-points", default="1,4,16,32,48,64",
                   help="(N values for async phase; subset of --concurrencies)")
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


def report_clock_distribution(csv_path):
    """Parse nvidia-smi CSV and report SM clock distribution."""
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
            return "no clock data"
        from collections import Counter
        c = Counter(clocks)
        total = len(clocks)
        dist = ", ".join(f"{mhz}MHz:{count/total:.0%}" for mhz, count in c.most_common(5))
        return f"n={total} samples, {dist}"
    except Exception as e:
        return f"error reading clocks: {e}"


def run_offline_sweep(args):
    """Phase 1: Offline LLM.generate() sweep."""
    import torch
    from transformers import AutoConfig
    from vllm import LLM, SamplingParams
    import vllm

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size
    kv_bpt = 2 * config.num_hidden_layers * getattr(config, "num_key_value_heads", config.num_attention_heads) * getattr(config, "head_dim", config.hidden_size // config.num_attention_heads) * 2

    lengths = [int(x) for x in args.lengths.split(",")]
    concurrencies = [int(x) for x in args.concurrencies.split(",")]

    print(f"Model: {args.model}")
    print(f"  vocab_size={vocab_size}, kv_bytes_per_token={kv_bpt}")
    print(f"  vLLM {vllm.__version__}")
    print(f"Lengths: {lengths}")
    print(f"Concurrencies: {concurrencies}")
    print()

    llm = LLM(
        model=args.model,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
    )

    sampling_params = SamplingParams(max_tokens=1)
    measurements = []
    oom_points = []

    header = (f"  {'Length':>6}  {'N':>4}  {'Wall(s)':>8}  {'Svc1(s)':>8}  "
              f"{'Ratio':>6}  {'Slope':>6}  {'Tput':>10}  {'Status':>6}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    n1_service = {}

    for length in lengths:
        for conc in concurrencies:
            total_tokens = length * conc

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
                for _ in range(args.warmup):
                    try:
                        llm.generate(prompts=prompts,
                                     sampling_params=sampling_params,
                                     use_tqdm=False)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        raise

                for rep in range(args.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    results = llm.generate(prompts=prompts,
                                           sampling_params=sampling_params,
                                           use_tqdm=False)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    entry["wall_times_s"].append(t1 - t0)

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
                entry["mean_wall_s"] = mean_wall
                entry["service_time_s"] = mean_wall / conc
                entry["throughput_tok_per_s"] = total_tokens / mean_wall

                if conc == 1:
                    n1_service[length] = mean_wall

                if entry["per_request_times_s"]:
                    all_sorted = entry["per_request_times_s"]
                    entry["first_s"] = sum(s[0] for s in all_sorted) / len(all_sorted)
                    entry["median_s"] = sum(s[len(s)//2] for s in all_sorted) / len(all_sorted)
                    entry["last_s"] = sum(s[-1] for s in all_sorted) / len(all_sorted)
                else:
                    entry["first_s"] = mean_wall / conc
                    entry["median_s"] = mean_wall * (conc // 2 + 0.5) / conc
                    entry["last_s"] = mean_wall

                svc1 = n1_service.get(length, mean_wall)
                ratio = mean_wall / svc1
                slope = (ratio - 1) / (conc - 1) if conc > 1 else 0.0

                print(f"  {length:>6}  {conc:>4}  {mean_wall:>8.3f}  {svc1:>8.3f}  "
                      f"{ratio:>6.2f}  {slope:>6.3f}  "
                      f"{entry['throughput_tok_per_s']:>10.0f}  {'ok':>6}")
                sys.stdout.flush()

            except torch.cuda.OutOfMemoryError:
                entry["status"] = "OOM"
                oom_points.append({"length": length, "concurrency": conc,
                                   "reason": "CUDA OOM"})
                print(f"  {length:>6}  {conc:>4}  {'---':>8}  {'---':>8}  "
                      f"{'---':>6}  {'---':>6}  {'---':>10}  {'OOM':>6}")
                sys.stdout.flush()
                torch.cuda.empty_cache()
                measurements.append(entry)
                break

            except Exception as e:
                entry["status"] = f"ERROR: {e}"
                err_str = str(e)
                if "out of memory" in err_str.lower() or "cannot allocate" in err_str.lower():
                    oom_points.append({"length": length, "concurrency": conc,
                                       "reason": err_str[:200]})
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

    return measurements, oom_points, n1_service, kv_bpt


def run_async_sweep(args, n1_service):
    """Phase 2: AsyncLLMEngine with near-simultaneous arrivals."""
    import torch
    from transformers import AutoConfig
    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    vocab_size = config.vocab_size
    lengths = [int(x) for x in args.lengths.split(",")]
    async_ns = [int(x) for x in args.async_points.split(",")]

    engine_args = AsyncEngineArgs(
        model=args.model,
        enable_prefix_caching=False,
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=args.max_model_len,
        tensor_parallel_size=1,
        enforce_eager=True,
        max_num_batched_tokens=args.max_num_batched_tokens,
        disable_log_stats=True,
    )

    async def do_sweep():
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        sampling_params = SamplingParams(max_tokens=1)
        results_all = []
        oom_points = []

        print()
        print("=" * 80)
        print("PHASE 2: AsyncLLMEngine — concurrent arrivals")
        print("=" * 80)
        header = (f"  {'Length':>6}  {'N':>4}  {'Wall(s)':>8}  {'Svc1(s)':>8}  "
                  f"{'Ratio':>6}  {'Spread':>8}  {'Status':>6}")
        print(header)
        print("  " + "-" * (len(header) - 2))

        for length in lengths:
            for conc in async_ns:
                prompts = []
                for i in range(conc):
                    ids = torch.randint(0, vocab_size, (length,)).tolist()
                    prompts.append(ids)

                entry = {
                    "length": length,
                    "concurrency": conc,
                    "wall_times_s": [],
                    "per_request_latencies": [],
                    "status": "ok",
                }

                try:
                    for rep in range(args.reps):
                        finish_times = {}
                        submit_times = {}
                        req_results = {}

                        async def submit_one(idx, rep_num):
                            req_id = f"sweep-L{length}-N{conc}-r{rep_num}-{idx}"
                            submit_times[idx] = time.perf_counter()
                            final = None
                            async for output in engine.generate(
                                prompt={"prompt_token_ids": prompts[idx]},
                                sampling_params=sampling_params,
                                request_id=req_id,
                            ):
                                final = output
                            finish_times[idx] = time.perf_counter()
                            req_results[idx] = final

                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        await asyncio.gather(*[submit_one(i, rep) for i in range(conc)])
                        torch.cuda.synchronize()
                        t1 = time.perf_counter()

                        wall = t1 - t0
                        entry["wall_times_s"].append(wall)

                        lats = []
                        for idx in range(conc):
                            r = req_results.get(idx)
                            if r and hasattr(r, 'metrics') and r.metrics:
                                m = r.metrics
                                if (hasattr(m, 'finished_time') and m.finished_time and
                                    hasattr(m, 'arrival_time') and m.arrival_time):
                                    lats.append(m.finished_time - m.arrival_time)
                        if lats:
                            entry["per_request_latencies"].append(sorted(lats))

                    mean_wall = sum(entry["wall_times_s"]) / len(entry["wall_times_s"])
                    entry["mean_wall_s"] = mean_wall
                    svc1 = n1_service.get(length, mean_wall)
                    ratio = mean_wall / svc1

                    if entry["per_request_latencies"]:
                        all_lats = entry["per_request_latencies"]
                        entry["first_s"] = sum(s[0] for s in all_lats) / len(all_lats)
                        entry["last_s"] = sum(s[-1] for s in all_lats) / len(all_lats)
                        spread = entry["last_s"] - entry["first_s"]
                    else:
                        spread = float('nan')

                    print(f"  {length:>6}  {conc:>4}  {mean_wall:>8.3f}  {svc1:>8.3f}  "
                          f"{ratio:>6.2f}  {spread:>8.4f}  {'ok':>6}")
                    sys.stdout.flush()

                except torch.cuda.OutOfMemoryError:
                    entry["status"] = "OOM"
                    oom_points.append({"length": length, "concurrency": conc,
                                       "reason": "CUDA OOM (async)"})
                    print(f"  {length:>6}  {conc:>4}  {'---':>8}  {'---':>8}  "
                          f"{'---':>6}  {'---':>8}  {'OOM':>6}")
                    sys.stdout.flush()
                    torch.cuda.empty_cache()
                    results_all.append(entry)
                    break

                except Exception as e:
                    entry["status"] = f"ERROR: {e}"
                    print(f"  {length:>6}  {conc:>4}  {'---':>8}  {'---':>8}  "
                          f"{'---':>6}  {'---':>8}  {'ERR':>6}")
                    sys.stdout.flush()
                    results_all.append(entry)
                    continue

                results_all.append(entry)

        engine.shutdown()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
        return results_all, oom_points

    return asyncio.run(do_sweep())


def fit_qn(measurements, n1_service):
    """Fit Q(N) relationship from measurements."""
    import numpy as np

    points = []
    for m in measurements:
        if m["status"] != "ok" or m["concurrency"] < 2:
            continue
        svc1 = n1_service.get(m["length"])
        if svc1 is None:
            continue
        ratio = m["mean_wall_s"] / svc1
        points.append((m["length"], m["concurrency"], ratio, m["mean_wall_s"], svc1))

    if not points:
        return None

    print()
    print("=" * 80)
    print("Q(N) FIT ACROSS FULL MEASURED RANGE")
    print("=" * 80)

    ns = np.array([p[1] for p in points], dtype=np.float64)
    ratios = np.array([p[2] for p in points], dtype=np.float64)

    print(f"\n  {'Length':>6}  {'N':>4}  {'Ratio':>7}  {'Slope':>7}  {'Residual':>9}")
    for length, conc, ratio, wall, svc1 in sorted(points):
        slope = (ratio - 1) / (conc - 1)
        print(f"  {length:>6}  {conc:>4}  {ratio:>7.3f}  {slope:>7.4f}")

    # Fit 1: single linear slope across all data
    # ratio = 1 + slope * (N - 1)  =>  ratio - 1 = slope * (N - 1)
    nm1 = ns - 1
    deltas = ratios - 1
    slope_linear = float(np.sum(nm1 * deltas) / np.sum(nm1 ** 2))
    pred_linear = 1 + slope_linear * nm1
    resid_linear = ratios - pred_linear
    rmse_linear = float(np.sqrt(np.mean(resid_linear ** 2)))

    print(f"\n  Fit 1 — Single linear: ratio = 1 + {slope_linear:.4f}*(N-1)")
    print(f"    RMSE = {rmse_linear:.4f}")
    print(f"    Compare to job 155248 slope: 0.85")

    # Fit 2: quadratic  ratio = 1 + a*(N-1) + b*(N-1)^2
    A = np.column_stack([nm1, nm1**2])
    coeffs_q, _, _, _ = np.linalg.lstsq(A, deltas, rcond=None)
    a_q, b_q = coeffs_q
    pred_quad = 1 + a_q * nm1 + b_q * nm1**2
    resid_quad = ratios - pred_quad
    rmse_quad = float(np.sqrt(np.mean(resid_quad ** 2)))

    print(f"\n  Fit 2 — Quadratic: ratio = 1 + {a_q:.4f}*(N-1) + {b_q:.6f}*(N-1)^2")
    print(f"    RMSE = {rmse_quad:.4f}")
    curvature = "SUB-LINEAR" if b_q < 0 else "SUPER-LINEAR"
    print(f"    Curvature: {curvature} (b {'< 0' if b_q < 0 else '> 0'})")

    # Fit 3: power law  ratio = 1 + alpha * (N-1)^beta
    # log(ratio - 1) = log(alpha) + beta * log(N-1)
    mask = (ns > 1) & (ratios > 1)
    if np.sum(mask) >= 3:
        log_nm1 = np.log(nm1[mask])
        log_delta = np.log(deltas[mask])
        A_pow = np.column_stack([np.ones_like(log_nm1), log_nm1])
        coeffs_pow, _, _, _ = np.linalg.lstsq(A_pow, log_delta, rcond=None)
        log_alpha, beta = coeffs_pow
        alpha = np.exp(log_alpha)
        pred_pow = 1 + alpha * nm1**beta
        resid_pow = ratios - pred_pow
        rmse_pow = float(np.sqrt(np.mean(resid_pow ** 2)))

        print(f"\n  Fit 3 — Power law: ratio = 1 + {alpha:.4f}*(N-1)^{beta:.4f}")
        print(f"    RMSE = {rmse_pow:.4f}")
        if beta < 0.95:
            print(f"    beta < 1: SUB-LINEAR scaling (batching improves at high N)")
        elif beta > 1.05:
            print(f"    beta > 1: SUPER-LINEAR scaling (overhead grows at high N)")
        else:
            print(f"    beta ≈ 1: consistent with linear Q(N)")
    else:
        alpha, beta, rmse_pow = None, None, None
        print("\n  Fit 3 — Power law: insufficient data (need N>1 with ratio>1)")

    # Fit 4: per-length slopes (do slopes vary with L?)
    print(f"\n  Per-length slopes:")
    by_length = {}
    for length, conc, ratio, wall, svc1 in points:
        by_length.setdefault(length, []).append((conc, ratio))
    per_length_slopes = {}
    for length in sorted(by_length):
        pts = by_length[length]
        if len(pts) >= 2:
            ns_l = np.array([p[0] for p in pts], dtype=np.float64)
            rs_l = np.array([p[1] for p in pts], dtype=np.float64)
            nm1_l = ns_l - 1
            d_l = rs_l - 1
            slope_l = float(np.sum(nm1_l * d_l) / np.sum(nm1_l ** 2))
            pred_l = 1 + slope_l * nm1_l
            rmse_l = float(np.sqrt(np.mean((rs_l - pred_l) ** 2)))
            max_n = int(max(ns_l))
            per_length_slopes[length] = slope_l
            print(f"    L={length:>6}: slope={slope_l:.4f}, RMSE={rmse_l:.4f}, max_N={max_n}")

    print()
    # Verdict
    print("  VERDICT:")
    if rmse_quad < rmse_linear * 0.8 and abs(b_q) > 1e-5:
        print(f"    Linear model inadequate. Quadratic term b={b_q:.6f} is significant.")
        print(f"    The Q(N) relationship is {curvature} at high N.")
        if b_q < 0:
            print("    Implication: metastability drain times are SHORTER than predicted")
            print("    by the linear model. The headline finding weakens.")
        else:
            print("    Implication: metastability drain times are LONGER than predicted")
            print("    by the linear model. The headline finding strengthens.")
    else:
        print(f"    Linear model adequate (RMSE linear {rmse_linear:.4f} vs quad {rmse_quad:.4f}).")
        print(f"    Q(N) = {slope_linear:.4f}*(N-1)*service holds across measured range.")

    return {
        "linear": {"slope": slope_linear, "rmse": rmse_linear},
        "quadratic": {"a": float(a_q), "b": float(b_q), "rmse": rmse_quad},
        "power_law": {"alpha": float(alpha) if alpha is not None else None,
                      "beta": float(beta) if beta is not None else None,
                      "rmse": float(rmse_pow) if rmse_pow is not None else None},
        "per_length_slopes": {int(k): v for k, v in per_length_slopes.items()},
        "n_points": len(points),
        "max_N_measured": int(max(ns)),
    }


def main():
    args = parse_args()

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.out) / f"qn_extended_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    r = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,clocks.sm,clocks.max.sm,power.draw,power.limit",
         "--format=csv,noheader"],
        capture_output=True, text=True)
    print(f"GPU: {r.stdout.strip()}")
    print()

    clock_csv = out_dir / f"gpu_clocks_{ts}.csv"
    clock_proc = start_clock_logger(str(clock_csv), interval_ms=100)

    print("=" * 80)
    print(f"EXTENDED Q(N) SWEEP — {ts}")
    print("=" * 80)
    print(f"Goal: determine whether Q(N) = 0.85*(N-1)*service holds at N >> 16")
    print()

    # Phase 1: Offline
    print("=" * 80)
    print("PHASE 1: Offline LLM.generate()")
    print("=" * 80)
    measurements, oom_offline, n1_service, kv_bpt = run_offline_sweep(args)

    # Phase 2: Async
    async_results = []
    oom_async = []
    if not args.skip_async:
        try:
            async_results, oom_async = run_async_sweep(args, n1_service)
        except Exception as e:
            print(f"\nAsync phase failed: {e}")
            import traceback
            traceback.print_exc()

    # Q(N) fit
    ok_measurements = [m for m in measurements if m["status"] == "ok"]
    fit_result = fit_qn(ok_measurements, n1_service) if ok_measurements else None

    # OOM summary
    all_oom = oom_offline + oom_async
    if all_oom:
        print()
        print("=" * 80)
        print("OOM / CAPACITY BOUNDARY")
        print("=" * 80)
        for oom in all_oom:
            kv_gib = (kv_bpt * oom["length"] * oom["concurrency"]) / (1024**3)
            print(f"  L={oom['length']}, N={oom['concurrency']}: "
                  f"total KV={kv_gib:.1f} GiB — {oom['reason']}")

    # Clock summary
    stop_clock_logger(clock_proc)
    clock_dist = report_clock_distribution(str(clock_csv))
    print(f"\nSM clock distribution: {clock_dist}")

    # Save
    out_data = {
        "timestamp": ts,
        "model": args.model,
        "kv_bytes_per_token": kv_bpt,
        "gpu_info": r.stdout.strip(),
        "clock_csv": str(clock_csv),
        "clock_distribution": clock_dist,
        "offline_measurements": measurements,
        "async_measurements": async_results,
        "n1_service_times": {str(k): v for k, v in n1_service.items()},
        "oom_points": all_oom,
        "fit_result": fit_result,
    }

    out_path = out_dir / f"results.json"
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Human-readable summary
    summary_path = out_dir / "summary.txt"
    lines = []
    lines.append(f"Extended Q(N) sweep — {ts}")
    lines.append(f"GPU: {r.stdout.strip()}")
    lines.append(f"SM clocks: {clock_dist}")
    lines.append("")
    if fit_result:
        lines.append(f"Linear fit: slope = {fit_result['linear']['slope']:.4f} "
                      f"(RMSE {fit_result['linear']['rmse']:.4f})")
        lines.append(f"  cf. job 155248 slope = 0.85")
        lines.append(f"Quadratic: a={fit_result['quadratic']['a']:.4f}, "
                      f"b={fit_result['quadratic']['b']:.6f} "
                      f"(RMSE {fit_result['quadratic']['rmse']:.4f})")
        if fit_result["power_law"]["beta"] is not None:
            lines.append(f"Power law: alpha={fit_result['power_law']['alpha']:.4f}, "
                          f"beta={fit_result['power_law']['beta']:.4f} "
                          f"(RMSE {fit_result['power_law']['rmse']:.4f})")
        lines.append(f"Max N measured: {fit_result['max_N_measured']}")
        lines.append(f"Per-length slopes: {fit_result['per_length_slopes']}")
    if all_oom:
        lines.append("")
        lines.append("OOM boundary:")
        for oom in all_oom:
            lines.append(f"  L={oom['length']}, N={oom['concurrency']}")
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
