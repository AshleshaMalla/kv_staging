#!/usr/bin/env python3
"""Measure realistic KV-cache fetch overhead: raw binary load + pinned H2D.

Reports D (disk-to-host) and H (host-to-GPU) per-token costs using mechanisms
that match production KV cache systems:
  D: os.preadv into pre-allocated buffer (no deserialization, no pickle)
  H: pinned-memory .cuda() transfer (not pageable bounce buffer)

Also measures pinned buffer allocation cost (one-time, amortized in real systems).

Usage:
  python3 scripts/measure_fetch_overhead.py [--target-dir /mnt/REPACSS]

Must run on a GPU node with sufficient memory (--mem=200G recommended).
"""

import argparse
import gc
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


KV_BYTES_PER_TOKEN = 131072  # 128 KiB
REPO_ROOT = Path(__file__).resolve().parent.parent
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_args():
    p = argparse.ArgumentParser(description="Measure realistic fetch overhead")
    p.add_argument("--target-dir", default="/mnt/REPACSS",
                   help="NFS directory for test files")
    p.add_argument("--reps", type=int, default=15)
    return p.parse_args()


def measure_raw_read(fpath, nbytes, buf, reps):
    """Read raw binary into pre-allocated buffer via os.preadv. No deserialization."""
    times = []
    for _ in range(reps):
        try:
            fd = os.open(fpath, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass

        fd = os.open(fpath, os.O_RDONLY)
        view = memoryview(buf)
        t0 = time.perf_counter()
        total = 0
        while total < nbytes:
            chunk = min(nbytes - total, 128 * 1024 * 1024)
            n = os.readv(fd, [view[total:total + chunk]])
            if n == 0:
                break
            total += n
        t1 = time.perf_counter()
        os.close(fd)
        times.append(t1 - t0)
    return times


def measure_pinned_h2d(cpu_tensor, reps):
    """Transfer pinned CPU tensor to GPU. Measures actual PCIe bandwidth."""
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        gpu_t = cpu_tensor.to("cuda", non_blocking=True)
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del gpu_t
    return times


def measure_pin_alloc(nbytes, reps):
    """Measure cost of allocating a pinned buffer."""
    n_elements = nbytes // 2
    times = []
    for _ in range(reps):
        gc.collect()
        torch.cuda.empty_cache()
        t0 = time.perf_counter()
        t = torch.empty(n_elements, dtype=torch.float16, pin_memory=True)
        t1 = time.perf_counter()
        times.append(t1 - t0)
        del t
    return times


def main():
    args = parse_args()
    base = os.path.join(args.target_dir, os.environ.get("USER", "test"), "fetch_overhead_test")
    os.makedirs(base, exist_ok=True)

    print("=" * 70)
    print("REALISTIC KV-CACHE FETCH OVERHEAD MEASUREMENT")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"KV bytes/token: {KV_BYTES_PER_TOKEN}")
    print(f"Target: {base}")
    print(f"Reps: {args.reps}")
    print()
    print("Mechanism assumptions (matching production KV cache systems):")
    print("  D: os.preadv into pre-allocated bytearray (no pickle, no deserialization)")
    print("  H: pinned-memory .to('cuda', non_blocking=True) with synchronize")
    print("  Pin allocation measured separately (one-time cost, pool-amortized)")
    print()

    # Warm up GPU
    x = torch.randn(256, 256, device="cuda")
    del x
    torch.cuda.synchronize()

    LENGTHS = [2048, 4096, 8192, 16384, 32768, 65536]
    results = []

    for length in LENGTHS:
        nbytes = length * KV_BYTES_PER_TOKEN
        size_mb = nbytes / (1024 * 1024)
        n_elements = nbytes // 2

        print(f"--- L={length} ({size_mb:.0f} MB) ---")

        # Create raw binary test file
        fpath = os.path.join(base, f"kv_raw_L{length}.bin")
        if not os.path.exists(fpath) or os.path.getsize(fpath) < nbytes:
            print(f"  Creating {size_mb:.0f} MB test file...")
            data = np.random.randn(n_elements).astype(np.float16)
            data.tofile(fpath)
            os.sync()
            del data
            gc.collect()
        else:
            print(f"  Test file exists, reusing")

        # D: raw read into pre-allocated buffer
        buf = bytearray(nbytes)
        d_times = measure_raw_read(fpath, nbytes, buf, args.reps)
        del buf
        gc.collect()

        d_med = statistics.median(d_times)
        d_per_tok = d_med / length * 1e6  # us/tok
        d_gbps = nbytes / d_med / 1e9     # GB/s

        print(f"  D (raw read):     {d_med*1e6:>10.0f} us total  "
              f"({d_per_tok:>.2f} us/tok)  implied {d_gbps:.2f} GB/s")

        # H: pinned transfer
        pin_tensor = torch.empty(n_elements, dtype=torch.float16, pin_memory=True)
        pin_tensor[:] = torch.randn(n_elements, dtype=torch.float16)
        h_times = measure_pinned_h2d(pin_tensor, args.reps)
        del pin_tensor
        gc.collect()
        torch.cuda.empty_cache()

        h_med = statistics.median(h_times)
        h_per_tok = h_med / length * 1e6
        h_gbps = nbytes / h_med / 1e9

        print(f"  H (pinned H2D):   {h_med*1e6:>10.0f} us total  "
              f"({h_per_tok:>.2f} us/tok)  implied {h_gbps:.2f} GB/s")

        # Pin allocation cost
        pa_times = measure_pin_alloc(nbytes, min(args.reps, 5))
        pa_med = statistics.median(pa_times)

        print(f"  Pin alloc:        {pa_med*1e6:>10.0f} us (one-time)")

        dh_per_tok = d_per_tok + h_per_tok
        print(f"  D+H total:        {dh_per_tok:>.2f} us/tok = {dh_per_tok/1e6:.4e} s/tok")
        print()

        results.append(dict(
            length=length, size_mb=size_mb,
            d_median_us=d_med * 1e6, d_per_tok_us=d_per_tok, d_gbps=d_gbps,
            h_median_us=h_med * 1e6, h_per_tok_us=h_per_tok, h_gbps=h_gbps,
            dh_per_tok_us=dh_per_tok,
            pin_alloc_us=pa_med * 1e6,
        ))

    # ── Summary ─────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Length':>8}  {'Size':>8}  {'D us/tok':>10}  {'D GB/s':>8}  "
          f"{'H us/tok':>10}  {'H GB/s':>8}  {'D+H us/tok':>12}  {'D+H s/tok':>12}")
    for r in results:
        print(f"  {r['length']:>8}  {r['size_mb']:>6.0f}MB  {r['d_per_tok_us']:>10.2f}  "
              f"{r['d_gbps']:>8.2f}  {r['h_per_tok_us']:>10.2f}  {r['h_gbps']:>8.2f}  "
              f"{r['dh_per_tok_us']:>12.2f}  {r['dh_per_tok_us']/1e6:>12.4e}")

    # Use L>=8192 for the recommended coefficient
    large = [r for r in results if r['length'] >= 8192]
    avg_d = statistics.mean([r['d_per_tok_us'] for r in large])
    avg_h = statistics.mean([r['h_per_tok_us'] for r in large])
    avg_dh = statistics.mean([r['dh_per_tok_us'] for r in large])
    avg_d_gbps = statistics.mean([r['d_gbps'] for r in large])
    avg_h_gbps = statistics.mean([r['h_gbps'] for r in large])

    print()
    print(f"  Recommended coefficients (mean for L>=8192):")
    print(f"    D:   {avg_d:.2f} us/tok  ({avg_d/1e6:.4e} s/tok)  implied {avg_d_gbps:.2f} GB/s")
    print(f"    H:   {avg_h:.2f} us/tok  ({avg_h/1e6:.4e} s/tok)  implied {avg_h_gbps:.2f} GB/s")
    print(f"    D+H: {avg_dh:.2f} us/tok  ({avg_dh/1e6:.4e} s/tok)")
    print()

    # Sanity checks
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)
    if avg_d_gbps > 6.5:
        print(f"  WARNING: D implied bandwidth ({avg_d_gbps:.1f} GB/s) exceeds peak measured "
              f"fio read (6.0 GB/s). Possible page cache hit — results may be optimistic.")
    else:
        print(f"  D implied bandwidth ({avg_d_gbps:.1f} GB/s) is within measured fio range "
              f"(2.6-6.0 GB/s). Plausible.")

    if avg_h_gbps < 30:
        print(f"  WARNING: H implied bandwidth ({avg_h_gbps:.1f} GB/s) is below expected "
              f"pinned PCIe Gen5 range (40-60 GB/s). May indicate fallback to pageable path.")
    elif avg_h_gbps > 70:
        print(f"  WARNING: H implied bandwidth ({avg_h_gbps:.1f} GB/s) exceeds PCIe Gen5 x16 "
              f"theoretical max (~63 GB/s). Check measurement.")
    else:
        print(f"  H implied bandwidth ({avg_h_gbps:.1f} GB/s) is in expected pinned PCIe "
              f"Gen5 range (40-60 GB/s). Good.")
    print()

    # Save results
    out = {
        "timestamp": TIMESTAMP,
        "gpu": torch.cuda.get_device_name(0),
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "mechanism": "raw binary read (os.preadv) + pinned H2D (torch pin_memory)",
        "reps": args.reps,
        "results": results,
        "recommended": {
            "d_us_per_tok": avg_d, "d_gbps": avg_d_gbps,
            "h_us_per_tok": avg_h, "h_gbps": avg_h_gbps,
            "dh_us_per_tok": avg_dh,
        },
    }
    out_path = REPO_ROOT / "data" / "raw" / f"fetch_overhead_{TIMESTAMP}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")

    # Clean up test files
    import shutil
    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
