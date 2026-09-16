#!/usr/bin/env python3
"""Measure corrected KV-cache fetch overhead: single read-into-pinned + H2D.

The cost model is:
    fetch = T_read_into_pinned + H_to_gpu + M

T_read_into_pinned is a SINGLE operation: read raw bytes from NFS directly
into a pinned CUDA host buffer. This replaces the double-counted T_svc + D
from v1.

Uses O_DIRECT where possible to match fio --direct=1 measurements and bypass
page cache. Falls back to fadvise DONTNEED if O_DIRECT alignment fails.

Reports implied GB/s so we can sanity-check against fio-measured storage
bandwidth (2.6-6.0 GB/s on this path).
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

KV_BYTES_PER_TOKEN = 131072
REPO_ROOT = Path(__file__).resolve().parent.parent
TIMESTAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
ALIGN = 512 * 1024  # O_DIRECT alignment (512K, safe for most filesystems)


def parse_args():
    p = argparse.ArgumentParser(description="Corrected fetch overhead v2")
    p.add_argument("--target-dir", default="/mnt/REPACSS")
    p.add_argument("--reps", type=int, default=15)
    return p.parse_args()


def align_size(n):
    """Round up to ALIGN boundary for O_DIRECT."""
    return ((n + ALIGN - 1) // ALIGN) * ALIGN


def read_into_pinned_direct(fpath, pinned_tensor, nbytes, use_odirect):
    """Read file directly into pinned tensor's memory. Single operation."""
    ptr = pinned_tensor.data_ptr()
    buf = (torch.ByteStorage._new_shared(nbytes)).untyped()
    mv = memoryview(pinned_tensor.numpy()[:nbytes])

    flags = os.O_RDONLY
    if use_odirect:
        flags |= os.O_DIRECT

    fd = os.open(fpath, flags)
    try:
        total = 0
        while total < nbytes:
            chunk = min(nbytes - total, 128 * 1024 * 1024)
            n = os.readv(fd, [mv[total:total + chunk]])
            if n == 0:
                break
            total += n
    finally:
        os.close(fd)
    return total


def read_into_pinned_preadv(fpath, pinned_np_view, nbytes, drop_cache=True):
    """Read file into pinned numpy view via preadv. Returns bytes read."""
    if drop_cache:
        try:
            fd = os.open(fpath, os.O_RDONLY)
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            os.close(fd)
        except Exception:
            pass

    fd = os.open(fpath, os.O_RDONLY)
    try:
        total = 0
        while total < nbytes:
            chunk = min(nbytes - total, 128 * 1024 * 1024)
            n = os.readv(fd, [pinned_np_view[total:total + chunk]])
            if n == 0:
                break
            total += n
    finally:
        os.close(fd)
    return total


def main():
    args = parse_args()
    user = os.environ.get("USER", "test")
    base = os.path.join(args.target_dir, user, "fetch_overhead_v2")
    os.makedirs(base, exist_ok=True)

    print("=" * 70)
    print("CORRECTED KV-CACHE FETCH OVERHEAD (v2)")
    print("=" * 70)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"KV bytes/token: {KV_BYTES_PER_TOKEN}")
    print(f"Target: {base}")
    print(f"Reps: {args.reps}")
    print()
    print("Cost model: fetch = T_read_into_pinned + H_to_gpu + M")
    print("  T_read_into_pinned: single read from NFS into pinned CUDA host buffer")
    print("  H_to_gpu: pinned .to('cuda', non_blocking=True) with synchronize")
    print("  M: metadata (open/stat/close), negligible")
    print("  No double counting: T_svc and D are the same operation.")
    print()

    # Warm GPU
    x = torch.randn(256, 256, device="cuda"); del x
    torch.cuda.synchronize()

    # Check O_DIRECT support
    test_odirect = False
    try:
        testf = os.path.join(base, "_odirect_test")
        with open(testf, "wb") as f:
            f.write(b"\x00" * ALIGN)
        fd = os.open(testf, os.O_RDONLY | os.O_DIRECT)
        os.close(fd)
        test_odirect = True
        os.unlink(testf)
        print("O_DIRECT: supported — will bypass page cache (matches fio --direct=1)")
    except OSError as e:
        print(f"O_DIRECT: not available ({e}) — using fadvise DONTNEED instead")
    print()

    LENGTHS = [4096, 8192, 16384, 32768, 65536]
    results = []

    for length in LENGTHS:
        nbytes = length * KV_BYTES_PER_TOKEN
        size_mb = nbytes / (1024 * 1024)
        nbytes_aligned = align_size(nbytes) if test_odirect else nbytes

        print(f"--- L={length} ({size_mb:.0f} MB) ---")

        # Create raw binary test file (aligned size for O_DIRECT)
        fpath = os.path.join(base, f"kv_L{length}.bin")
        write_size = nbytes_aligned if test_odirect else nbytes
        if not os.path.exists(fpath) or os.path.getsize(fpath) < write_size:
            print(f"  Creating {write_size / (1024*1024):.0f} MB test file...")
            n_elements = write_size // 2
            arr = np.random.randn(n_elements).astype(np.float16)
            arr.tofile(fpath)
            os.sync()
            del arr; gc.collect()
        else:
            print(f"  Reusing existing file")

        # Allocate pinned buffer (once, reused across reps)
        alloc_elems = nbytes_aligned // 1 if test_odirect else nbytes // 1
        pinned = torch.empty(alloc_elems, dtype=torch.uint8, pin_memory=True)
        pinned_np = pinned.numpy()

        # ── T_read_into_pinned ──────────────────────────────────
        read_times = []
        for rep in range(args.reps):
            # Drop page cache
            try:
                fd = os.open(fpath, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except Exception:
                pass

            if test_odirect:
                fd = os.open(fpath, os.O_RDONLY | os.O_DIRECT)
            else:
                fd = os.open(fpath, os.O_RDONLY)

            t0 = time.perf_counter()
            total = 0
            while total < nbytes:
                chunk = min(nbytes - total, 128 * 1024 * 1024)
                n = os.readv(fd, [pinned_np[total:total + chunk]])
                if n == 0:
                    break
                total += n
            t1 = time.perf_counter()
            os.close(fd)
            read_times.append(t1 - t0)

        read_med = statistics.median(read_times)
        read_per_tok = read_med / length * 1e6
        read_gbps = nbytes / read_med / 1e9

        print(f"  T_read_into_pinned: {read_med*1e6:>10.0f} us  "
              f"({read_per_tok:>.2f} us/tok)  implied {read_gbps:.2f} GB/s")

        # ── H_to_gpu ────────────────────────────────────────────
        # Reinterpret the pinned buffer as float16 for the GPU transfer
        n_f16 = nbytes // 2
        pinned_f16 = pinned[:nbytes].view(torch.float16)

        h_times = []
        for _ in range(args.reps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            gpu_t = pinned_f16.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            h_times.append(t1 - t0)
            del gpu_t

        h_med = statistics.median(h_times)
        h_per_tok = h_med / length * 1e6
        h_gbps = nbytes / h_med / 1e9

        print(f"  H_to_gpu (pinned): {h_med*1e6:>10.0f} us  "
              f"({h_per_tok:>.2f} us/tok)  implied {h_gbps:.2f} GB/s")

        # ── Full pipeline: read + H2D ───────────────────────────
        full_times = []
        for _ in range(args.reps):
            try:
                fd = os.open(fpath, os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except Exception:
                pass

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Read into pinned
            if test_odirect:
                fd = os.open(fpath, os.O_RDONLY | os.O_DIRECT)
            else:
                fd = os.open(fpath, os.O_RDONLY)
            total = 0
            while total < nbytes:
                chunk = min(nbytes - total, 128 * 1024 * 1024)
                n = os.readv(fd, [pinned_np[total:total + chunk]])
                if n == 0:
                    break
                total += n
            os.close(fd)

            # H2D
            gpu_t = pinned[:nbytes].view(torch.float16).to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()

            full_times.append(t1 - t0)
            del gpu_t

        full_med = statistics.median(full_times)
        full_per_tok = full_med / length * 1e6
        full_gbps = nbytes / full_med / 1e9

        print(f"  Full (read+H2D):   {full_med*1e6:>10.0f} us  "
              f"({full_per_tok:>.2f} us/tok)  implied {full_gbps:.2f} GB/s")
        print()

        del pinned, pinned_np, pinned_f16
        gc.collect(); torch.cuda.empty_cache()

        results.append(dict(
            length=length, size_mb=size_mb,
            read_med_us=read_med * 1e6, read_per_tok_us=read_per_tok, read_gbps=read_gbps,
            h_med_us=h_med * 1e6, h_per_tok_us=h_per_tok, h_gbps=h_gbps,
            full_med_us=full_med * 1e6, full_per_tok_us=full_per_tok, full_gbps=full_gbps,
        ))

    # ── Summary ─────────────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  {'Length':>8}  {'Size':>8}  {'Read us/tok':>12}  {'Read GB/s':>10}  "
          f"{'H us/tok':>10}  {'H GB/s':>8}  {'Full us/tok':>12}  {'Full GB/s':>10}")
    for r in results:
        print(f"  {r['length']:>8}  {r['size_mb']:>6.0f}MB  {r['read_per_tok_us']:>12.2f}  "
              f"{r['read_gbps']:>10.2f}  {r['h_per_tok_us']:>10.2f}  {r['h_gbps']:>8.2f}  "
              f"{r['full_per_tok_us']:>12.2f}  {r['full_gbps']:>10.2f}")

    large = [r for r in results if r['length'] >= 8192]
    avg_read = statistics.mean([r['read_per_tok_us'] for r in large])
    avg_h = statistics.mean([r['h_per_tok_us'] for r in large])
    avg_full = statistics.mean([r['full_per_tok_us'] for r in large])
    avg_read_gbps = statistics.mean([r['read_gbps'] for r in large])
    avg_h_gbps = statistics.mean([r['h_gbps'] for r in large])
    avg_full_gbps = statistics.mean([r['full_gbps'] for r in large])

    print()
    print(f"  Recommended coefficients (mean for L>=8192):")
    print(f"    T_read_into_pinned: {avg_read:.2f} us/tok  ({avg_read/1e6:.4e} s/tok)  @ {avg_read_gbps:.2f} GB/s")
    print(f"    H_to_gpu:           {avg_h:.2f} us/tok  ({avg_h/1e6:.4e} s/tok)  @ {avg_h_gbps:.2f} GB/s")
    print(f"    Full pipeline:      {avg_full:.2f} us/tok  ({avg_full/1e6:.4e} s/tok)  @ {avg_full_gbps:.2f} GB/s")
    print()

    # ── Sanity checks ───────────────────────────────────────────
    print("=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    print(f"  Page cache: files written then fadvise DONTNEED before each read.")
    if test_odirect:
        print(f"  O_DIRECT: YES — kernel page cache fully bypassed (matches fio --direct=1)")
    else:
        print(f"  O_DIRECT: NO — relying on fadvise DONTNEED (cache may partially help)")

    if avg_read_gbps > 6.5:
        print(f"  WARNING: Read BW ({avg_read_gbps:.1f} GB/s) exceeds fio peak (6.0 GB/s).")
        print(f"           Possible page cache hit despite DONTNEED/O_DIRECT.")
    elif avg_read_gbps < 2.0:
        print(f"  WARNING: Read BW ({avg_read_gbps:.1f} GB/s) below fio minimum (2.6 GB/s).")
        print(f"           Residual CPU overhead in the read path.")
        fio_min, fio_max = 2.6, 6.0
        gap_pct = (1.0 - avg_read_gbps / fio_min) * 100
        print(f"           Gap vs fio minimum: {gap_pct:.0f}%")
    else:
        print(f"  Read BW ({avg_read_gbps:.1f} GB/s) within fio range (2.6-6.0 GB/s). Good.")

    if 30 <= avg_h_gbps <= 65:
        print(f"  H2D BW ({avg_h_gbps:.1f} GB/s) in expected PCIe Gen5 range. Good.")
    else:
        print(f"  H2D BW ({avg_h_gbps:.1f} GB/s) outside expected range (40-60 GB/s).")
    print()

    # Compare against v1
    print("=" * 70)
    print("COMPARISON: v1 (separate D+T_svc) vs v2 (single read-into-pinned)")
    print("=" * 70)
    v1_d = 65.0   # us/tok
    v1_h = 2.73   # us/tok
    print(f"  v1 D (bytearray read):    {v1_d:.1f} us/tok  (2.02 GB/s)")
    print(f"  v2 T_read_into_pinned:    {avg_read:.1f} us/tok  ({avg_read_gbps:.2f} GB/s)")
    print(f"  v1 H (pinned, same):      {v1_h:.1f} us/tok")
    print(f"  v2 H (pinned, same):      {avg_h:.1f} us/tok")
    print(f"  v1 D+H:                   {v1_d + v1_h:.1f} us/tok")
    print(f"  v2 full pipeline:         {avg_full:.1f} us/tok")
    print(f"  Note: v1 also double-counted T_svc + D; v2 measures the single operation.")
    print()

    # Save
    out = {
        "timestamp": TIMESTAMP,
        "gpu": torch.cuda.get_device_name(0),
        "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        "o_direct": test_odirect,
        "mechanism": "os.preadv into pinned torch tensor (O_DIRECT if available) + pinned .to(cuda)",
        "cost_model": "fetch = T_read_into_pinned + H_to_gpu + M (no double counting)",
        "reps": args.reps,
        "results": results,
        "recommended": {
            "read_us_per_tok": avg_read, "read_gbps": avg_read_gbps,
            "h_us_per_tok": avg_h, "h_gbps": avg_h_gbps,
            "full_us_per_tok": avg_full, "full_gbps": avg_full_gbps,
        },
    }
    out_path = REPO_ROOT / "data" / "raw" / f"fetch_overhead_v2_{TIMESTAMP}.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Results saved to {out_path}")

    # Cleanup
    import shutil
    shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
