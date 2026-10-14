#!/usr/bin/env python3
"""Controlled experiment: is the 'dequantize' stage slow because of page faults on a fresh 32 MiB buffer?

Hypothesis. A 2048x2048 buffer of doubles is exactly 32 MiB, which is at glibc's maximum
dynamic mmap threshold, so malloc gives every call fresh zeroed pages from the kernel, and the
first write to each 4 KiB page costs a page fault. Test: run the SAME benchmark with malloc
told never to use mmap / never to return memory (so the buffer is recycled and its pages stay
mapped). If the hypothesis is right the 'staged' path speeds up a lot and the 'fused' path
(which never allocates a big buffer) does not change.

    python3 python/page_fault_experiment.py [--size 2048] [--quality 50] [--reps 25]

Writes results/malloc_experiment.csv and docs/snippets/malloc_experiment.tex.
"""
import argparse
import csv
import io
import os
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
BENCH = os.path.join(ROOT, "build", "jlc_bench")
NO_MMAP = {"MALLOC_MMAP_THRESHOLD_": str(1 << 30), "MALLOC_TRIM_THRESHOLD_": str(1 << 30), "MALLOC_TOP_PAD_": str(256 << 20)}


def run(size, quality, reps, fused, env_extra):
    env = dict(os.environ, **env_extra)
    out = subprocess.run([BENCH, "--sizes", str(size), "--quality", str(quality), "--reps", str(reps), "--fused", str(fused)],
                         check=True, capture_output=True, text=True, env=env).stdout
    return next(csv.DictReader(io.StringIO(out)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--quality", type=int, default=50)
    ap.add_argument("--reps", type=int, default=25)
    args = ap.parse_args()
    rows = []
    for label, fused, env in (("staged, default malloc", 0, {}), ("staged, mmap disabled", 0, NO_MMAP),
                              ("fused, default malloc", 1, {}), ("fused, mmap disabled", 1, NO_MMAP)):
        r = run(args.size, args.quality, args.reps, fused, env)
        rows.append((label, float(r["enc_total_ms"]), float(r["dec_total_ms"]), float(r["dec_dequant_ms"])))
        print(f"{label:26s} encode {rows[-1][1]:7.1f} ms   decode {rows[-1][2]:7.1f} ms   dequantize stage {rows[-1][3]:6.1f} ms")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "malloc_experiment.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["configuration", "encode_ms", "decode_ms", "dequantize_stage_ms"])
        w.writerows(rows)
    snip = os.path.join(ROOT, "docs", "snippets")
    os.makedirs(snip, exist_ok=True)
    with open(os.path.join(snip, "malloc_experiment.tex"), "w") as f:
        f.write(" \\\\\n".join(f"{label} & {e:.1f} & {d:.1f} & {dq:.1f}" for label, e, d, dq in rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
