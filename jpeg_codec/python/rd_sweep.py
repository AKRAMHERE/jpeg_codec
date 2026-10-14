#!/usr/bin/env python3
"""Rate-distortion analysis: compression ratio, bits/pixel, MSE and PSNR versus quality.

For each image the C++ codec is swept over many quality settings (`jlc sweep`). As an
independent sanity check the same sweep is run through libjpeg (via Pillow) in grayscale
mode with the same Q-table scaling and optimised Huffman tables, so our curve can be
compared against a mature implementation.

    python3 python/rd_sweep.py [--images camera astronaut ...] [--outdir results]

Outputs (in --outdir):  rd_<image>.csv, rd_curves.png, rd_quality.png, rd_summary.md
"""
import argparse
import csv
import io
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from make_test_images import read_pgm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
JLC = os.path.join(ROOT, "build", "jlc")
QUALITIES = [1, 2, 3, 5, 8, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 98, 100]


def sweep_ours(path):
    out = subprocess.run([JLC, "sweep", path, "-Q", ",".join(map(str, QUALITIES))],
                         check=True, capture_output=True, text=True).stdout
    rows = list(csv.DictReader(io.StringIO(out)))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def sweep_libjpeg(img):
    from PIL import Image
    rows = []
    for q in QUALITIES:
        buf = io.BytesIO()
        Image.fromarray(img, "L").save(buf, "JPEG", quality=q, optimize=True)
        dec = np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert("L")).astype(np.float64)
        mse = float(np.mean((dec - img.astype(np.float64)) ** 2))
        rows.append({"quality": q, "bytes": buf.tell(), "bpp": 8.0 * buf.tell() / img.size,
                     "mse": mse, "psnr": float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="*", default=["camera", "astronaut", "texture", "synthetic", "gradient", "noise"])
    ap.add_argument("--datadir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--outdir", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in args.images if os.path.exists(os.path.join(args.datadir, f"{n}.pgm"))]
    results = {}
    for n in names:
        path = os.path.join(args.datadir, f"{n}.pgm")
        img = read_pgm(path)
        ours, lib = sweep_ours(path), sweep_libjpeg(img)
        results[n] = (img, ours, lib)
        with open(os.path.join(args.outdir, f"rd_{n}.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["quality", "bytes", "bpp", "ratio", "mse", "psnr", "maxerr", "libjpeg_bytes", "libjpeg_bpp", "libjpeg_psnr"])
            for a, b in zip(ours, lib):
                w.writerow([int(a["quality"]), int(a["bytes"]), f"{a['bpp']:.5f}", f"{a['ratio']:.4f}",
                            f"{a['mse']:.5f}", f"{a['psnr']:.4f}", int(a["maxerr"]),
                            b["bytes"], f"{b['bpp']:.5f}", f"{b['psnr']:.4f}"])

    # ---- Figure 1: rate-distortion curves (PSNR vs bits per pixel)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    cmap = plt.get_cmap("tab10")
    for i, n in enumerate(names):
        _, ours, lib = results[n]
        fin = [r for r in ours if np.isfinite(r["psnr"])]
        axes[0].plot([r["bpp"] for r in fin], [r["psnr"] for r in fin], "-o", ms=3, color=cmap(i), label=n)
        finl = [r for r in lib if np.isfinite(r["psnr"])]
        axes[0].plot([r["bpp"] for r in finl], [r["psnr"] for r in finl], "--", lw=1, color=cmap(i), alpha=.6)
    axes[0].set_xlabel("bits per pixel (lower = smaller file)")
    axes[0].set_ylabel("PSNR [dB] (higher = closer to original)")
    axes[0].set_title("Rate-distortion: ours (solid) vs libjpeg (dashed)")
    axes[0].grid(alpha=.3)
    axes[0].legend(fontsize=8)
    axes[0].set_xscale("log")

    # compression ratio vs quality
    for i, n in enumerate(names):
        _, ours, _ = results[n]
        axes[1].plot([r["quality"] for r in ours], [r["ratio"] for r in ours], "-o", ms=3, color=cmap(i), label=n)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("quality setting")
    axes[1].set_ylabel("compression ratio  (raw bytes / file bytes)")
    axes[1].set_title("Compression ratio vs quality")
    axes[1].grid(alpha=.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "rd_curves.png"), dpi=160)

    # ---- Figure 2: distortion vs quality
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, n in enumerate(names):
        _, ours, _ = results[n]
        ax[0].plot([r["quality"] for r in ours], [r["mse"] for r in ours], "-o", ms=3, color=cmap(i), label=n)
        fin = [r for r in ours if np.isfinite(r["psnr"])]
        ax[1].plot([r["quality"] for r in fin], [r["psnr"] for r in fin], "-o", ms=3, color=cmap(i), label=n)
    ax[0].set_yscale("log"); ax[0].set_xlabel("quality"); ax[0].set_ylabel("MSE"); ax[0].grid(alpha=.3, which="both")
    ax[0].set_title("MSE vs quality"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("quality"); ax[1].set_ylabel("PSNR [dB]"); ax[1].grid(alpha=.3); ax[1].set_title("PSNR vs quality")
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "rd_quality.png"), dpi=160)

    # ---- Summary table (markdown)
    with open(os.path.join(args.outdir, "rd_summary.md"), "w") as f:
        for n in names:
            _, ours, lib = results[n]
            f.write(f"### {n}\n\n| quality | bytes | bpp | ratio | MSE | PSNR dB | max err | libjpeg bytes | libjpeg PSNR |\n|---|---|---|---|---|---|---|---|---|\n")
            for a, b in zip(ours, lib):
                if int(a["quality"]) in (1, 10, 25, 50, 75, 90, 95, 100):
                    f.write(f"| {int(a['quality'])} | {int(a['bytes'])} | {a['bpp']:.3f} | {a['ratio']:.2f} | {a['mse']:.2f} | "
                            f"{a['psnr']:.2f} | {int(a['maxerr'])} | {b['bytes']} | {b['psnr']:.2f} |\n")
            f.write("\n")
    print("wrote", args.outdir)
    for n in names:
        _, ours, lib = results[n]
        q50 = next(r for r in ours if r["quality"] == 50)
        l50 = next(r for r in lib if r["quality"] == 50)
        print(f"{n:10s} q=50: ours {q50['bytes']:.0f} B {q50['psnr']:.2f} dB | libjpeg {l50['bytes']} B {l50['psnr']:.2f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
