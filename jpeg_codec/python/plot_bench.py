#!/usr/bin/env python3
"""Plot per-stage timing results written by `jlc_bench --csv`.

    python3 python/plot_bench.py results/bench_*.csv [--outdir results]

One CSV per machine / partition is expected (the file name, minus "bench_" and ".csv",
becomes the label). Produces:
  bench_stages.png     stacked bars: where does encode / decode time go (largest image, quality 50)
  bench_scaling.png    throughput (Mpixel/s) vs image size, per machine
  bench_summary.md     table of the numbers (paste into your report)
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

ENC = ["enc_dct_ms", "enc_quant_zigzag_ms", "enc_scan_ms", "enc_huff_build_ms", "enc_bitpack_ms"]
DEC = ["dec_bitdecode_ms", "dec_dequant_ms", "dec_idct_ms", "dec_crop_ms"]
ENC_LABEL = ["DCT", "quantize + zig-zag", "DC-pred + RLE tokens", "Huffman build", "bit packing"]
DEC_LABEL = ["Huffman decode", "dequantize", "IDCT + round", "crop"]


def load(path):
    with open(path) as f:
        return [{k: (v if k == "image" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--quality", type=int, default=50, help="quality used for the stage breakdown")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for p in args.csvs:
        label = os.path.basename(p)
        label = label[len("bench_"):] if label.startswith("bench_") else label
        data[label.rsplit(".csv", 1)[0]] = load(p)
    os.makedirs(args.outdir, exist_ok=True)

    # ---- stacked stage bars for the largest image
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    labels = list(data)
    cmap = plt.get_cmap("tab10")
    for ax, cols, names, title in ((axes[0], ENC, ENC_LABEL, "Encode"), (axes[1], DEC, DEC_LABEL, "Decode")):
        bottoms = [0.0] * len(labels)
        for ci, (col, nm) in enumerate(zip(cols, names)):
            vals = []
            for lab in labels:
                rows = [r for r in data[lab] if r["quality"] == args.quality]
                big = max(rows, key=lambda r: r["width"] * r["height"])
                vals.append(big[col])
            ax.bar(labels, vals, bottom=bottoms, label=nm, color=cmap(ci))
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        side = int(max(r["width"] for r in data[labels[0]]))
        ax.set_title(f"{title}: time per stage, {side}x{side}, quality {args.quality}")
        ax.set_ylabel("milliseconds (median)")
        ax.legend(fontsize=8)
        ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "bench_stages.png"), dpi=160)

    # ---- throughput vs size
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ci, lab in enumerate(labels):
        for ax, col, title in ((axes[0], "enc_mpix_s", "Encode throughput"), (axes[1], "dec_mpix_s", "Decode throughput")):
            by_q = defaultdict(list)
            for r in data[lab]:
                by_q[r["quality"]].append((r["width"] * r["height"] / 1e6, r[col]))
            for q, pts in sorted(by_q.items()):
                pts.sort()
                ax.plot([p[0] for p in pts], [p[1] for p in pts], "-o", ms=3,
                        label=f"{lab} q={int(q)}", color=cmap(ci), alpha=0.35 + 0.65 * (q / 100))
            ax.set_xscale("log"); ax.set_xlabel("image size [Mpixel]"); ax.set_ylabel("Mpixel / s"); ax.set_title(title)
            ax.grid(alpha=.3, which="both")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "bench_scaling.png"), dpi=160)

    # ---- markdown summary
    with open(os.path.join(args.outdir, "bench_summary.md"), "w") as f:
        for lab in labels:
            f.write(f"### {lab}\n\n| image | quality | bytes | enc ms | dec ms | enc Mpix/s | dec Mpix/s | encode: DCT / quant / scan / huff-build / pack (ms) | decode: huff / dequant / idct / crop (ms) |\n|---|---|---|---|---|---|---|---|---|\n")
            for r in data[lab]:
                f.write(f"| {r['image']} | {int(r['quality'])} | {int(r['bytes'])} | {r['enc_total_ms']:.1f} | {r['dec_total_ms']:.1f} | "
                        f"{r['enc_mpix_s']:.1f} | {r['dec_mpix_s']:.1f} | "
                        + " / ".join(f"{r[c]:.1f}" for c in ENC) + " | " + " / ".join(f"{r[c]:.1f}" for c in DEC) + " |\n")
            f.write("\n")
    print("wrote bench_stages.png, bench_scaling.png, bench_summary.md to", args.outdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
