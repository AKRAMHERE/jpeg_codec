#!/usr/bin/env python3
"""Generate every figure and every number used in docs/guide.tex from the real codec.

Nothing in the guide is typed in by hand: figures go to docs/figures/, LaTeX fragments to
docs/snippets/. Run from the project root after `make`, `rd_sweep.py` and `numerics_study.py`:

    python3 python/make_figures.py
"""
import csv
import math
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jlc_ref as ref  # noqa: E402
from make_test_images import read_pgm, write_pgm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "docs", "figures")
SNIP = os.path.join(ROOT, "docs", "snippets")
JLC = os.path.join(ROOT, "build", "jlc")
os.makedirs(FIG, exist_ok=True)
os.makedirs(SNIP, exist_ok=True)

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "figure.dpi": 150})
MACROS = []


def macro(name, value):
    MACROS.append(f"\\newcommand{{\\{name}}}{{{value}}}")


def tex_matrix(m, fmt="{:d}", env="pmatrix"):
    rows = [" & ".join(fmt.format(v) for v in row) for row in m]
    return f"\\begin{{{env}}}\n" + " \\\\\n".join(rows) + "\n\\end{" + env + "}"


def write(name, text):
    # A table body must not end in a row terminator: the chapters append
    # "\\ \bottomrule" themselves (a "\\" at end-of-file followed by
    # \bottomrule breaks LaTeX's tabular, see docs/guide.tex \inputrows).
    text = text.rstrip()
    if text.endswith("\\\\"):
        text = text[:-2].rstrip()
    with open(os.path.join(SNIP, name), "w") as f:
        f.write(text)


def cpp_decode(img, q):
    with tempfile.TemporaryDirectory() as t:
        write_pgm(f"{t}/a.pgm", img)
        subprocess.run([JLC, "encode", f"{t}/a.pgm", f"{t}/a.jlc", "-q", str(q)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([JLC, "decode", f"{t}/a.jlc", f"{t}/b.pgm"], check=True)
        return read_pgm(f"{t}/b.pgm"), os.path.getsize(f"{t}/a.jlc")


cam = read_pgm(os.path.join(ROOT, "data", "camera.pgm"))

# ------------------------------------------------------------------ 1. DCT basis images
fig, axes = plt.subplots(8, 8, figsize=(5.6, 5.6))
for u in range(8):
    for v in range(8):
        basis = np.outer(ref.C[u], ref.C[v])  # vertical frequency u, horizontal frequency v
        axes[u, v].imshow(basis, cmap="gray", vmin=-0.4, vmax=0.4, interpolation="nearest")
        axes[u, v].set_xticks([]); axes[u, v].set_yticks([])
        for sp in axes[u, v].spines.values():
            sp.set_linewidth(0.3)
fig.suptitle("The 64 DCT building blocks (row u = vertical frequency, column v = horizontal frequency)", fontsize=8)
fig.tight_layout(rect=(0, 0, 1, 0.97), h_pad=0.15, w_pad=0.15)
fig.savefig(os.path.join(FIG, "dct_basis.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ 2. zig-zag path
fig, ax = plt.subplots(figsize=(3.6, 3.6))
pos = {int(n): divmod(int(n), 8) for n in ref.ZZ}  # natural index -> (row, col)
order = [divmod(int(n), 8) for n in ref.ZZ]
ax.plot([c for r, c in order], [r for r, c in order], "-", color="tab:blue", lw=1.2, alpha=.8)
for i, (r, c) in enumerate(order):
    ax.text(c, r, str(i), ha="center", va="center", fontsize=6.5,
            bbox=dict(boxstyle="round,pad=0.15", fc="white" if i else "gold", ec="0.6", lw=0.4))
ax.set_xlim(-0.6, 7.6); ax.set_ylim(7.6, -0.6)
ax.set_xticks(range(8)); ax.set_yticks(range(8))
ax.set_xlabel("horizontal frequency v  (low -> high)"); ax.set_ylabel("vertical frequency u  (low -> high)")
ax.set_title("Zig-zag scan: position in the scan is written in each cell", fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "zigzag.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ 3. quality gallery
r0, c0, sz = 28, 150, 96
crop = cam[r0:r0 + sz, c0:c0 + sz]
panels = [("original", cam, os.path.getsize(os.path.join(ROOT, "data", "camera.pgm")))]
for q in (90, 50, 20, 5):
    dec, nbytes = cpp_decode(cam, q)
    panels.append((f"quality {q}", dec, nbytes))
fig, axes = plt.subplots(1, 5, figsize=(7.4, 1.85))
for ax, (title, img, nbytes) in zip(axes, panels):
    ax.imshow(img[r0:r0 + sz, c0:c0 + sz], cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    ratio = cam.size / nbytes
    ax.set_title(f"{title}\n" + ("raw 8 bpp" if title == "original" else f"{nbytes/1000:.1f} kB, {ratio:.0f}:1"), fontsize=7)
    ax.axis("off")
fig.tight_layout(pad=0.3)
fig.savefig(os.path.join(FIG, "quality_gallery.pdf"))
plt.close(fig)

# error map at quality 20
dec20, _ = cpp_decode(cam, 20)
err = np.abs(dec20.astype(int) - cam.astype(int))
fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.2))
axes[0].imshow(dec20, cmap="gray", vmin=0, vmax=255); axes[0].set_title("decoded, quality 20", fontsize=8); axes[0].axis("off")
im = axes[1].imshow(err, cmap="magma", vmin=0, vmax=40); axes[1].set_title("|decoded - original|  (bright = big error)", fontsize=8); axes[1].axis("off")
fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.02)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIG, "error_map.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ 4. energy compaction
coef = ref.forward_dct_blocks(cam).reshape(-1, 64)[:, ref.ZZ]
zq50 = ref.analyze(cam, 50)
fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
axes[0].semilogy(np.mean(np.abs(coef), axis=0) + 1e-3, "o-", ms=2.5, lw=0.8)
axes[0].set_xlabel("position in zig-zag scan (0 = DC)"); axes[0].set_ylabel("mean |coefficient|")
axes[0].set_title("Energy sits in the first few coefficients", fontsize=8); axes[0].grid(alpha=.3, which="both")
axes[1].plot(100 * np.mean(zq50 == 0, axis=0), "o-", ms=2.5, lw=0.8, color="tab:red")
axes[1].set_xlabel("position in zig-zag scan"); axes[1].set_ylabel("% of blocks where the value is 0")
axes[1].set_title("After quantization (q=50): the tail is almost all zeros", fontsize=8); axes[1].grid(alpha=.3)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIG, "energy_compaction.pdf"))
plt.close(fig)
macro("camZeroShareAC", f"{100 * np.mean(zq50[:, 1:] == 0):.0f}")
macro("camZeroShareLast", f"{100 * np.mean(zq50[:, 32:] == 0):.1f}")
macro("camNonzeroPerBlock", f"{np.count_nonzero(zq50, axis=1).mean():.1f}")

# ------------------------------------------------------------------ 5. DC prediction benefit
dc = zq50[:, 0]
diff = np.diff(np.concatenate([[0], dc]))


def entropy(a):
    _, c = np.unique(a, return_counts=True)
    p = c / c.sum()
    return float(-(p * np.log2(p)).sum())


fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
axes[0].hist(dc, bins=60, color="tab:gray"); axes[0].set_title(f"DC values (entropy {entropy(dc):.2f} bit)", fontsize=8)
axes[1].hist(diff, bins=60, color="tab:blue"); axes[1].set_title(f"DC differences (entropy {entropy(diff):.2f} bit)", fontsize=8)
for a in axes:
    a.set_xlabel("quantized value"); a.grid(alpha=.3)
axes[0].set_ylabel("number of blocks")
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(FIG, "dc_prediction.pdf"))
plt.close(fig)
macro("dcEntropyRaw", f"{entropy(dc):.2f}")
macro("dcEntropyDiff", f"{entropy(diff):.2f}")

# ------------------------------------------------------------------ 6. worked example block (real data)
Q50 = ref.quant_table(50)
zz_all = ref.analyze(cam, 50)
bw = cam.shape[1] // 8
chosen = None
for b in range(bw + 1, len(zz_all)):
    if b % bw == 0:
        continue
    nz = np.count_nonzero(zz_all[b, 1:])
    last = int(np.max(np.nonzero(zz_all[b])[0]))
    if 5 <= nz <= 7 and 10 <= last <= 30 and abs(zz_all[b, 0] - zz_all[b - 1, 0]) > 2:
        chosen = b
        break
assert chosen is not None
by, bx = divmod(chosen, bw)
pix = cam[by * 8:(by + 1) * 8, bx * 8:(bx + 1) * 8].astype(int)
shifted = pix - 128
dct = (ref.C @ shifted @ ref.C.T)
quant = np.floor(dct / Q50.reshape(8, 8) + 0.5 + 1e-9).astype(int)
zz = quant.reshape(64)[ref.ZZ]
prev_dc = int(zz_all[chosen - 1, 0])
macro("exBlockX", str(bx)); macro("exBlockY", str(by)); macro("exPrevDC", str(prev_dc))
write("ex_pixels.tex", tex_matrix(pix))
write("ex_shifted.tex", tex_matrix(shifted))
write("ex_dct.tex", tex_matrix(np.rint(dct).astype(int)))
write("ex_qtable.tex", tex_matrix(Q50.reshape(8, 8)))
write("ex_quant.tex", tex_matrix(quant))
write("ex_zigzag.tex", "[" + ",\\ ".join(str(int(v)) for v in zz[:max(np.nonzero(zz)[0]) + 4]) + ",\\ \\dots]")

# Huffman tables of the whole camera image at q=50 (exactly what the C++ encoder writes)
tokens_all = ref.tokenize(zz_all)
dcf, acf = [0] * 16, [0] * 256
for is_ac, s, _, _ in tokens_all:
    (acf if is_ac else dcf)[s] += 1
dct_t, act_t = ref.HuffTable(ref.huffman_lengths(dcf)), ref.HuffTable(ref.huffman_lengths(acf))
blk_tokens = ref.tokenize(np.vstack([np.array([prev_dc] + [0] * 63), zz.reshape(1, 64)]))
# drop the first (dummy) block's tokens: it consists of a DC token plus EOB
first_len = len(ref.tokenize(np.array([[prev_dc] + [0] * 63])))
blk_tokens = blk_tokens[first_len:]
rows, total_bits = [], 0
for is_ac, sym, nbits, amp in blk_tokens:
    t = act_t if is_ac else dct_t
    code = format(t.code[sym], f"0{t.length[sym]}b")
    ampb = format(amp, f"0{nbits}b") if nbits else "--"
    if not is_ac:
        meaning = f"DC difference, category {sym}"
        val = int(zz[0]) - prev_dc
    elif sym == ref.EOB:
        meaning, val = "EOB (rest is zero)", ""
    elif sym == ref.ZRL:
        meaning, val = "ZRL (16 zeros)", ""
    else:
        meaning = f"run {sym >> 4}, category {sym & 15}"
        val = ref.extend(amp, nbits)
    total_bits += len(code) + (nbits if nbits else 0)
    rows.append(f"{'DC' if not is_ac else 'AC'} & \\texttt{{0x{sym:02X}}} & {meaning} & {val} & \\texttt{{{code}}} & \\texttt{{{ampb}}} & {len(code) + nbits} \\\\")
write("ex_tokens.tex", "\n".join(rows))
macro("exBlockBits", str(total_bits))
macro("exBlockRawBits", "512")
macro("exBlockRatio", f"{512 / total_bits:.1f}")

# ------------------------------------------------------------------ 7. textbook Huffman example
freq = {"a": 45, "b": 13, "c": 12, "d": 16, "e": 9, "f": 5}
lens = ref.huffman_lengths(list(freq.values()))
tab = ref.HuffTable(lens)
write("huff_example.tex", "\n".join(
    f"\\texttt{{{s}}} & {f} & {lens[i]} & \\texttt{{{format(tab.code[i], f'0{lens[i]}b')}}} \\\\" for i, (s, f) in enumerate(freq.items())))
tot = sum(f * l for f, l in zip(freq.values(), lens))
macro("huffExBits", str(tot)); macro("huffExFixedBits", str(3 * sum(freq.values())))
macro("huffExTotal", str(sum(freq.values())))

# ------------------------------------------------------------------ 8. rate-distortion tables / numbers
def read_rd(name):
    with open(os.path.join(ROOT, "results", f"rd_{name}.csv")) as f:
        return list(csv.DictReader(f))


for name in ("camera", "noise", "gradient"):
    rows = read_rd(name)
    lines = []
    for r in rows:
        if int(r["quality"]) in (1, 10, 25, 50, 75, 90, 95, 100):
            psnr = r["psnr"] if r["psnr"] != "inf" else "$\\infty$"
            lines.append(f"{r['quality']} & {int(r['bytes']):,} & {float(r['bpp']):.3f} & {float(r['ratio']):.2f} & {float(r['mse']):.2f} & {psnr if psnr.startswith('$') else format(float(psnr), '.2f')} & {r['maxerr']} \\\\".replace(",", "\\,"))
    write(f"rd_{name}.tex", "\n".join(lines))
cam_rd = {int(r["quality"]): r for r in read_rd("camera")}
noise_rd = {int(r["quality"]): r for r in read_rd("noise")}
macro("camQfiftyBytes", f"{int(cam_rd[50]['bytes']):,}".replace(",", "\\,"))
macro("camQfiftyRatio", f"{float(cam_rd[50]['ratio']):.1f}")
macro("camQfiftyPSNR", f"{float(cam_rd[50]['psnr']):.1f}")
macro("camQfiftyBpp", f"{float(cam_rd[50]['bpp']):.2f}")
macro("camQhundredPSNR", f"{float(cam_rd[100]['psnr']):.1f}")
macro("camQhundredMSE", f"{float(cam_rd[100]['mse']):.3f}")
macro("noiseQhundredRatio", f"{float(noise_rd[100]['ratio']):.2f}")
macro("camLibBytes", f"{int(cam_rd[50]['libjpeg_bytes']):,}".replace(",", "\\,"))

# libjpeg comparison row for every image at q=50
lines = []
for name in ("camera", "astronaut", "texture", "synthetic", "gradient", "noise"):
    r = {int(x["quality"]): x for x in read_rd(name)}[50]
    ours, lib = int(r["bytes"]), int(r["libjpeg_bytes"])
    lines.append(f"{name} & {ours:,} & {lib:,} & {100 * (ours - lib) / lib:+.1f}\\% & {float(r['psnr']):.2f} & {float(r['libjpeg_psnr']):.2f} \\\\".replace(",", "\\,"))
write("libjpeg_compare.tex", "\n".join(lines))

# ------------------------------------------------------------------ 9. numerics tables (recomputed here)
sigma = math.sqrt(1 / 12)
p_flip = math.erfc(0.5 / sigma / math.sqrt(2))
macro("pFlip", f"{p_flip:.4f}")
lines = []
for name in ("camera", "astronaut", "texture", "synthetic", "noise"):
    im = read_pgm(os.path.join(ROOT, "data", f"{name}.pgm"))
    dec = ref.decode(ref.encode(im, 100))
    d = dec.astype(int) - im.astype(int)
    lines.append(f"{name} & {np.mean(d.astype(float) ** 2):.4f} & {p_flip:.4f} & {np.abs(d).max()} & {100 * np.mean(d != 0):.2f}\\% \\\\")
write("q100_table.tex", "\n".join(lines))

lines = []
for name in ("camera", "noise"):
    im = read_pgm(os.path.join(ROOT, "data", f"{name}.pgm"))
    for q in (10, 50, 90, 100):
        meas = float(np.mean((ref.decode(ref.encode(im, q)).astype(float) - im) ** 2))
        model = float(np.mean(ref.quant_table(q).astype(float) ** 2 / 12))
        lines.append(f"{name} & {q} & {meas:.2f} & {model:.2f} & {model / meas:.1f} \\\\")
write("model_vs_measured.tex", "\n".join(lines))

blocks = ref.to_blocks(ref.pad_to_blocks(cam).astype(float) - 128)
lines = []
for label, dtype in (("float64", np.float64), ("float32", np.float32)):
    Cm = ref.C.astype(dtype)
    e = (Cm.T @ (Cm @ blocks.astype(dtype) @ Cm.T) @ Cm).astype(float) - blocks
    lines.append(f"{label} & {np.abs(e).max():.1e} & {np.sqrt(np.mean(e ** 2)):.1e} \\\\")
for bits in (8, 10, 12, 13, 16):
    Cq = np.round(ref.C * 2 ** bits) / 2 ** bits
    e = Cq.T @ (Cq @ blocks @ Cq.T) @ Cq - blocks
    lines.append(f"fixed-point constants, {bits} fractional bits & {np.abs(e).max():.1e} & {np.sqrt(np.mean(e ** 2)):.1e} \\\\")
write("precision_table.tex", "\n".join(lines))

c = ref.forward_dct_blocks(cam).reshape(-1, 64)
ties = []
for q in (100, 90, 50):
    v = c / ref.quant_table(q)
    k = int((np.abs((v - np.floor(v)) - 0.5) < 1e-9).sum())
    ties.append(f"{q} & {v.size:,} & {k:,} & {100 * k / v.size:.2f}\\% \\\\".replace(",", "\\,"))
write("ties_table.tex", "\n".join(ties))

# ------------------------------------------------------------------ 10. sandbox profile example (clearly labelled in the guide)
def read_bench(path):
    with open(path) as f:
        return [{k: (v if k == "image" else float(v)) for k, v in r.items()} for r in csv.DictReader(f)]


sb = os.path.join(ROOT, "results", "sandbox_example")
if os.path.exists(os.path.join(sb, "bench_sandbox_fused0.csv")):
    f0 = read_bench(os.path.join(sb, "bench_sandbox_fused0.csv"))
    f1 = read_bench(os.path.join(sb, "bench_sandbox_fused1.csv"))
    lines = []
    for a, b in zip(f0, f1):
        if int(a["quality"]) != 50:
            continue
        side = int(a["width"])
        lines.append(f"{side}$\\times${side} & {a['enc_total_ms']:.1f} & {b['enc_total_ms']:.1f} & {a['enc_total_ms'] / b['enc_total_ms']:.2f}$\\times$ "
                     f"& {a['dec_total_ms']:.1f} & {b['dec_total_ms']:.1f} & {a['dec_total_ms'] / b['dec_total_ms']:.2f}$\\times$ \\\\")
    write("sandbox_fused.tex", "\n".join(lines))
    big = [r for r in f0 if int(r["width"]) == 2048 and int(r["quality"]) == 50][0]
    write("sandbox_stages.tex",
          "\n".join([
              f"DCT & {big['enc_dct_ms']:.1f} \\\\", f"quantize + zig-zag & {big['enc_quant_zigzag_ms']:.1f} \\\\",
              f"DC prediction + run-length tokens & {big['enc_scan_ms']:.1f} \\\\", f"build Huffman tables & {big['enc_huff_build_ms']:.2f} \\\\",
              f"bit packing & {big['enc_bitpack_ms']:.1f} \\\\", "\\midrule",
              f"Huffman decode & {big['dec_bitdecode_ms']:.1f} \\\\", f"dequantize + de-zig-zag & {big['dec_dequant_ms']:.1f} \\\\",
              f"inverse DCT + round & {big['dec_idct_ms']:.1f} \\\\", f"crop & {big['dec_crop_ms']:.2f} \\\\"]))
    macro("sbDequantMs", f"{big['dec_dequant_ms']:.0f}")
    macro("sbHuffDecMs", f"{big['dec_bitdecode_ms']:.0f}")

with open(os.path.join(SNIP, "macros.tex"), "w") as f:
    f.write("\n".join(MACROS) + "\n")
print("figures:", sorted(os.listdir(FIG)))
print("snippets:", sorted(os.listdir(SNIP)))
print(f"worked example block: x={bx} y={by}; prev DC={prev_dc}; block bits={total_bits}")
