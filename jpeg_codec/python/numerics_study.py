#!/usr/bin/env python3
"""Numerical-error study of the codec: rounding, coefficient range, reconstruction error.

Experiments (all on the NumPy reference model, which is bit-identical to the C++ code):

  1. Coefficient range        observed min/max of DC and AC coefficients vs the theoretical bound
                              (|DC| <= 1024) and what that means for symbol categories / int16 storage.
  2. Where does the error     quantization error per coefficient is ~uniform in [-Q/2, Q/2], so by
     come from?               Parseval (orthonormal DCT) the pixel MSE should be about
                              (1/64) * sum_i Q_i^2 / 12.  Compare prediction with measurement.
  3. Quality 100              all steps are 1 -> pure rounding error. Gaussian model predicts the
                              MSE after final rounding to integers; compare with measurement.
  4. Floating-point precision DCT + inverse DCT without quantization in float64 / float32 / fixed-point
                              constants of B fractional bits.
  5. Rounding ties            how many coefficients are within 1e-9 of an exact .5 tie (why the
                              deterministic tie-break exists).

    python3 python/numerics_study.py
"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jlc_ref as ref  # noqa: E402
from make_test_images import read_pgm  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
OUT = []


def say(s=""):
    print(s)
    OUT.append(s)


def mse(a, b):
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def main():
    names = [n for n in ["camera", "astronaut", "texture", "synthetic", "noise"]
             if os.path.exists(os.path.join(ROOT, "data", f"{n}.pgm"))]
    imgs = {n: read_pgm(os.path.join(ROOT, "data", f"{n}.pgm")) for n in names}

    # ------------------------------------------------------------------ 1. coefficient range
    say("## 1. Coefficient range (raw DCT coefficients, before quantization)\n")
    say("| image | DC min | DC max | max abs AC | max abs AC (category) |\n|---|---|---|---|---|")
    for n, im in imgs.items():
        c = ref.forward_dct_blocks(im).reshape(-1, 64)
        dc, ac = c[:, 0], c[:, 1:]
        say(f"| {n} | {dc.min():.1f} | {dc.max():.1f} | {np.abs(ac).max():.1f} | {ref.category(int(round(np.abs(ac).max())))} |")
    # worst case constructed analytically: all pixels 255 (level-shifted +127) or 0 (-128)
    worst_dc = 8 * 128
    say(f"\nTheory: level-shifted pixels lie in [-128, 127], so |DC| = |sum|/8 <= 8*128 = {worst_dc}.")
    say("Stronger: the transform is orthonormal, so by Parseval EVERY coefficient (DC or AC) satisfies |Y| <= ||X||_2 <= sqrt(64*128^2) = 1024; "
        "the astronaut image reaches it exactly (a pure black block).")
    say("Consequence: quantized values fit in int16; a DC *difference* is at most 2048 -> category 12; an AC value is at most 1024 -> category 11; "
        "so the 4-bit `size` nibble (max 15) is always sufficient.\n")

    # ------------------------------------------------------------------ 2. predicted vs measured MSE
    say("## 2. Predicted vs measured MSE\n")
    say("Model: coefficient error ~ uniform(-Q/2, Q/2) -> variance Q^2/12 (valid when coefficients are much larger than Q).")
    say("Parseval: pixel-domain MSE = mean over the 64 coefficient positions of their error variance.\n")
    say("| image | quality | measured MSE | model MSE | model / measured |\n|---|---|---|---|---|")
    for n in ["camera", "noise"]:
        if n not in imgs:
            continue
        im = imgs[n]
        for q in (10, 30, 50, 75, 90, 95, 100):
            meas = mse(im, ref.decode(ref.encode(im, q)))
            Q = ref.quant_table(q).astype(np.float64)
            model = float(np.mean(Q ** 2 / 12.0))
            say(f"| {n} | {q} | {meas:.3f} | {model:.3f} | {model / meas:.2f} |")
    say("\nReading: for noise (large coefficients everywhere) the model is within ~40% at quality 10 and within ~10% from quality 50 upwards. "
        "For the natural photo it OVER-estimates massively at low quality (40x at quality 10) because most high-frequency coefficients are tiny: "
        "they quantize to zero and their error is just their own small value (< Q/2), far smaller than the uniform-error assumption. "
        "The model converges to the truth as Q -> 1 (ratio 1.00 at quality 100). Takeaway: Q^2/12 is an UPPER-bound-style estimate "
        "for real images; the dead-zone (zeroed coefficients) is where lossy compression gets its savings.\n")

    # ------------------------------------------------------------------ 3. quality 100
    say("## 3. Quality 100: pure rounding error\n")
    # pre-rounding pixel error e ~ N(0, 1/12); pixel error after final rounding is nonzero iff |e| > 0.5
    sigma = math.sqrt(1 / 12)
    p = math.erfc(0.5 / sigma / math.sqrt(2))
    say(f"With every Q = 1 each coefficient is off by at most 0.5, variance 1/12 = {1/12:.5f}. By Parseval each pixel carries "
        f"error variance 1/12 before the final rounding (std {sigma:.4f}).")
    say(f"The final rounding to integers erases the error whenever |e| < 0.5, so the pixel is wrong (by +-1) with probability "
        f"P(|e| > 0.5) = erfc(0.5 / (sigma*sqrt2)) = {p:.5f}. Since each wrong pixel contributes error^2 = 1, MSE ~ {p:.5f}.")
    say("(It is a numerical coincidence that this lands within 0.1% of 1/12 = 0.08333.)\n")
    say("| image | measured MSE | model MSE | max abs error | pixels with error != 0 |\n|---|---|---|---|---|")
    for n, im in imgs.items():
        dec = ref.decode(ref.encode(im, 100))
        diff = dec.astype(int) - im.astype(int)
        say(f"| {n} | {mse(im, dec):.5f} | {p:.5f} | {np.abs(diff).max()} | {100 * np.mean(diff != 0):.2f}% |")
    say("")

    # ------------------------------------------------------------------ 4. floating-point precision
    say("## 4. Transform precision: float64 vs float32 vs fixed-point constants (no quantization)\n")
    say("Each row: forward DCT then inverse DCT of every block of the camera image, error measured before final rounding.\n")
    im = imgs.get("camera", next(iter(imgs.values())))
    blocks = ref.to_blocks(ref.pad_to_blocks(im).astype(np.float64) - 128.0)
    say("| arithmetic | max abs error | rms error |\n|---|---|---|")
    for label, dtype in (("float64", np.float64), ("float32", np.float32)):
        Cm = ref.C.astype(dtype)
        fwd = Cm @ blocks.astype(dtype) @ Cm.T
        back = Cm.T @ fwd @ Cm
        e = back.astype(np.float64) - blocks
        say(f"| {label} | {np.abs(e).max():.2e} | {np.sqrt(np.mean(e ** 2)):.2e} |")
    for bits in (6, 8, 10, 12, 13, 14, 16):
        Cq = np.round(ref.C * 2 ** bits) / 2 ** bits  # constants stored with `bits` fractional bits
        fwd = Cq @ blocks @ Cq.T
        back = Cq.T @ fwd @ Cq
        e = back - blocks
        say(f"| fixed-point constants, {bits:2d} fractional bits | {np.abs(e).max():.2e} | {np.sqrt(np.mean(e ** 2)):.2e} |")
    say("\nReading: float64 is ~1e-13 and float32 ~1e-4 (float32 has ~7 decimal digits, and values up to ~1e3 give ~1e-4 absolute), "
        "both far below the 0.5 needed to flip a pixel. With fixed-point constants the matrix stops being exactly orthonormal, so forward+inverse "
        "no longer cancel. The error falls as bits are added (not strictly monotonically, because each constant is rounded independently): "
        "about 10 bits are needed to keep the worst case below half a grey level, 13-14 bits for ~0.05. "
        "Caveat: this models only the rounding of the CONSTANTS; a real integer implementation also rounds intermediate sums, which adds error.\n")

    # ------------------------------------------------------------------ 5. rounding ties
    say("## 5. Rounding ties\n")
    say("Number of coefficients (out of all, at the given quality) within 1e-9 of an exact x.5 tie, i.e. where float noise decides the rounding "
        "direction unless the tie-break is made deterministic:\n")
    say("Why so many at quality 100? The DCT basis values at positions (u,v) in {0,4}x{0,4} are rational (cos(pi/4)^2 = 1/2), so those four "
        "coefficients per block are exact multiples of 1/8 and land on a .5 tie 1 time in 8: (4/64) * (1/8) = 0.78%, matching the measurement.\n")
    say("| image | quality | coefficients | near-ties | share |\n|---|---|---|---|---|")
    for n in ["camera", "synthetic", "noise"]:
        if n not in imgs:
            continue
        c = ref.forward_dct_blocks(imgs[n]).reshape(-1, 64)
        for q in (100, 90, 50):
            v = c / ref.quant_table(q)
            frac = np.abs((v - np.floor(v)) - 0.5)
            k = int((frac < 1e-9).sum())
            say(f"| {n} | {q} | {v.size} | {k} | {100 * k / v.size:.3f}% |")
    say("")

    with open(os.path.join(ROOT, "results", "numerics.md"), "w") as f:
        f.write("\n".join(OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
