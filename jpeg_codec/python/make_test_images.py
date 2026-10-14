#!/usr/bin/env python3
"""Create the PGM test images used by the codec (no internet needed).

  camera.pgm     512x512 classic "cameraman" photo shipped with scikit-image
  astronaut.pgm  512x512 grayscale version of scikit-image's astronaut photo
  texture.pgm    brick-like natural texture shipped with scikit-image (grayscale)
  noise.pgm      256x256 white noise  (worst case for compression)
  gradient.pgm   256x256 smooth ramp  (best case for compression)
  synthetic.pgm  512x512 deterministic synthetic picture (same generator idea as the C++ bench)

Any PNG/JPG of your own can be converted with:   python3 make_test_images.py --convert photo.png photo.pgm
"""
import argparse
import os
import sys

import numpy as np


def write_pgm(path, arr):
    arr = np.asarray(arr, dtype=np.uint8)
    h, w = arr.shape
    with open(path, "wb") as f:
        f.write(f"P5\n{w} {h}\n255\n".encode())
        f.write(arr.tobytes())


def read_pgm(path):
    with open(path, "rb") as f:
        data = f.read()
    # header: P5 <w> <h> <maxval> followed by exactly one whitespace byte
    tokens, pos = [], 0
    while len(tokens) < 4:
        while data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":
            while data[pos:pos + 1] != b"\n":
                pos += 1
            continue
        start = pos
        while not data[pos:pos + 1].isspace():
            pos += 1
        tokens.append(data[start:pos])
    pos += 1
    assert tokens[0] == b"P5" and int(tokens[3]) == 255, "only 8-bit binary PGM supported"
    w, h = int(tokens[1]), int(tokens[2])
    return np.frombuffer(data, dtype=np.uint8, count=w * h, offset=pos).reshape(h, w).copy()


def to_gray(a):
    a = np.asarray(a)
    if a.ndim == 3:
        a = a[..., :3] @ np.array([0.299, 0.587, 0.114])
    if a.dtype != np.uint8:
        a = np.clip(np.rint(a if a.max() > 1.5 else a * 255.0), 0, 255)
    return a.astype(np.uint8)


def convert(src, dst):
    from PIL import Image
    write_pgm(dst, np.asarray(Image.open(src).convert("L")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--convert", nargs=2, metavar=("SRC", "DST"))
    args = ap.parse_args()
    if args.convert:
        convert(*args.convert)
        return 0

    os.makedirs(args.outdir, exist_ok=True)
    from skimage import data, transform

    cam = to_gray(data.camera())
    write_pgm(f"{args.outdir}/camera.pgm", cam)

    astro = to_gray(data.astronaut())
    write_pgm(f"{args.outdir}/astronaut.pgm", astro)

    try:
        brick = to_gray(data.brick())
        write_pgm(f"{args.outdir}/texture.pgm", brick)
    except Exception as e:  # not every scikit-image build bundles it
        print("skipping texture:", e, file=sys.stderr)

    rng = np.random.default_rng(0)
    write_pgm(f"{args.outdir}/noise.pgm", rng.integers(0, 256, (256, 256)))
    yy, xx = np.mgrid[0:256, 0:256]
    write_pgm(f"{args.outdir}/gradient.pgm", (xx + yy) / 2)

    y, x = np.mgrid[0:512, 0:512]
    v = 110 + 40 * x / 512 + 30 * y / 512 + 25 * np.sin(0.045 * x) * np.cos(0.031 * y)
    v += 12 * np.sin(0.31 * (x + 0.7 * y)) + 22 * (((x // 64) + (y // 64)) % 2 == 0)
    v += rng.integers(-3, 4, v.shape)
    write_pgm(f"{args.outdir}/synthetic.pgm", np.clip(np.rint(v), 0, 255))
    print("wrote test images to", args.outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
