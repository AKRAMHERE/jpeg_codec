#!/usr/bin/env python3
"""Cross-check the C++ codec against the independent NumPy reference model.

For every test image and several qualities it verifies:
  1. quantized coefficients      C++ (via decode of its own stream) == NumPy analyze()
  2. byte-identical streams      C++ encode == NumPy encode
  3. decoder agreement           NumPy decodes the C++ stream to the same pixels as the C++ decoder,
                                 and C++ decodes the NumPy stream to the same pixels

Run from the project root after `make`:   python3 python/crosscheck.py
Exit status is 0 only if everything matches.
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import jlc_ref as ref  # noqa: E402
from make_test_images import read_pgm, write_pgm  # noqa: E402

JLC = os.path.join(os.path.dirname(__file__), "..", "build", "jlc")
DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def run(*args):
    subprocess.run([JLC, *map(str, args)], check=True, stdout=subprocess.DEVNULL)


def main():
    rng = np.random.default_rng(42)
    cases = []
    for name in ["camera", "gradient", "noise", "synthetic"]:
        p = os.path.join(DATA, f"{name}.pgm")
        if os.path.exists(p):
            img = read_pgm(p)
            # keep the slow pure-Python entropy coder fast: crop the big ones
            cases.append((name, img[:128, :128] if img.shape[0] > 128 else img))
    cases.append(("odd_size_37x53", rng.integers(0, 256, (53, 37)).astype(np.uint8)))
    cases.append(("tiny_1x1", np.array([[200]], dtype=np.uint8)))
    cases.append(("flat_64x64", np.full((64, 64), 77, dtype=np.uint8)))
    cases.append(("black_white_blocks", (((np.add.outer(np.arange(64) // 8, np.arange(64) // 8)) % 2) * 255).astype(np.uint8)))

    failures, checks = 0, 0
    worst_pixel_diff = 0
    with tempfile.TemporaryDirectory() as tmp:
        for name, img in cases:
            src = os.path.join(tmp, "in.pgm")
            write_pgm(src, img)
            for q in (1, 10, 50, 85, 100):
                cpp_stream = os.path.join(tmp, "cpp.jlc")
                cpp_dec = os.path.join(tmp, "cpp_dec.pgm")
                py_stream = os.path.join(tmp, "py.jlc")
                py_dec = os.path.join(tmp, "py_dec.pgm")

                run("encode", src, cpp_stream, "-q", q)
                cpp_bytes = open(cpp_stream, "rb").read()
                py_bytes = ref.encode(img, q)

                zz_cpp, _, h, w = ref.decode_coefficients(cpp_bytes)   # parse C++ stream with Python
                zz_ref = ref.analyze(img, q)
                ok_coef = np.array_equal(zz_cpp, zz_ref)
                ok_bytes = cpp_bytes == py_bytes

                run("decode", cpp_stream, cpp_dec)
                img_cpp = read_pgm(cpp_dec)
                img_py_of_cpp = ref.decode(cpp_bytes)

                open(py_stream, "wb").write(py_bytes)
                run("decode", py_stream, py_dec)
                img_cpp_of_py = read_pgm(py_dec)

                # decoders may differ by 1 grey level only if a value sits exactly on a .5 rounding tie
                d1 = int(np.abs(img_cpp.astype(int) - img_py_of_cpp.astype(int)).max())
                d2 = int(np.abs(img_cpp.astype(int) - img_cpp_of_py.astype(int)).max())
                worst_pixel_diff = max(worst_pixel_diff, d1, d2)

                checks += 4
                bad = [label for label, ok in [("coefficients", ok_coef), ("bytes", ok_bytes),
                                               ("py-decode-of-cpp", d1 == 0), ("cpp-decode-of-py", d2 == 0)] if not ok]
                status = "ok" if not bad else "MISMATCH: " + ", ".join(bad)
                failures += len(bad)
                print(f"{name:20s} {img.shape[1]:4d}x{img.shape[0]:<4d} q={q:3d}  {len(cpp_bytes):7d} B  {status}")

    print(f"\n{checks} comparisons, {failures} mismatches, worst pixel difference between decoders: {worst_pixel_diff}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
