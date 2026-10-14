# JLC: a JPEG-like grayscale image codec (C++17 + NumPy reference)

Baseline-JPEG-style lossy codec for 8-bit grayscale images:

```
pixels -> pad to 8x8 tiles -> level shift (-128) -> 8x8 DCT-II (orthonormal, Y = C X C^T)
       -> quantize (libjpeg-scaled Annex-K luminance table, quality 1..100)
       -> zig-zag -> DC DPCM + (run,size) AC symbols with ZRL/EOB
       -> per-image canonical Huffman (DC and AC tables) -> bit stream ("JLC1" file)
```

The decoder runs the chain backwards. The only lossy steps are quantization,
rounding to integer pixels and clamping to [0,255].

Not included (on purpose): colour (YCbCr, subsampling), JFIF markers, byte stuffing,
progressive mode. See Chapter 1 of the guide.

## Layout

| path | content |
|------|---------|
| `include/jlc/`, `src/` | library (`libjlc.a`): image I/O, DCT, quantization, zig-zag, RLE, bit I/O, Huffman, metrics, codec |
| `apps/jlc_cli.cpp` | `jlc encode / decode / eval / sweep / info` |
| `apps/jlc_bench.cpp` | benchmark (per-stage timers, staged vs fused, CSV output) |
| `tests/test_main.cpp` | 28 test groups, 32,581 checks, no external framework |
| `python/jlc_ref.py` | independent NumPy implementation of the same file format |
| `python/crosscheck.py` | C++ vs NumPy: 160 comparisons, byte-identical files expected |
| `python/rd_sweep.py` | rate-distortion sweep, comparison with libjpeg (Pillow) |
| `python/numerics_study.py` | coefficient range, error model, float32/float64/fixed-point study, rounding ties |
| `python/page_fault_experiment.py` | glibc mmap-threshold experiment behind the "fused vs staged" result |
| `python/make_figures.py` | figures and table snippets for the guide |
| `slurm/` | bwUniCluster 3.0 job scripts (build+test, benchmark, gprof) |
| `docs/guide.tex` -> `docs/guide.pdf` | the guide (math, intuition, code walkthrough, cluster guide, interview prep) |
| `results/` | machine-independent results (rate-distortion, numerics). `results/sandbox_example/` holds timings from the sandbox where the code was written. **They are not cluster results.** |

## Quick start (any Linux box with g++ >= 9, Python 3 with numpy, pillow, matplotlib, scikit-image)

```bash
make            # builds build/libjlc.a, jlc, jlc_bench, jlc_tests
make test       # unit + property tests
python3 python/make_test_images.py
./build/jlc encode data/camera.pgm cam.jlc -q 50
./build/jlc decode cam.jlc cam_out.pgm
./build/jlc eval   data/camera.pgm -q 50      # size, ratio, MSE, PSNR
python3 python/crosscheck.py                  # C++ == NumPy, byte for byte
```

## bwUniCluster 3.0

See Chapter 13 of the guide for the full walk-through. Short version:

```bash
rsync -av --exclude build --exclude results/sandbox_example jpeg_codec/ <user>@uc3.scc.kit.edu:~/jpeg_codec/
ssh <user>@uc3.scc.kit.edu
cd ~/jpeg_codec
sbatch slurm/01_build_test.sbatch        # dev_cpu: compile + tests
bash slurm/submit_all.sh                 # benchmarks on cpu_il and cpu
sbatch slurm/03_profile.sbatch           # gprof
```

Numbers you quote about the cluster must be ones you measured there.

## Rebuilding the guide

```bash
python3 python/make_figures.py
cd docs && pdflatex guide.tex && pdflatex guide.tex
```

Snippets in `docs/snippets/` are generated from the code and results. Do not edit them by hand.
