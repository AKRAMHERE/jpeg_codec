## 1. Coefficient range (raw DCT coefficients, before quantization)

| image | DC min | DC max | max abs AC | max abs AC (category) |
|---|---|---|---|---|
| camera | -996.3 | 930.8 | 669.1 | 10 |
| astronaut | -1024.0 | 1009.5 | 829.0 | 10 |
| texture | -310.1 | 404.8 | 350.1 | 9 |
| synthetic | -328.6 | 791.6 | 79.2 | 7 |
| noise | -218.1 | 270.1 | 358.2 | 9 |

Theory: level-shifted pixels lie in [-128, 127], so |DC| = |sum|/8 <= 8*128 = 1024.
Stronger: the transform is orthonormal, so by Parseval EVERY coefficient (DC or AC) satisfies |Y| <= ||X||_2 <= sqrt(64*128^2) = 1024; the astronaut image reaches it exactly (a pure black block).
Consequence: quantized values fit in int16; a DC *difference* is at most 2048 -> category 12; an AC value is at most 1024 -> category 11; so the 4-bit `size` nibble (max 15) is always sufficient.

## 2. Predicted vs measured MSE

Model: coefficient error ~ uniform(-Q/2, Q/2) -> variance Q^2/12 (valid when coefficients are much larger than Q).
Parseval: pixel-domain MSE = mean over the 64 coefficient positions of their error variance.

| image | quality | measured MSE | model MSE | model / measured |
|---|---|---|---|---|
| camera | 10 | 93.404 | 3719.206 | 39.82 |
| camera | 30 | 48.620 | 1034.087 | 21.27 |
| camera | 50 | 35.739 | 375.036 | 10.49 |
| camera | 75 | 20.187 | 94.997 | 4.71 |
| camera | 90 | 6.014 | 14.992 | 2.49 |
| camera | 95 | 2.012 | 3.754 | 1.87 |
| camera | 100 | 0.083 | 0.083 | 1.00 |
| noise | 10 | 2704.109 | 3719.206 | 1.38 |
| noise | 30 | 878.031 | 1034.087 | 1.18 |
| noise | 50 | 345.046 | 375.036 | 1.09 |
| noise | 75 | 91.423 | 94.997 | 1.04 |
| noise | 90 | 14.810 | 14.992 | 1.01 |
| noise | 95 | 3.817 | 3.754 | 0.98 |
| noise | 100 | 0.085 | 0.083 | 0.98 |

Reading: for noise (large coefficients everywhere) the model is within ~40% at quality 10 and within ~10% from quality 50 upwards. For the natural photo it OVER-estimates massively at low quality (40x at quality 10) because most high-frequency coefficients are tiny: they quantize to zero and their error is just their own small value (< Q/2), far smaller than the uniform-error assumption. The model converges to the truth as Q -> 1 (ratio 1.00 at quality 100). Takeaway: Q^2/12 is an UPPER-bound-style estimate for real images; the dead-zone (zeroed coefficients) is where lossy compression gets its savings.

## 3. Quality 100: pure rounding error

With every Q = 1 each coefficient is off by at most 0.5, variance 1/12 = 0.08333. By Parseval each pixel carries error variance 1/12 before the final rounding (std 0.2887).
The final rounding to integers erases the error whenever |e| < 0.5, so the pixel is wrong (by +-1) with probability P(|e| > 0.5) = erfc(0.5 / (sigma*sqrt2)) = 0.08326. Since each wrong pixel contributes error^2 = 1, MSE ~ 0.08326.
(It is a numerical coincidence that this lands within 0.1% of 1/12 = 0.08333.)

| image | measured MSE | model MSE | max abs error | pixels with error != 0 |
|---|---|---|---|---|
| camera | 0.08344 | 0.08326 | 1 | 8.34% |
| astronaut | 0.07602 | 0.08326 | 2 | 7.60% |
| texture | 0.08327 | 0.08326 | 1 | 8.33% |
| synthetic | 0.08330 | 0.08326 | 1 | 8.33% |
| noise | 0.08499 | 0.08326 | 1 | 8.50% |

## 4. Transform precision: float64 vs float32 vs fixed-point constants (no quantization)

Each row: forward DCT then inverse DCT of every block of the camera image, error measured before final rounding.

| arithmetic | max abs error | rms error |
|---|---|---|
| float64 | 2.70e-13 | 4.75e-14 |
| float32 | 6.10e-05 | 1.04e-05 |
| fixed-point constants,  6 fractional bits | 8.76e+00 | 4.85e+00 |
| fixed-point constants,  8 fractional bits | 2.74e+00 | 1.56e+00 |
| fixed-point constants, 10 fractional bits | 2.73e-01 | 3.49e-02 |
| fixed-point constants, 12 fractional bits | 1.28e-01 | 3.12e-02 |
| fixed-point constants, 13 fractional bits | 5.39e-02 | 3.07e-02 |
| fixed-point constants, 14 fractional bits | 3.44e-02 | 1.88e-02 |
| fixed-point constants, 16 fractional bits | 1.04e-02 | 5.87e-03 |

Reading: float64 is ~1e-13 and float32 ~1e-4 (float32 has ~7 decimal digits, and values up to ~1e3 give ~1e-4 absolute), both far below the 0.5 needed to flip a pixel. With fixed-point constants the matrix stops being exactly orthonormal, so forward+inverse no longer cancel. The error falls as bits are added (not strictly monotonically, because each constant is rounded independently): about 10 bits are needed to keep the worst case below half a grey level, 13-14 bits for ~0.05. Caveat: this models only the rounding of the CONSTANTS; a real integer implementation also rounds intermediate sums, which adds error.

## 5. Rounding ties

Number of coefficients (out of all, at the given quality) within 1e-9 of an exact x.5 tie, i.e. where float noise decides the rounding direction unless the tie-break is made deterministic:

Why so many at quality 100? The DCT basis values at positions (u,v) in {0,4}x{0,4} are rational (cos(pi/4)^2 = 1/2), so those four coefficients per block are exact multiples of 1/8 and land on a .5 tie 1 time in 8: (4/64) * (1/8) = 0.78%, matching the measurement.

| image | quality | coefficients | near-ties | share |
|---|---|---|---|---|
| camera | 100 | 262144 | 2033 | 0.776% |
| camera | 90 | 262144 | 317 | 0.121% |
| camera | 50 | 262144 | 55 | 0.021% |
| synthetic | 100 | 262144 | 2181 | 0.832% |
| synthetic | 90 | 262144 | 400 | 0.153% |
| synthetic | 50 | 262144 | 30 | 0.011% |
| noise | 100 | 65536 | 512 | 0.781% |
| noise | 90 | 65536 | 129 | 0.197% |
| noise | 50 | 65536 | 29 | 0.044% |
