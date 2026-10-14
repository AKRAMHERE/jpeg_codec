// quant.hpp - quantization tables and the (only) lossy step of the codec.
#pragma once

#include <array>
#include <cmath>

namespace jlc {

using QTable = std::array<int, 64>;  // natural (row-major) order, entries in [1, 255]

// The standard JPEG luminance table (ITU-T T.81, Annex K.1), "quality 50".
const QTable& baseLuminanceTable();

// Builds the table for quality in [1, 100] (values outside are clamped), using the
// same scaling rule as libjpeg:
//   scale = (q < 50) ? 5000 / q : 200 - 2 q
//   Q[i]  = clamp((base[i] * scale + 50) / 100, 1, 255)   (integer arithmetic)
QTable makeQuantTable(int quality);

// Round to nearest integer, ties (x.5) go up. A tiny epsilon makes the tie-break
// deterministic: values that are mathematically exactly x.5 but come out of floating
// point as x.4999999999999 or x.5000000000001 (different summation orders, FMA, a
// different compiler, NumPy vs C++, CPU vs GPU ...) all round the same way.
// 1e-9 is ~10^5 times larger than the accumulated rounding noise of an 8x8 DCT (~1e-14) and
// ~10^9 times smaller than any quantizer step, so it never changes a real decision.
constexpr double kTieEpsilon = 1e-9;
inline double roundHalfUp(double x) { return std::floor(x + 0.5 + kTieEpsilon); }

// Quantize one coefficient: round(c / q).
inline int quantizeValue(double coefficient, int q) {
    return static_cast<int>(roundHalfUp(coefficient / q));
}

}  // namespace jlc
