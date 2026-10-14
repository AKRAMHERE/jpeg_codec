// dct.hpp - 8x8 two-dimensional DCT-II (orthonormal) and its inverse.
#pragma once

namespace jlc {

constexpr int kBlock = 8;            // block edge length
constexpr int kBlockSize = 64;       // coefficients per block

// The 8x8 DCT matrix C, row-major: C[u*8 + x] = alpha(u) * cos((2x+1) * u * pi / 16)
// with alpha(0) = sqrt(1/8) and alpha(u>0) = sqrt(2/8) = 1/2.
// Because C is orthonormal (C * C^T = I), the inverse transform is just C^T.
const double* dctMatrix();

// Forward 2D DCT of one 8x8 block:  Y = C * X * C^T.
// `in` and `out` are 64 doubles in row-major order. They may not alias.
void forwardDCT8x8(const double* in, double* out);

// Inverse 2D DCT of one 8x8 block:  X = C^T * Y * C.
void inverseDCT8x8(const double* in, double* out);

// Direct implementation of the textbook double-sum definition (O(N^4) per block).
// Slow; exists only so the tests can verify the fast separable version.
void forwardDCT8x8Naive(const double* in, double* out);

}  // namespace jlc
