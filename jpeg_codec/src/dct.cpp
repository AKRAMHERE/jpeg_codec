#include "jlc/dct.hpp"

#include <cmath>

namespace jlc {

namespace {

struct DctMatrix {
    double c[64];
    DctMatrix() {
        const double pi = std::acos(-1.0);
        for (int u = 0; u < kBlock; ++u) {
            const double alpha = (u == 0) ? std::sqrt(1.0 / kBlock) : std::sqrt(2.0 / kBlock);
            for (int x = 0; x < kBlock; ++x) {
                c[u * kBlock + x] = alpha * std::cos((2 * x + 1) * u * pi / (2.0 * kBlock));
            }
        }
    }
};

}  // namespace

const double* dctMatrix() {
    static const DctMatrix m;  // built once, thread-safe since C++11
    return m.c;
}

void forwardDCT8x8(const double* in, double* out) {
    const double* C = dctMatrix();
    double tmp[64];
    // Step 1: transform the columns:  T = C * X     T[u][y] = sum_x C[u][x] X[x][y]
    for (int u = 0; u < 8; ++u) {
        for (int y = 0; y < 8; ++y) {
            double s = 0.0;
            for (int x = 0; x < 8; ++x) s += C[u * 8 + x] * in[x * 8 + y];
            tmp[u * 8 + y] = s;
        }
    }
    // Step 2: transform the rows:  Y = T * C^T      Y[u][v] = sum_y T[u][y] C[v][y]
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            double s = 0.0;
            for (int y = 0; y < 8; ++y) s += tmp[u * 8 + y] * C[v * 8 + y];
            out[u * 8 + v] = s;
        }
    }
}

void inverseDCT8x8(const double* in, double* out) {
    const double* C = dctMatrix();
    double tmp[64];
    // T = C^T * Y      T[x][v] = sum_u C[u][x] Y[u][v]
    for (int x = 0; x < 8; ++x) {
        for (int v = 0; v < 8; ++v) {
            double s = 0.0;
            for (int u = 0; u < 8; ++u) s += C[u * 8 + x] * in[u * 8 + v];
            tmp[x * 8 + v] = s;
        }
    }
    // X = T * C        X[x][y] = sum_v T[x][v] C[v][y]
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            double s = 0.0;
            for (int v = 0; v < 8; ++v) s += tmp[x * 8 + v] * C[v * 8 + y];
            out[x * 8 + y] = s;
        }
    }
}

void forwardDCT8x8Naive(const double* in, double* out) {
    const double pi = std::acos(-1.0);
    for (int u = 0; u < 8; ++u) {
        for (int v = 0; v < 8; ++v) {
            const double au = (u == 0) ? std::sqrt(1.0 / 8) : std::sqrt(2.0 / 8);
            const double av = (v == 0) ? std::sqrt(1.0 / 8) : std::sqrt(2.0 / 8);
            double s = 0.0;
            for (int x = 0; x < 8; ++x) {
                for (int y = 0; y < 8; ++y) {
                    s += in[x * 8 + y] * std::cos((2 * x + 1) * u * pi / 16.0) *
                         std::cos((2 * y + 1) * v * pi / 16.0);
                }
            }
            out[u * 8 + v] = au * av * s;
        }
    }
}

}  // namespace jlc
