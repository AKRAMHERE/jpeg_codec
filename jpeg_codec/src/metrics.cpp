#include "jlc/metrics.hpp"

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace jlc {

static void requireSameShape(const Image& a, const Image& b) {
    if (a.width != b.width || a.height != b.height) {
        throw std::invalid_argument("metrics: image sizes differ");
    }
}

double mse(const Image& a, const Image& b) {
    requireSameShape(a, b);
    double sum = 0.0;
    for (size_t i = 0; i < a.pixels.size(); ++i) {
        const double d = static_cast<double>(a.pixels[i]) - static_cast<double>(b.pixels[i]);
        sum += d * d;
    }
    return sum / static_cast<double>(a.pixels.size());
}

int maxAbsError(const Image& a, const Image& b) {
    requireSameShape(a, b);
    int m = 0;
    for (size_t i = 0; i < a.pixels.size(); ++i) {
        const int d = std::abs(static_cast<int>(a.pixels[i]) - static_cast<int>(b.pixels[i]));
        if (d > m) m = d;
    }
    return m;
}

double psnrFromMse(double mseValue) {
    if (mseValue <= 0.0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(255.0 * 255.0 / mseValue);
}

double psnr(const Image& a, const Image& b) { return psnrFromMse(mse(a, b)); }

}  // namespace jlc
