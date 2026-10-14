// metrics.hpp - distortion measures for comparing a reconstruction with its original.
#pragma once

#include "jlc/image.hpp"

namespace jlc {

// Mean squared error over all pixels. Images must have identical dimensions.
double mse(const Image& a, const Image& b);

// Largest absolute pixel difference.
int maxAbsError(const Image& a, const Image& b);

// Peak signal-to-noise ratio in dB for 8-bit images: 10 log10(255^2 / MSE).
// Returns +infinity if the images are identical.
double psnr(const Image& a, const Image& b);

// Same, but computed from an already known MSE.
double psnrFromMse(double mseValue);

}  // namespace jlc
