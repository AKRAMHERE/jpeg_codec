constexpr double kTieEpsilon = 1e-9;
inline double roundHalfUp(double x) { return std::floor(x + 0.5 + kTieEpsilon); }

// Quantize one coefficient: round(c / q).
inline int quantizeValue(double coefficient, int q) {
    return static_cast<int>(roundHalfUp(coefficient / q));
}
