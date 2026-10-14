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
