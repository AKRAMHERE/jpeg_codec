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
