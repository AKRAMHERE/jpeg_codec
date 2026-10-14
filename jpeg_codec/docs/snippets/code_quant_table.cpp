QTable makeQuantTable(int quality) {
    if (quality < 1) quality = 1;
    if (quality > 100) quality = 100;
    const int scale = (quality < 50) ? 5000 / quality : 200 - 2 * quality;
    QTable q{};
    const QTable& base = baseLuminanceTable();
    for (int i = 0; i < 64; ++i) {
        int v = (base[i] * scale + 50) / 100;
        if (v < 1) v = 1;
        if (v > 255) v = 255;
        q[i] = v;
    }
    return q;
}
