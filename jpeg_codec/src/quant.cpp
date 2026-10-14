#include "jlc/quant.hpp"

namespace jlc {

const QTable& baseLuminanceTable() {
    static const QTable t = {
        16, 11, 10, 16, 24,  40,  51,  61,   //
        12, 12, 14, 19, 26,  58,  60,  55,   //
        14, 13, 16, 24, 40,  57,  69,  56,   //
        14, 17, 22, 29, 51,  87,  80,  62,   //
        18, 22, 37, 56, 68,  109, 103, 77,   //
        24, 35, 55, 64, 81,  104, 113, 92,   //
        49, 64, 78, 87, 103, 121, 120, 101,  //
        72, 92, 95, 98, 112, 100, 103, 99};
    return t;
}

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

}  // namespace jlc
