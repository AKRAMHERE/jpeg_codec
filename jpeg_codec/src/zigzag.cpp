#include "jlc/zigzag.hpp"

namespace jlc {

namespace {

std::array<int, 64> buildZigzag() {
    std::array<int, 64> order{};
    int k = 0;
    // Walk the 15 anti-diagonals (row + col = s). Odd diagonals run top-right to
    // bottom-left, even diagonals run bottom-left to top-right.
    for (int s = 0; s <= 14; ++s) {
        const int lo = (s > 7) ? s - 7 : 0;  // smallest row on this diagonal
        const int hi = (s < 7) ? s : 7;      // largest row on this diagonal
        if (s % 2 == 1) {
            for (int row = lo; row <= hi; ++row) order[k++] = row * 8 + (s - row);
        } else {
            for (int row = hi; row >= lo; --row) order[k++] = row * 8 + (s - row);
        }
    }
    return order;
}

}  // namespace

const std::array<int, 64>& zigzagOrder() {
    static const std::array<int, 64> order = buildZigzag();
    return order;
}

}  // namespace jlc
