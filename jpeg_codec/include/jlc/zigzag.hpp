// zigzag.hpp - the zig-zag scan order of an 8x8 block.
#pragma once

#include <array>

namespace jlc {

// zigzagOrder()[i] is the row-major index (row*8 + col) of the i-th coefficient
// visited by the zig-zag scan: 0, 1, 8, 16, 9, 2, 3, 10, 17, 24, ...
// i = 0 is the DC coefficient; low frequencies come first, high frequencies last.
const std::array<int, 64>& zigzagOrder();

}  // namespace jlc
